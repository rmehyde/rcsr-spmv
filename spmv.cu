#include <cuda.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cusparse_v2.h"
#include "serial_tools.h"

// define constants
#define VERBOSE 1
#define PRINT_ARRS 1 // print the dense arrays
// DO NOT GO CHANING THIS SHIT WILLY NILLY!
// ELLs must be coalesced according to block size!
#define BLOCKSIZE 32
#define SHAREDMEM 12288

struct sysinfo {
	int numdevices; // number of devices in the system
	int sharedsize; // minimum constant memory size in bytes of any device on the system
	int warpsize; // minimum warpsize in threads of any device on the system
};

char * logbuf;
struct timespec wallstart;
clock_t cpustart;

__global__ void ComputeRibbon(int * rowsts, int * cols, float * vals, float * d_xslice, int ribwidth, int ribid, int fullribwidth, int * row_ranges, float * ribres) {
	int thread = threadIdx.x;
	// slices should be padded to warp size
	// ******** FIX IT TO BLOCK SIZE ********
	__shared__ float x_slice[SHAREDMEM];
	int max_iters = ribwidth/blockDim.x;
	if(ribwidth%blockDim.x != 0)
		max_iters++;
	for(int i=0; i<max_iters; i++) {
//		printf("moving index %d of slice to shared memory (max_iters = %d)\n", i*blockDim.x+thread, max_iters);
		x_slice[i*blockDim.x+thread] = d_xslice[i*blockDim.x+thread];
	}
	__syncthreads();

	// compute the range of rows assigned to this thread
	int startrow = row_ranges[thread];
	int endrow = row_ranges[thread+1];

	double rowsum;
	for(int row=startrow; row < endrow; row++) {
		rowsum = 0.0f;
		for(int s = rowsts[row]; s < rowsts[row+1]; s++) {
			rowsum += x_slice[cols[s]]*vals[s];
		}
		ribres[row] = (float)rowsum;
	}
}

// trivial reductive sum
__global__ void VVSum(float * a, float * b) {
	a[blockIdx.x*BLOCKSIZE+threadIdx.x] += b[blockIdx.x*BLOCKSIZE+threadIdx.x];
}

int ** build_row_ranges(struct ricsr mtx, int blocksize) {
	int ** row_ranges = (int **)malloc(mtx.numrib * sizeof(int *));
	for(int r=0; r<mtx.numrib; r++) {
		row_ranges[r] = (int *)malloc((blocksize+1) * sizeof(int));
		struct csr rib = *mtx.csrs[r];
		int nnz_per_thread = rib.nnz/blocksize;
		//go through each row, deciding whether or not to assign it
		int cur_thread = 0;
		int last_ind = 0;
		row_ranges[r][0] = 0;
		for(int row=0; row<rib.m; row++) {
			if(cur_thread >= blocksize) {
				continue;
			}
			else if(rib.rowsts[row]-last_ind > nnz_per_thread) {
				row_ranges[r][cur_thread+1] = row;
				last_ind = rib.rowsts[row];
				cur_thread++;
			}
		}
		// hopefully we wont have any leftover threads but if we do
		while(cur_thread < blocksize) {
			row_ranges[r][cur_thread+1] = rib.m-1;
			cur_thread++;
		}
		// just for good measure
		row_ranges[r][blocksize] = rib.m-1;
	}
	return row_ranges;
}

// examines the system and creates a RELL format from a supplied matrix market filename
struct sysinfo get_system_info() {
	// get num devices and min constant memory size and min warp size and initialize devices
	int numdevices;
	unsigned int sharedsize = INT_MAX;
	int warpsize = INT_MAX;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&numdevices);
	for(int i=0; i<numdevices; i++) {
		cudaSetDevice(i);
		cudaFree(0);
		cudaGetDeviceProperties(&prop, i);
		if(prop.sharedMemPerBlock < sharedsize) {
			sharedsize = prop.sharedMemPerBlock;
		}
		if(prop.warpSize < warpsize) {
			warpsize = prop.warpSize;
		}
	}
	struct sysinfo ret;
	ret.numdevices = numdevices;
	ret.sharedsize = sharedsize;
	ret.warpsize = warpsize;
	return ret;
}

// using balance array moves ribbons into devices, returns array listing last ribbon of each device
struct gpu_data move_data_to_devices(struct ricsr mtx, float ** slicedx, float * balance_arr, int ** row_ranges, int numdevices, int blocksize) {
	// generate ribbon assignments to devices
	int * devribs = (int *)malloc(sizeof(int)*(numdevices+1));
	devribs[0] = 0;
	float sum = 0.0f;
	for(int i=1; i<numdevices; i++) {
		sum += balance_arr[i];
		devribs[i] = (int)round(sum*(mtx.numrib));
	}
	devribs[numdevices] = mtx.numrib;

	//setup gpu data container
	struct gpu_data ret;
	ret.x_slices = (float **)malloc(mtx.numrib*sizeof(float *));
	ret.numrib = mtx.numrib;
	ret.devribs = devribs;
	ret.csrs = (struct csr **)malloc(mtx.numrib*sizeof(struct csr *));
	ret.m = mtx.m;
	ret.m_padded = (*mtx.csrs[0]).m_padded;
	ret.row_ranges = (int **)malloc(mtx.numrib * sizeof(int *));

	// in parallel copy x slices and ribbons to each device
	# pragma omp parallel num_threads(numdevices)
	{
	int devid = omp_get_thread_num();
	cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
	cudaStat1 = cudaSetDevice(devid);
	if(! (cudaStat1 == cudaSuccess)) {
		printf("ERROR: failed to set device to %d\n", devid);
		printf("%s : %s\n", cudaGetErrorName(cudaStat1), cudaGetErrorString(cudaStat1));
	}

	// for each ribbon to be assigned to this device
	for(int r=devribs[devid]; r<devribs[devid+1]; r++) {
		// initialize variables
		struct csr * ribcontainer = (struct csr *)malloc(sizeof(struct csr));
		int * cols;
		int * rowsts;
		float * vals;
		float * x_slice;
		int * rib_row_ranges;
		int m = (*mtx.csrs[r]).m;
		int n = (*mtx.csrs[r]).n;
		int nnz = (*mtx.csrs[r]).nnz;
		// allocate memory on gpu
		cudaStat1 = cudaMalloc(&cols, nnz * sizeof(int));
		cudaStat2 = cudaMalloc(&vals, nnz * sizeof(float));
		cudaStat3 = cudaMalloc(&rowsts, (m+1) * sizeof(int));
		cudaStat4 = cudaMalloc(&x_slice, mtx.ribwidth * sizeof(float));
		cudaStat5 = cudaMalloc(&rib_row_ranges, (blocksize+1)*sizeof(int));
		if (!(cudaStat1 == cudaSuccess && cudaStat2 == cudaSuccess && cudaStat3 == cudaSuccess && cudaStat4 == cudaSuccess && cudaStat5 == cudaSuccess)) {
			printf("ERROR: failed to allocate device memory for ribbon number %d on device %d\n", r, devid);
			break;
		}
		// copy data to gpu
		cudaStat1 = cudaMemcpy(cols, (*mtx.csrs[r]).cols, nnz*sizeof(int), cudaMemcpyHostToDevice);
		cudaStat2 = cudaMemcpy(vals, (*mtx.csrs[r]).vals, nnz*sizeof(float), cudaMemcpyHostToDevice);
		cudaStat3 = cudaMemcpy(rowsts, (*mtx.csrs[r]).rowsts, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaStat4 = cudaMemcpy(x_slice, slicedx[r], n * sizeof(float), cudaMemcpyHostToDevice);
		cudaStat5 = cudaMemcpy(rib_row_ranges, row_ranges[r], (blocksize+1) * sizeof(int), cudaMemcpyHostToDevice);
		if (!(cudaStat1 == cudaSuccess && cudaStat2 == cudaSuccess && cudaStat3 == cudaSuccess && cudaStat4 == cudaSuccess && cudaStat5 == cudaSuccess)) {
			printf("ERROR: failed to copy ribbon number %d to device %d\n got errors:\n %s\n%s\n%s\n", r, devid, cudaGetErrorString(cudaStat1), cudaGetErrorString(cudaStat2), cudaGetErrorString(cudaStat3));
			break;
		}

		// assign to structures
		ribcontainer->m = m;
		ribcontainer->n = n;
		ribcontainer->nnz = (*mtx.csrs[r]).nnz;
		ribcontainer->cols = cols;
		ribcontainer->vals = vals;
		ribcontainer->rowsts = rowsts;
		ribcontainer->m_padded = (*mtx.csrs[0]).m_padded;

		ret.csrs[r] = ribcontainer;
		ret.x_slices[r] = x_slice;
		ret.row_ranges[r] = rib_row_ranges;
		ret.fullribwidth = mtx.ribwidth;
	}
	}
	return ret;
}

void ricsr_spmv(int numdevices, struct gpu_data dcontainer, int m, float * result) {
	cudaError_t cudaStat1;
	// first lets allocate device memory for the intermediate ribbon results
	// should be a totalribbons-sized array full of pointers to m-sized arrays
	float * ribbon_results[dcontainer.numrib];
	cudaStat1 = cudaDeviceSynchronize();
	for(int d=0; d<numdevices; d++) {
		cudaStat1 = cudaSetDevice(d);
		if(cudaStat1 != cudaSuccess)
			printf("ERROR: failed to set device to %d\n", d);
		for(int r=dcontainer.devribs[d]; r<dcontainer.devribs[d+1]; r++) {
			cudaStat1 = cudaMalloc(&(ribbon_results[r]), dcontainer.m_padded*sizeof(float));
			if(cudaStat1 != cudaSuccess) {
				printf("ERROR: filed to allocated memory on device %d for ribbon %d result array\n", d, r);
			}
		}
	}

	// parallel across devices, solve ribbons one at a time
	#pragma omp parallel num_threads(numdevices)
	{
		int m_padded = dcontainer.m_padded;
		int devid = omp_get_thread_num();
		cudaSetDevice(devid);
		// for each ribbon
		for(int r=dcontainer.devribs[devid]; r<dcontainer.devribs[devid+1]; r++) {
			struct csr devrib = *(dcontainer.csrs[r]);
			int ribwidth = devrib.n;

			if(VERBOSE) {
//				printf("\nrowsts: %p\ncols: %p\nvals: %p\nx_slices[r]: %p\nribbon_results[r]: %p\n\n", devrib.rowsts, devrib. cols, devrib.vals, dcontainer.x_slices[r], ribbon_results[r]);
			}


			ComputeRibbon<<<1,BLOCKSIZE>>>(devrib.rowsts, devrib.cols, devrib.vals, dcontainer.x_slices[r], ribwidth, r, dcontainer.fullribwidth, dcontainer.row_ranges[r], ribbon_results[r]);
		}
		// now sum all the ribbons on this device
		// ROWS MUST BE PADDED TO BLOCK SIZE!!!!!!!
		if(!(dcontainer.m_padded%BLOCKSIZE==0)) {
			printf("WARNING: Number of rows not padded to block size. You MUST fix this!!! Leave now! Do it!\n");
		}
		for(int r=dcontainer.devribs[devid+1]-1; r>dcontainer.devribs[devid]; r--) {
			VVSum<<<m_padded/BLOCKSIZE, BLOCKSIZE>>>(ribbon_results[r-1], ribbon_results[r]);
		}
		// bring it back home
		cudaStat1 = cudaDeviceSynchronize();
		#pragma omp barrier
	}

	// now we can combine our results across devices
	// do we have enough ribbons already allocated on device 0?
	if(dcontainer.devribs[1] >= numdevices) {
		for(int d=1; d<numdevices; d++) {
			cudaStat1 = cudaMemcpy(ribbon_results[d], ribbon_results[dcontainer.devribs[d]], m*sizeof(float), cudaMemcpyDeviceToDevice);
			if(cudaStat1 != cudaSuccess) {
				printf("ERROR: Failed to copy device %d result to device 0\n", d);
			}
		}
		// ADD ALPHA, BETA HERE BY WRITING A DIFFERENT KERNEL FUNCTION
		for(int d=numdevices-1; d>0; d--) {
			int m_padded = dcontainer.m_padded;
			VVSum<<<m_padded/BLOCKSIZE, BLOCKSIZE>>>(ribbon_results[d-1], ribbon_results[d]);
		}
	}
	else {
		printf("ERROR: Device 1 did not have enough ribbon result arrays already allocated to move the other device results to. You should \
			change this else statement from this annoying message to an actual solution to this problem that involves reallocating some memory \
			on that device. Thank you.\n");
	}
	// finally copy our result to the desired place on host
	cudaStat1 = cudaMemcpy(result, ribbon_results[0], m*sizeof(float), cudaMemcpyDeviceToHost);
}

int free_csr_dev(struct csr csr_d) {
	cudaFree(csr_d.cols);
	cudaFree(csr_d.vals);
	cudaFree(&csr_d);
	return 0;
}

int free_ricsr_dev(struct ricsr ricsr_d) {
	for(int r=0; r<ricsr_d.numrib; r++) {
		free_csr_dev(*ricsr_d.csrs[r]);
	}
	cudaFree(ricsr_d.csrs);
	cudaFree(&ricsr_d);
	return 0;
}

void execute_cusparse_spmv(struct coo arr_coo, float * x, float * res) {
	// cusparse setup
	cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	cusparseCreate(&handle);
	cusparseMatDescr_t descriptor = 0;

	// allocate device memory for CSR array, y, and x
	int * d_csrRowPtr;
	int * d_colInds;
	float * d_vals;
	float * d_y;
	float * d_x;
	cudaStat1 = cudaMalloc(&d_csrRowPtr, arr_coo.m*sizeof(int));
	cudaStat2 = cudaMalloc(&d_colInds, arr_coo.nnz*sizeof(int));
	cudaStat3 = cudaMalloc(&d_vals, arr_coo.nnz*sizeof(float));
	cudaStat4 = cudaMalloc(&d_y, arr_coo.m*sizeof(float));
	cudaStat5 = cudaMalloc(&d_x, arr_coo.n*sizeof(float));
	if(!((cudaStat1 == cudaSuccess) && (cudaStat2 == cudaSuccess) && (cudaStat3 == cudaSuccess) && (cudaStat4 == cudaSuccess) && (cudaStat5 == cudaSuccess))) {
		printf("cuSPARSE: Device memory allocation failed, exiting\n");
		return;
	}

	float alpha = 1.0;
	float beta = 0.0;

	// grab metavars, convert array to csr and free coo
	int m = arr_coo.m;
	int n = arr_coo.n;
	int nnz = arr_coo.nnz;
	struct csr arr_csr = coo_to_csr(arr_coo);
	free_coo(arr_coo);

	// copy csr, y, and x to device
	cudaStat1 = cudaMemcpy(d_csrRowPtr, arr_csr.rowsts, m*sizeof(int), cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_colInds, arr_csr.cols, nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_vals, arr_csr.vals, nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	if(!((cudaStat1 == cudaSuccess) && (cudaStat2 == cudaSuccess) && (cudaStat3 == cudaSuccess) && (cudaStat4 == cudaSuccess))) {
		printf("cuSPARSE: Initial memory copy failed, exiting\n");
		return;
	}

	// initialize cusparse
	status = cusparseCreate(&handle);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("cuSPARSE environment initialization failed, exiting\n");
		return;
	}
	cusparseCreateMatDescr(&descriptor);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("cuSPARSE matrix descriptor initialization failed, exiting\n");
		return;
	}
	cusparseSetMatType(descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descriptor, CUSPARSE_INDEX_BASE_ZERO);

	// execute spmv!
	status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descriptor, d_vals, d_csrRowPtr, d_colInds, d_x, &beta, d_y);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("cuSPARSE SpMV function execution failed! exiting\n");
		return;
	}

	// copy result back
	cudaStat1 = cudaMemcpy(res, d_y, m*sizeof(float), cudaMemcpyDeviceToHost);
	if(cudaStat1 != cudaSuccess) {
		printf("cuSPARSE: Error copy result back to host, exiting\n");
		return;
	}
}

// prints all messages with v = 0 and those with v = 1 if VERBOSE
void printlog(int v) {
	if(!(v && !VERBOSE)) {
		struct timespec wall;
		clock_gettime(CLOCK_REALTIME, &wall);
		float walldiff = (float)(wall.tv_sec - wallstart.tv_sec);
		walldiff += ((float)(wall.tv_nsec - wallstart.tv_nsec))/1000000000;
		float clockdiff = ((float)(clock()-cpustart))/CLOCKS_PER_SEC;
		printf("(W: %.4fs | P: %.4fs) : ", walldiff, clockdiff);
		printf(logbuf);
		printf("\n");
	}
}

int main(int argc, char * argv[]) {
	// ensure usage
	if(argc != 2) {
		printf("Usage: spmv [matrix market file]\n");
		return 1;
	}
	// initialize timing and message variables
	logbuf = (char *)malloc(512 * sizeof(char));
	clock_gettime(CLOCK_REALTIME, &wallstart);
	cpustart = clock();

	// read file
	snprintf(logbuf, 512, "reading file %s", argv[1]);
	printlog(1);
	struct coo arr_coo = real_mm_to_coo(argv[1]);
	struct sysinfo system_info = get_system_info();

	// convert to ricsr
	if (system_info.sharedsize/sizeof(float) != SHAREDMEM) {
		printf("ERROR: the shared memory size defined is incorrect! You must fix this!\n");
	}
	int ribwidth = SHAREDMEM;
	snprintf(logbuf, 512, "converting COO to RICSR");
	printlog(1);
	struct ricsr arr_ricsr = coo_to_ricsr(arr_coo, ribwidth, BLOCKSIZE);
	
	// generate and slice an x vector
	snprintf(logbuf, 512, "generating random x");
	printlog(1);
	float * x = gen_rand_x(arr_coo.n, 0.0, 2.0);
	snprintf(logbuf, 512, "slicing x vector");
	printlog(1);
	float ** slicedx = slice_x(x, arr_coo.n, ribwidth, 32);

	// move data to gpu
	float balance_arr[] = {1.0};
	snprintf(logbuf, 512, "building row_ranges");
	printlog(1);
	int ** row_ranges = build_row_ranges(arr_ricsr, BLOCKSIZE);

	snprintf(logbuf, 512, "moving data to gpu");
	printlog(1);
	struct gpu_data dcontainer = move_data_to_devices(arr_ricsr, slicedx, balance_arr, row_ranges, system_info.numdevices, BLOCKSIZE);

	// allocate result vectors
	float * cusparse_res = (float *)malloc(sizeof(float)*arr_coo.m);
	float * ricsr_res = (float *)malloc(sizeof(float)*round_val(arr_coo.m, BLOCKSIZE));

	// do the cusparse and store it there
	snprintf(logbuf, 512, "executing cuSPARSE matrix-vector multiplication");
	printlog(1);
	execute_cusparse_spmv(arr_coo, x, cusparse_res);

	cudaError_t cudaStat1 = cudaDeviceSynchronize();
	if(!cudaStat1 == cudaSuccess) {
		int devid;
		cudaGetDevice(&devid);
		printf("ERROR: failed to synchronize device %d after cuSPARSE\n", devid);
	}
	snprintf(logbuf, 512, "executing RICSR matrix-vector multiplication");
	printlog(1);
	ricsr_spmv(system_info.numdevices, dcontainer, dcontainer.m, ricsr_res);

	// free stuff
//	free_coo(arr_coo);

	// compare results
	snprintf(logbuf, 512, "comparing results");
	printlog(1);
	float maxerr = 0.01f;
	if (!(arrs_are_same(ricsr_res, cusparse_res, arr_coo.m, maxerr))) {
		snprintf(logbuf, 512, "cuSPARSE achieved a different result from RICSR");
		printlog(0);
	}
	else {
		snprintf(logbuf, 512, "RICSR algorithm result matches CUSPARSE within max error of %.2f%\n", 100*maxerr);
		printlog(0);
	}
	if(ribwidth % BLOCKSIZE != 0) {
		printf("ERROR: you didn't align your ribbon width to block size!!! Go fix it!\n");
	}
}

