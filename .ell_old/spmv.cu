#include <cuda.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cusparse_v2.h"
#include "serial_tools.h"

// define constants
#define TIME 1	// whether to time functions
#define VERBOSE 1 // show verbose data
#define PRINT_ARRS 1 // print the dense arrays
// DO NOT GO CHANING THIS SHIT WILLY NILLY!
// ELLs must be coalesced according to block size!
#define BLOCK_SIZE 32

/*
TIMING SHIT
clock_t start = clock();
double time_used = ((double)(clock()-start)) / CLOCKS_PER_SEC;
*/

// WHY DOES CONSTANT MEMORY NEED TO BE DECLARED BEFORE THE STRUCT???

// allocate 65536 bytes constant memory
// assumes sizeof(float) == 4
// check later if this isn't kosher
__constant__ float constX[16384];

struct sysinfo {
	int numdevices; // number of devices in the system
	int cmsize; // minimum constant memory size in bytes of any device on the system
	int warpsize; // minimum warpsize in threads of any device on the system
};

// pretty trivial does one row
// is it strided correctly? im sleepy
__global__ void MatMult(int * cols, float * vals, float * x, float * res, int n) {
	float my_sum = 0.0;
	__shared__ float threadres[BLOCK_SIZE];
	for(int i=0; i<n/BLOCK_SIZE; i++) {
		my_sum += x[cols[n*blockIdx.x+i*BLOCK_SIZE+threadIdx.x]]*vals[n*blockIdx.x+i*BLOCK_SIZE+threadIdx.x];
	}
	threadres[threadIdx.x] = my_sum;
	// make sure everyones done and reduce all results
	__syncthreads();
	// iters should be a power of 2 already
	int iters = (int)log((float)BLOCK_SIZE);
	// reduce our block results to threadres[0]
	while(iters > 0) {
		if(threadIdx.x > iters/2 && threadIdx.x < iters) {
			threadres[threadIdx.x - iters/2] += threadres[threadIdx.x];
		}
		iters /= 2;
	}
	// now set this row value in the result for this ribbon
	if(threadIdx.x == 0) {
		res[blockIdx.x] = threadres[0];
	}
}

// trivial reductive sum
__global__ void VVSum(float * a, float * b) {
	a[blockIdx.x*BLOCK_SIZE+threadIdx.x] += b[blockIdx.x*BLOCK_SIZE+threadIdx.x];
}

// examines the system and creates a RELL format from a supplied matrix market filename
// also slices input x and points the DESTINATION pointer to it
struct sysinfo get_system_info() {
	// get num devices and min constant memory size and min warp size and initialize devices
	int numdevices;
	int cmsize = INT_MAX;
	int warpsize = INT_MAX;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&numdevices);
	for(int i=0; i<numdevices; i++) {
		cudaSetDevice(i);
		cudaFree(0);
		cudaGetDeviceProperties(&prop, i);
		if(prop.totalConstMem < cmsize) {
			cmsize = prop.totalConstMem;
		}
		if(prop.warpSize < warpsize) {
			warpsize = prop.warpSize;
		}
	}
	struct sysinfo ret;
	ret.numdevices = numdevices;
	ret.cmsize = cmsize;
	ret.warpsize = warpsize;
	return ret;
}

// using balance array moves ribbons into devices, returns array listing last ribbon of each device
struct gpu_data move_rell_to_devices(struct rell mtx, float ** slicedx, float * balance_arr, int numdevices) {
	int * devribs = (int *)malloc(sizeof(int)*(numdevices+1));
	devribs[0] = 0;
	// assign devribs
	for(int i=1; i<numdevices; i++) {
		devribs[i] = (int)round(balance_arr[i]*(mtx.numrib));
	}
	// clarification device d should do ribbons [devribs[d], devribs[d+1])
	devribs[numdevices] = mtx.numrib;
	// were gonna return the device pointers
	// NOTE: for now lets go ahad and push X into device memory, then move it from global to constant memory on the device
	// another option is to move it straight from the host to constant memory with each ribbon
	struct gpu_data ret;
	ret.x_slices = (float **)malloc(mtx.numrib*sizeof(float *));
	ret.numrib = mtx.numrib;
	ret.devribs = devribs;
	ret.rib_ens = (int *)malloc(mtx.numrib*sizeof(int));
	ret.ells = (struct ell **)malloc(mtx.numrib*sizeof(struct ell *));
	ret.m_padded = mtx.m_padded;

	// in parallel copy x slices and ribbons to each device



	# pragma omp parallel num_threads(numdevices)
	{
	int devid = omp_get_thread_num();

	cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4;
	// lets make a 
	// for each ribbon to be assigned to this device


	// test if we can copy from something else

	for(int r=devribs[devid]; r<devribs[devid+1]; r++) {
		// is this array properly initialized?
		struct ell * ribcontainer = (struct ell *)malloc(sizeof(struct ell));
		int * cols;
		float * vals;
		float * x_slice;
		int m = (*mtx.ells[r]).m;
		int n = (*mtx.ells[r]).n;
		cudaStat1 = cudaMalloc(&cols, m * n * sizeof(int *));
		cudaStat2 = cudaMalloc(&vals, m * n * sizeof(float *));
		cudaStat4 = cudaMalloc(&x_slice, mtx.ribwidth * sizeof(float));
		if (!(cudaStat1 == cudaSuccess && cudaStat2 == cudaSuccess && cudaStat3 == cudaSuccess && cudaStat4 == cudaSuccess)) {
			printf("ERROR: failed to allocate device memory for ribbon number %d on device %d\n", r, devid);
			break;
		}
		cudaStat1 = cudaMemcpy(cols, (*mtx.ells[r]).cols, m*n*sizeof(int), cudaMemcpyHostToDevice);
		cudaStat2 = cudaMemcpy(vals, (*mtx.ells[r]).vals, m * n * sizeof(float), cudaMemcpyHostToDevice);
		cudaStat3 = cudaMemcpy(x_slice, slicedx[r], mtx.ribwidth * sizeof(float), cudaMemcpyHostToDevice);
		if (!(cudaStat1 == cudaSuccess && cudaStat2 == cudaSuccess && cudaStat3 == cudaSuccess)) {
			printf("ERROR: failed to copy ribbon number %d to device %d\n got errors:\n %s\n%s\n%s\n", r, devid, cudaGetErrorString(cudaStat1), cudaGetErrorString(cudaStat2), cudaGetErrorString(cudaStat3));
			break;
		}

		ribcontainer->m = m;
		ribcontainer->n = n;
		ribcontainer->nnz = (*mtx.ells[r]).nnz;
		ribcontainer->en = (*mtx.ells[r]).en;
		ribcontainer->cols = cols;
		ribcontainer->vals = vals;
		ret.rib_ens[r] = (*mtx.ells[r]).en;


		ret.ells[r] = ribcontainer;
		ret.x_slices[r] = x_slice;
	}
	}
	return ret;
}

void execute_cusparse_spmv(struct coo arr_coo, float * x, float * res) {
	// cusparse
	cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
	cusparseStatus_t status;

	// make cusparse handle and descriptor
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
		printf("Device memory allocation failed, exiting\n");
		return;
	}

	// make alpha and beta: lame
	// WHY ARE THESE ALLOWED TO BE IN HOST MEMORY???
	float alpha = 1.0;
	float beta = 0.0;

	// grab metavars, convert array to csr and free coo
	int m = arr_coo.m;
	int n = arr_coo.n;
	int nnz = arr_coo.nnz;
	struct csr arr_csr = coo_to_csr(arr_coo);
	if(PRINT_ARRS) {
		printf("converted to CSR, printing CSR\n");
		print_csr(arr_csr);
	}
	free_coo(arr_coo);

	// print our array
	if(PRINT_ARRS) {
		printf("printing dense array\n");
		float * dense = csr_to_dense(arr_csr.rowsts, arr_csr.cols, arr_csr.vals, m, n, nnz);
		print_arr(dense, m, n);
	}

	// copy csr, y, and x to device
	cudaStat1 = cudaMemcpy(d_csrRowPtr, arr_csr.rowsts, m*sizeof(int), cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_colInds, arr_csr.cols, nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_vals, arr_csr.vals, nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	if(!((cudaStat1 == cudaSuccess) && (cudaStat2 == cudaSuccess) && (cudaStat3 == cudaSuccess) && (cudaStat4 == cudaSuccess))) {
		printf("Initial memory copy failed, exiting\n");
		return;
	}

	// initialize cusparse
	status = cusparseCreate(&handle);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("CuSPARSE environment initialization failed, exiting\n");
		return;
	}
	cusparseCreateMatDescr(&descriptor);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("CuSPARSE matrix descriptor initialization failed, exiting\n");
		return;
	}
	cusparseSetMatType(descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descriptor, CUSPARSE_INDEX_BASE_ZERO);

	// execute spmv!
	status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descriptor, d_vals, d_csrRowPtr, d_colInds, d_x, &beta, d_y);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("CuPSARSE SpMV function execution failed! exiting\n");
		return;
	}

	// copy result back
	cudaStat1 = cudaMemcpy(res, d_y, m*sizeof(float), cudaMemcpyDeviceToHost);
	if(cudaStat1 != cudaSuccess) {
		printf("Error copy result back to host, exiting\n");
		return;
	}
}

void rell_spmv(int numdevices, struct gpu_data dcontainer, int m, float * result) {
	cudaError_t cudaStat1;
	// first lets allocate device memory for the intermediate ribbon results
	// should be a totalribbons-sized array full of pointers to m-sized arrays
	float * ribbon_results[dcontainer.numrib];
	cudaStat1 = cudaDeviceSynchronize();
	for(int d=0; d<numdevices; d++) {
		cudaStat1 = cudaSetDevice(d);
		if(cudaStat1 != cudaSuccess)
			printf("ERROR: failed to set device to %d\n", d);
		for(int r=dcontainer.devribs[d]; r<=dcontainer.devribs[d+1]; r++) {
			cudaStat1 = cudaMalloc(&(ribbon_results[r]), m*sizeof(float));
			if(cudaStat1 != cudaSuccess) {
				printf("ERROR: filed to allocated memory on device %d for ribbon %d result array\n", d, r);
			}
		}
	}
	// okay great now we can solve each device's ribbons one at a time
	#pragma omp parallel num_threads(numdevices)
	{
		int m;
		//cudaError_t cudaStat1;
		int devid = omp_get_thread_num();
		cudaSetDevice(devid);
		// for each ribbon
		for(int r=dcontainer.devribs[devid-1]; r<=dcontainer.devribs[devid]; r++) {
			m = dcontainer.ells[r]->m;
			// copy our x slice to the constant memory
			cudaMemcpyToSymbol(constX, dcontainer.x_slices[r], dcontainer.rib_ens[r]*sizeof(float));
			MatMult<<<dcontainer.m_padded, BLOCK_SIZE>>>(dcontainer.ells[r]->cols, dcontainer.ells[r]->vals, constX, ribbon_results[r], dcontainer.rib_ens[r]);
		}
		// now sum all the ribbons on this device
		// ROWS MUST BE PADDED TO BLOCK SIZE!!!!!!!
		if(!dcontainer.m_padded%BLOCK_SIZE!=0) {
			printf("WARNING: Number of rows not padded to block size. You MUST fix this!!! Leave now! Do it!\n");
		}
		for(int r=dcontainer.devribs[devid+1]; r>dcontainer.devribs[devid]; r--) {
			VVSum<<<m/BLOCK_SIZE, BLOCK_SIZE>>>(ribbon_results[r-1], ribbon_results[r]);
		}
		// make sure everythings done
		cudaStat1 = cudaDeviceSynchronize();
		// and that each device is done
		#pragma omp barrier
	}

	// now we can combine our results across devices
	// do we have enough ribbons already allocated on device 0?

	if(dcontainer.devribs[1] >= numdevices) {
		for(int d=1; d<numdevices; d++) {
			cudaStat1 = cudaMemcpy(ribbon_results[d], ribbon_results[dcontainer.devribs[d]], m*sizeof(float), cudaMemcpyDeviceToDevice);
			if(cudaStat1 != cudaSuccess) {
				printf("ERROR: Failed to copy device %d result to device 0\n");
			}
		}
		// ADD ALPHA, BETA HERE BY WRITING A DIFFERENT KERNEL FUNCTION
		for(int d=numdevices-1; d>0; d--) {
			VVSum<<<m/BLOCK_SIZE, BLOCK_SIZE>>>(ribbon_results[d-1], ribbon_results[d]);
		}
	}
	else {
		printf("ERROR: Device 1 did not have enough ribbon result arrays already allocated to move the other device results to. You should change this else statement from this annoying message to an actual solution to this problem that involves reallocating some memory on that device. Thank you.\n");
	}
	// finally copy our result to the desired place on host
	cudaStat1 = cudaMemcpy(result, ribbon_results[0], m*sizeof(float), cudaMemcpyDeviceToHost);
}

int free_ell_dev(struct ell ell_d) {
	cudaFree(ell_d.cols);
	cudaFree(ell_d.vals);
	cudaFree(&ell_d);
	return 0;
}

int free_rell_dev(struct rell rell_d) {
	for(int r=0; r<rell_d.numrib; r++) {
		free_ell_dev(*rell_d.ells[r]);
	}
	cudaFree(rell_d.ells);
	cudaFree(&rell_d);
	return 0;
}

int main(int argc, char * argv[]) {

	if(argc != 2) {
		printf("Usage: spmv [matrix market file]\n");
		return 1;
	}
	struct coo arr_coo = real_mm_to_coo(argv[1]);
	struct sysinfo system_info = get_system_info();

	// convert coo to rell, allocating memory in function
	if(system_info.cmsize != sizeof(constX)) {
		printf("WARNING: min size of constant memory is %d but allocated %lu bytes for x vector\n", system_info.cmsize, sizeof(constX));
	}
	int ribwidth = system_info.cmsize/sizeof(float);
	struct rell arr_rell = coo_to_rell(arr_coo, ribwidth, system_info.warpsize);
	free_coo(arr_coo);
	print_rell_stats(arr_rell);



/*

	/// FIX THIS JESUS CHRIST YOU IDIOT
	// generate a random x vector and then slice it, allocating memory along the way
	float * x = gen_rand_x(arr_coo.n, 0.0, 2.0);
	float ** slicedx = slice_x(x, arr_coo.n, ribwidth, system_info.warpsize);

	// move ribboned matrix into device memory
	// we start with one device
	float balance_arr[] = {1.0};
	struct gpu_data devdataptr = move_rell_to_devices(arr_rell, slicedx, balance_arr, system_info.numdevices);

	// allocate result vectors
	float * cusparse_res = (float *)malloc(sizeof(float)*arr_coo.m);
	float * rell_res = (float *)malloc(sizeof(float)*round_val(arr_coo.m, BLOCK_SIZE));

	if(VERBOSE) {
		printf("Printing Cols of COO\n");
		print_int_arr(arr_coo.cols, 1, arr_coo.nnz);
	}

	// do the cusparse and store it there
	execute_cusparse_spmv(arr_coo, x, cusparse_res);

	cudaError_t cudaStat1 = cudaDeviceSynchronize();
	// do the rell spmv
	rell_spmv(system_info.numdevices, devdataptr, devdataptr.m_padded, rell_res);

	// compare results
	if (!(arrs_are_same(rell_res, cusparse_res, arr_coo.m))) {
		printf("CUSPARSE achieved different result from RELL\n");
	}

*/
}

