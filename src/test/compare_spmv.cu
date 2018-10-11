#include <stdio.h>
#include <cuda.h>
#include "cusparse_v2.h"
#include "spmv.h"

#define VERBOSE 1
#define BLOCKSIZE 32

struct cusparse_data move_cusparse_data_to_device(struct coo arr_coo, float * x) {
	cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
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
		printf("cuSPARSE: Device memory allocation failed\n");
	}

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
		printf("cuSPARSE: Initial memory copy failed\n");
	}

	// put into a struct and return
	struct cusparse_data ret;
	ret.d_csrRowPtr = d_csrRowPtr;
	ret.d_colInds = d_colInds;
	ret.d_vals = d_vals;
	ret.d_y = d_y;
	ret.d_x = d_x;
	ret.m = m;
	ret.n = n;
	ret.nnz = nnz;
	return ret;
}

 void execute_cusparse_spmv(struct cusparse_data container, float * res) {
 	snprintf(logbuf, 512, "setting up cuSPARSE environment");
 	printlog(1);
 	cudaError_t cudaStat1;
	// cusparse setup
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	cusparseCreate(&handle);
	cusparseMatDescr_t descriptor = 0;

	float alpha = 1.0;
	float beta = 0.0;

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

	snprintf(logbuf, 512, "calling cuSPARSE spmv");
	printlog(1);

	// execute spmv!
	status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, container.m, container.n, container.nnz, &alpha, descriptor, container.d_vals, container.d_csrRowPtr, container.d_colInds, container.d_x, &beta, container.d_y);
	if(status != CUSPARSE_STATUS_SUCCESS) {
		printf("cuSPARSE SpMV function execution failed! exiting\n");
		return;
	}

	snprintf(logbuf, 512, "copying result from GPU");
	printlog(1);

	// copy result back
	cudaStat1 = cudaMemcpy(res, container.d_y, container.m*sizeof(float), cudaMemcpyDeviceToHost);
	if(cudaStat1 != cudaSuccess) {
		printf("cuSPARSE: Error copy result back to host, exiting\n");
		return;
	}
	
	// free memory
	cudaFree(container.d_csrRowPtr);
	cudaFree(container.d_colInds);
	cudaFree(container.d_vals);
	cudaFree(container.d_y);
	cudaFree(container.d_x);
}

int main(int argc, char * argv[]) {
	// ensure usage
	if(argc != 2) {
		printf("Usage: spmv [matrix market file]\n");
		return 1;
	}
	printf("\n");
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
	int ribwidth = system_info.sharedsize/sizeof(float);
	snprintf(logbuf, 512, "converting COO to RICSR");
	printlog(1);
	struct ricsr arr_ricsr = coo_to_ricsr(arr_coo, ribwidth, BLOCKSIZE);
	
	// generate and slice an x vector
	snprintf(logbuf, 512, "generating random x");
	printlog(1);
	srand(100);
	float * x = gen_rand_x(arr_coo.n, 0.0, 2.0);
	snprintf(logbuf, 512, "slicing x vector");
	printlog(1);
	float ** slicedx = slice_x(x, arr_coo.n, ribwidth, 32);
//	free(x);

	// move data to gpu
	float balance_arr[] = {1.0};
	snprintf(logbuf, 512, "building row_ranges");
	printlog(1);

	snprintf(logbuf, 512, "moving data to gpu");
	printlog(1);
	struct gpu_data dcontainer = move_data_to_devices(arr_ricsr, slicedx, balance_arr, system_info.numdevices, BLOCKSIZE);

	// allocate result vectors
	float * cusparse_res = (float *)malloc(sizeof(float)*arr_coo.m);
	float * ricsr_res = (float *)malloc(sizeof(float)*round_val(arr_coo.m, BLOCKSIZE));

	// do the cusparse and store it there
	snprintf(logbuf, 512, "moving data to GPU for cuSPARSE");
	printlog(1);
	struct cusparse_data cusparse_container = move_cusparse_data_to_device(arr_coo, x);
	snprintf(logbuf, 512, "executing cuSPARSE SpMV");
	printlog(1);
	execute_cusparse_spmv(cusparse_container, cusparse_res);

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