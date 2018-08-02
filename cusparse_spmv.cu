#include <cuda.h>
#include "cusparse_v2.h"
#include <stdio.h>
extern "C" {
	#include "serial_tools.h"
}

#define PRINT_DENSE 1

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
	free_coo(arr_coo);

	// print our array
	if(PRINT_DENSE) {
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

int main(int argc, char * argv[]) {

	// check number of args
	if(argc != 2) {
		printf("Usage: spmv [matrix market file]\n");
		return 1;
	}

	// read mm file
	struct coo arr_coo = real_mm_to_coo(argv[1]);
	int m = arr_coo.m;
	int n = arr_coo.n;

	// get an x and allocate space for y
	float * x = gen_rand_x(n, 0.0, 10.0);
	float * y = (float *)malloc(m*sizeof(float));

	execute_cusparse_spmv(arr_coo, x, y);

	printf("Hey I really did the CUSPARSE! Results below:\nX:\n");
	print_arr(x, 1, n);
	printf("Y:\n");
	print_arr(y, 1, m);


}
























