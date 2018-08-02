# ricsr-spmv

Implement the Ribboned-CSR (RCSR) data structure for sparse matrices and execute sparse matrix-vector multiplication on multiple CUDA devices

Sorry there's no makefile at this time; I'm the only one who uses it.
I currently compile with:
nvcc -o spmv -arch compute_61 -code sm_61 -lcusparse --compiler-options -fopenmp spmv.cu serial_tools.cpp


