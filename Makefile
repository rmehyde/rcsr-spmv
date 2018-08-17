CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp
CUDAFLAGS= -gencode arch=compute_61,code=sm_61
#CUDAFLAGS += -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 #uncomment for compatability
LIBS= -lcusparse


spmv: spmv.o serial_tools.o
	$(NVCC) -o spmv  $(CUDAFLAGS) $(LIBS) --compiler-options $(CXXFLAGS) spmv.o serial_tools.o

spmv.o : spmv.cu serial_tools.h
	$(NVCC) -c $(CUDAFLAGS)  spmv.cu
serial_tools.cpp: serial_tools.h

clean:
	rm -rf *.o