CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp
CUDAFLAGS= -arch=compute_61 -code=sm_61
LIBS= -lcusparse

spmv: spmv.o serial_tools.o
	$(NVCC) -o spmv  $(CUDAFLAGS) $(LIBS) --compiler-options $(CXXFLAGS) spmv.o serial_tools.o

spmv.o : spmv.cu serial_tools.h
	$(NVCC) -c $(CUDAFLAGS)  spmv.cu
serial_tools.cpp: serial_tools.h

clean:
	rm -rf *.o