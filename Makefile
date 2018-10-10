NVCC=nvcc

GPUARCHFLAGS= -gencode arch=compute_61,code=sm_61
#GPUARCHFLAGS += -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 #uncomment for compatability
COMPILEFLAGS = -g --compiler-options -fopenmp
LIBCOMPILEFLAGS = --compiler-options -fPIC

VPATH = src include
INCLUDEPATH = include
BUILDPATH = build

TESTLIBS= -lcusparse


spmvtest: $(BUILDPATH)/lib/libspmv.so
	$(NVCC) -I $(INCLUDEPATH) $(COMPILEFLAGS) $(GPUARCHFLAGS) -L$(BUILDPATH)/lib -lspmv $(TESTLIBS) -o $(BUILDPATH)/test/spmvtest src/test/compare_spmv.cu
	cp include/spmv.h build/include/spmv.h

$(BUILDPATH)/lib/libspmv.so : ricsr.o utils.o
	$(NVCC) -I $(INCLUDEPATH) $(COMPILEFLAGS) $(LIBCOMPILEFLAGS) $(GPUARCHFLAGS) -shared -o $(BUILDPATH)/lib/libspmv.so ricsr.o utils.o

ricsr.o : ricsr.cu utils.h 
	$(NVCC) -c -I $(INCLUDEPATH) $(COMPILEFLAGS) $(LIBCOMPILEFLAGS) $(GPUARCHFLAGS) src/ricsr.cu

utils.o : utils.cpp utils.h
	$(NVCC) -c -I $(INCLUDEPATH) $(COMPILEFLAGS) $(LIBCOMPILEFLAGS) $(GPUARCHFLAGS) src/utils.cpp

clean:
	rm -rf *.o