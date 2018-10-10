# Ribboned Compressed Sparse Row Sparse Matrix-Vector Multiplication (RiCSR SpMV)
This library tackles the SpMV problem on GPUs using data structures and algorithms that effortlessly scale across multiple devices attached to a single host. The input matrix subsected into ribbons, or vertical `m x (n/r)` slices of the underlying dense matrix, where ribbon width r is automatically set according to the shared memory available in the GPUs. The dense input vector is likewise sliced according to the same structure. The result is that the impact of erratic accesses to the **x** vector can be mitigated by reads from shared rather than global memory.

### Building and Running
`make` the project and it will build a shared library as well as a test application. Then simply add the library path `build/lib` to your LD_LIBRARY_PATH variable or move `libspmv.so` to wherever you keep your shared libraries.
The test application can be run with `build/test/spmvtest matrixfile.mtx` where `matrixfile.mtx` is a real Matrix Market exchange format file. This will run a verbose single execution of SpMV using a randomly generated **x** vector, with both the RiCSR algorithm and NVIDIA's cuSPARSE library for comparison.

### Using the library
While complete documentation is not available at this time, I encourage you to look at the `src/test/compare_spmv.cu` program to get a feel for usage.
