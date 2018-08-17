#ifndef SPMV_H
#define SPMV_H

// define errors
#define FILE_ERROR 4
#define MARKET_FORMAT_ERROR 5

/* STRUCT DEFINITIONS */
struct coo {
	int m;
	int n;
	int nnz;
	int * rows; //size nnz
	int * cols; //size nnz
	float * vals; //size nnz
};

struct csr {
	int m;
	int n;
	int nnz;
	int m_padded;
	int * rowsts; //size m
	int * cols; //size nnz
	float * vals; //size nnz
};

struct ell {
	int m;
	int n;
	int nnz;
	int en;
	int * cols;
	float * vals;
};

struct ricsr {
	struct csr** csrs;
	int numrib;
	int ribwidth;
	int m;
};

struct gpu_data {
	struct csr ** csrs;
	int numrib;
	int * devribs;
	float ** x_slices;
	int m;
	int m_padded;
	int ** row_ranges;
	int fullribwidth;
};

struct sysinfo {
	int numdevices; // number of devices in the system
	int sharedsize; // minimum constant memory size in bytes of any device on the system
	int warpsize; // minimum warpsize in threads of any device on the system
};

/* FUNCTION DECLARATIONS */
// from utils
float * trivial_dense(float *, float *, int, int);
float * trivial_coo(int *, int *, float *, float *, int, int, int);
float * trivial_csr(int *, int *, float *, float *, int, int, int);
float * trivial_ell(int *, float *, float *, int, int, int);
struct csr dense_to_csr(float *, int, int);
struct coo dense_to_coo(float *, int, int);
struct ell dense_to_ell(float *, int, int);
float * csr_to_dense(int *, int *, float *, int, int, int);
struct csr make_random_csr(int, int, int);
int * make_row_lens(int, int);
int * random_sorted_list(int, int);
int compare_int(const void*, const void*);
struct coo real_mm_to_coo(const char *);
void print_coo_format(struct coo);
struct ricsr coo_to_ricsr(struct coo, int, int);
float ** slice_x(float *, int, int, int);
float * gen_rand_x(int, float, float);
int * count_csr_rowlens(struct csr);
void print_csr(struct csr);
void free_rell(struct rell);
struct csr coo_to_csr(struct coo);
void free_coo(struct coo);
void print_arr(float *, int, int);
int coo_is_sorted(struct coo);
int arrs_are_same(float *, float *, int, float);
int round_val(int, int);
void print_coo(struct coo);
void print_int_arr(int *, int, int);
void print_ricsr_stats(struct ricsr);

// from ricsr
void ricsr_spmv(int, struct gpu_data, int, float *);
struct gpu_data move_data_to_devices(struct ricsr, float **, float *, int, int);
struct sysinfo get_system_info();
int free_csr_dev(struct csr);
int free_ricsr_dev(struct ricsr);
void printlog(int);

// variable declarations
char * logbuf;
struct timespec wallstart;
clock_t cpustart;

#endif /* SPMV_H */
 
