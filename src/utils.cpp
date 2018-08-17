#include "utils.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

/* FUNCTIONS */

float * trivial_dense(float * arr, float * x, int m, int n) {
	float * res = (float *)calloc(m, sizeof(float));
	for(int i=0; i<m; i++) {
		for(int j=0; j<n; j++) {
			res[i] += arr[i*n+j] * x[j];
		}
	}
	return res;
}

float * trivial_coo(int * rows, int * cols, float * vals, float * x, int m, int n, int nnz) {
	float * res = (float *)calloc(m, sizeof(float));
	for(int s=0; s<nnz; s++) {
		int i = rows[s];
		int j = cols[s];
		res[i] += vals[s]*x[j];
	}
	return res;
}

float * trivial_csr(int * rowsts, int * cols, float * vals, float * x, int m, int n, int nnz) {
	float * res = (float *)calloc(m, sizeof(float));
	int s;
	for(int i=1; i<m; i++) {
		for(s = rowsts[i]; s<rowsts[i+1]; s++) {
			int j = cols[s];
			res[i] += vals[s]*x[j];
		}
	}
	return res;
}

// computes zeroes
float * trivial_ell(int * cols, float * vals, float * x, int m, int n, int en) {
	float * res = (float *)calloc(m, sizeof(float));
	int j;
	for(int i=0; i<m; i++) {
		for(int c=0; c<en; c++) {
			j = cols[i*m+c];
			res[i] += vals[i*m+c] * x[j];
		}
	}
	return res;
}

struct coo dense_to_coo(float * arr, int m, int n) {
	// first pass to count elts
	int nnz = 0;
	for(int i=0; i<m; i++) {
		for(int j=0; j<n; j++) {
			if(arr[i*n+j]!=0) {
				nnz++;
			}
		}
	}
	//allocate results
	int * rows = (int *)malloc(sizeof(float)*nnz); 
	int * cols = (int *)malloc(sizeof(float)*nnz);
	float * vals = (float *)malloc(sizeof(float)*nnz);
	//fill format
	int s;
	for(int i=0; i<m; i++) {
		for(int j=0; j<n; j++) {
			if(arr[i*n+j] != 0) {
				rows[s] = i;
				cols[s] = j;
				vals[s] = arr[i*n+j];
				s++;
			}
		}
	}
	// put in struct and return
	struct coo res;
	res.rows = rows;
	res.cols = cols;
	res.vals = vals;
	res.m = m;
	res.n = n;
	res.nnz = nnz;
	return res;
}

struct csr dense_to_csr(float * arr, int m, int n) {
	// first pass to count elts
	int nnz = 0;
	for(int i=0; i<m; i++) {
		for(int j=0; j<n; j++) {
			if(arr[i*n+j] != 0) {
				nnz++;
			}
		}
	}
	// allocate results
	int * rowsts = (int *)malloc(sizeof(float)*m);
	int * cols = (int *)malloc(sizeof(float)*nnz);
	float * vals = (float *)malloc(sizeof(float)*nnz);
	//fill format
	int s;
	int first;
	for(int i=0; i<m; i++) {
		first = 1;
		for(int j=0; j<n; j++) {
			if(arr[i*n+j] != 0) {
				if(first) {
					rowsts[i] = s;
					first = 0;
				}
				cols[s] = j;
				vals[s] = arr[i*n+j];
				s++;
			}
		}
	}
	//put in struct and return
	struct csr res;
	res.rowsts = rowsts;
	res.cols = cols;
	res.vals = vals;
	res.m = m;
	res.n = n;
	res.nnz = nnz;
	return res;
}

struct ell dense_to_ell(float * arr, int m, int n) {
	//count longest row and nnz
	int maxlen = 0;
	int nnz = 0;
	int rowlen;
	for(int i=0; i<m; i++) {
		rowlen = 0;
		for(int j=0; j<n; j++) {
			if(arr[i*n+j] != 0) {
				nnz++;
				rowlen++;
			}
		}
		if(rowlen > maxlen) {
			maxlen = rowlen;
		}
	}
	//alocate memory
	int * cols = (int *)malloc(sizeof(float)*maxlen*m);
	float * vals = (float *)malloc(sizeof(float)*maxlen*m);
	//fill format
	for(int i=0; i<m; i++) {
		int ej = 0;
		for(int j=0; j<n; j++) {
			if(arr[i*n+j] != 0) {
				cols[i*maxlen+ej] = j;
				vals[i*maxlen+ej] = arr[i*n+j];
				ej++;
			}
		}
		for(int f=ej; f<maxlen; f++) {
			cols[i*maxlen+ej] = 0;
			vals[i*maxlen+ej] = 0;
		}
	}
	// put in struct and return
	struct ell res;
	res.cols = cols;
	res.vals = vals;
	res.m = m;
	res.n = n;
	res.nnz = nnz;
	res.en = maxlen;
	return res;
}

float * csr_to_dense(int * rowsts, int * cols, float * vals, int m, int n, int nnz) {
	float * arr = (float *)calloc(m*n, sizeof(float));
	for(int i=0; i<m-1; i++) {
		int eltsinrow = rowsts[i+1]-rowsts[i];
		for(int e=0; e<eltsinrow; e++) {
			arr[i*n+cols[rowsts[i]+e]] = vals[rowsts[i]+e];
		}
	}
	int eltsinrow = nnz-rowsts[m-1];
	for(int e=0; e<eltsinrow; e++) {
		arr[n*(m-1)+cols[rowsts[m-1]+e]] = vals[rowsts[m-1]+e];
	}
	return arr;
}

int coo_is_sorted(struct coo arr) {
	for(int s=1; s<arr.nnz; s++) {
		// if we're still on the same col
		if(arr.cols[s]-arr.cols[s-1] == 0) {
			// we should be increasing in row
			if(arr.rows[s]-arr.rows[s-1] < 1) {
				printf("COO not sorted (1): s = %d, col = %d, last col = %d, row = %d, last row = %d\n", s, arr.cols[s], arr.cols[s-1], arr.rows[s], arr.rows[s-1]);
				return 0;
			}
		}
		// otherwise col increase should be positive
		else if(arr.cols[s]-arr.cols[s-1] < 1) {
			printf("COO not sorted (1): s = %d, col = %d, last col = %d\n", s, arr.cols[s], arr.cols[s-1]);
			return 0;
		}
	}
	return 1;
}

/* converts a a coo formatted matrix to csr formatted
COO MUST BE SORTED BY COLUMN THEN ROW for accurate
mapping */
struct csr coo_to_csr(struct coo arr) {
	// check if sorted properly
	if(!coo_is_sorted(arr)) {
		printf("ERROR: you have supplied an unsorted COO array. Please sort your matrix by column then row\n");
	}
	int * rowcts = (int *)calloc(arr.m, sizeof(int));
	int * rowptrs = (int *)calloc(arr.m+1, sizeof(int));
	for(int s=0; s<arr.nnz; s++) {
		rowptrs[arr.rows[s]+1]++;
	}
	for(int i=1; i<arr.m; i++) {
		rowptrs[i] += rowptrs[i-1];
	}
	int * cols = (int *)malloc(arr.nnz*sizeof(int));
	float * vals = (float *)malloc(arr.nnz*sizeof(float));
	for(int s=0; s<arr.nnz; s++) {
		int m = arr.rows[s];
		cols[rowptrs[m]+rowcts[m]] = arr.cols[s];
		vals[rowptrs[m]+rowcts[m]] = arr.vals[s];
		rowcts[m]++;
	}
	struct csr ret;
	ret.rowsts = rowptrs;
	ret.cols = cols;
	ret.vals = vals;
	ret.m = arr.m;
	ret.n = arr.n;
	ret.nnz = arr.nnz;
	return ret;
}


/* Must be a real Matrix Market file in COO format
   WARNING: Little to no error checking here!!!  */
// MALLOCS rows, cols, vals and generates coo
// ASSUMES 1-BASED INDICES

// WARNING: DOES NOT HANDLE EXPONENTIAL NOTATION
struct coo real_mm_to_coo(const char * filename) {
	// open file for reading
	FILE * fp;
	fp = fopen(filename, "r");
	if(fp == NULL) {
		printf("ERROR: failed to open file\n");
	}
	char * buff = (char *)malloc(4098);
	size_t len = 0;
	// skip banner
	buff[0] = '%';
	while(buff[0] == '%') {
		getline(&buff, &len, fp);
	}
	// read size
	int * m = (int *)malloc(sizeof(int));
	int * n = (int *)malloc(sizeof(int));
	int * nnz = (int *)malloc(sizeof(int));
	sscanf(buff, "%d %d %d", m, n, nnz);
	printf("m = %d\nn= %d\nnnz = %d\n\n", *m, *n, *nnz);
	// allocate memory for results
	int * rows = (int *)malloc(sizeof(int) * (*nnz));
	int * cols = (int *)malloc(sizeof(int) * (*nnz));
	float * vals = (float *)malloc(sizeof(float) * (*nnz));
	// store results in arrays
	for(int s=0; s < *nnz; s++) {
		fscanf(fp, "%d", &rows[s]);
		fscanf(fp, "%d", &cols[s]);
		fscanf(fp, "%f", &vals[s]);
		// change base index to zero
		rows[s]--;
		cols[s]--;
	}

	/*
	if(!feof(fp)) {
		printf("WARNING: expected end of file. please check matrix market format. Printing rest of file below\n");
		char buff[255];
		fscanf(fp, "%s", buff);
		printf(buff);
	}
	*/

	
	fclose(fp);
	// put into coo format
	struct coo ret;
	ret.m = *m;
	ret.n = *n;
	ret.nnz = *nnz;
	ret.rows = rows;
	ret.cols = cols;
	ret.vals = vals;
	// free buffer and metadata
	free(buff);
	free(m);
	free(n);
	free(nnz);
	return ret;
}

// prints out underlying tables of coo for data verification
void print_coo_format(struct coo arr) {
	printf("  S    I    J    V  \n");
	for(int s=0; s<10; s++) {
		printf("%d %d %d %.3f\n", s, arr.rows[s], arr.cols[s], arr.vals[s]);
	}
	for(int i=0; i<3; i++) {
		printf("     .          .          .          .     \n");
	}
	for(int s=arr.nnz-10; s<arr.nnz; s++) {
		printf("%d %d %d %.3f\n", s, arr.rows[s], arr.cols[s], arr.vals[s]);
	}
	printf("\n\n");
}


struct ricsr coo_to_ricsr(struct coo arr, int ribbon_size, int blocksize) {
	// rounded down
	int num_ribbons = arr.n/ribbon_size;
	// allocate memory for pointers to ribbons
	struct csr ** csrs = (struct csr **)malloc(sizeof(struct csr *)*(num_ribbons+1));
	// s scans along entries
	int s = 0;
	// for each ribbon of full size
	int r;
	for(r=0; r<num_ribbons; r++) {
		int maxj = (r+1)*ribbon_size;
		// get the length of each row
		int * rowlens = (int *)calloc(arr.m, sizeof(int));
		int nnz = 0;
		for(int rs=s; arr.cols[rs]<maxj && rs < arr.nnz; rs++) {
			rowlens[arr.rows[rs]]++;
			nnz++;
		}
		// pad rows
		int m_padded = arr.m/blocksize;
		if (arr.m % blocksize != 0) {
			m_padded = (arr.m/blocksize+1)*blocksize;
		}
		// copy to rowsts
		int * rowsts = (int *)calloc(m_padded+1, sizeof(int));
		int sum = 0;
		for(int i=1; i<=arr.m; i++) {
			rowsts[i] = rowlens[i-1]+sum;
			sum += rowlens[i-1];
		}
		// pad rowsts
		for(int i=arr.m; i<=m_padded; i++) {
			rowsts[i] = nnz;
		}
		// allocate memory for this ribbon
		struct csr * ribptr = (struct csr*)malloc(sizeof(struct csr));
		int * curcols = (int *)calloc(nnz, sizeof(int));
		float * curvals = (float *)calloc(nnz,  sizeof(float));
		// fill values of arrays
		int row;
		int rowst;
		int rowlen;
		int ind;
		while(arr.cols[s]<maxj && s < arr.nnz) {
			row = arr.rows[s];
			rowst = rowsts[row];
			rowlen = rowsts[row+1]-rowsts[row];
			ind = rowst+rowlen-rowlens[row];
			rowlens[row]--;
			curcols[ind] = arr.cols[s]-r*ribbon_size;
			curvals[ind] = arr.vals[s];
			s++;
		}
		free(rowlens);
		// put into csr
		ribptr->cols = curcols;
		ribptr->vals = curvals;
		ribptr->rowsts = rowsts;
		ribptr->m = arr.m;
		ribptr->n = ribbon_size;
		ribptr->nnz = nnz;
		ribptr->m_padded = m_padded;
		csrs[r] = ribptr;
	}

	// remainder ribbon
	if(arr.n%ribbon_size != 0) {
		int maxj = arr.n;
		int * rowlens = (int *)calloc(arr.m, sizeof(int));
		int nnz = 0;
		for(int rs=s; rs < arr.nnz && arr.cols[rs]<maxj; rs++) {
			rowlens[arr.rows[rs]]++;
			nnz++;
		}
		// pad rows
		int m_padded = arr.m/blocksize;
		if (arr.m % blocksize != 0) {
			m_padded = (arr.m/blocksize+1)*blocksize;
		}
		// copy to rowsts
		int * rowsts = (int *)calloc(m_padded+1, sizeof(int));
		int sum = 0;
		for(int i=1; i<=arr.m; i++) {
			rowsts[i] = rowlens[i-1]+sum;
			sum += rowlens[i-1];
		}
		// pad rowsts
		for(int i=arr.m; i<=m_padded; i++) {
			rowsts[i] = nnz;
		}
		// allocate memory for this ribbon
		struct csr * ribptr = (struct csr*)malloc(sizeof(struct csr));
		int * curcols = (int *)calloc(nnz, sizeof(int));
		float * curvals = (float *)calloc(nnz,  sizeof(float));
		// fill values of arrays
		int row;
		int rowst;
		int rowlen;
		int ind;
		while(s < arr.nnz && arr.cols[s]<maxj) {
			row = arr.rows[s];
			rowst = rowsts[row];
			rowlen = rowsts[row+1]-rowsts[row];
			ind = rowst+rowlen-rowlens[row];
			rowlens[row]--;
			curcols[ind] = arr.cols[s]-num_ribbons*ribbon_size;
			curvals[ind] = arr.vals[s];
			s++;
		}
		free(rowlens);
		// put into csr
		ribptr->cols = curcols;
		ribptr->vals = curvals;
		ribptr->rowsts = rowsts;
		ribptr->m = arr.m;
		ribptr->n = arr.n%ribbon_size;
		ribptr->nnz = nnz;
		ribptr->m_padded = m_padded;
		csrs[r] = ribptr;
		num_ribbons++;
	}
	// put all this into a ricsr
	struct ricsr ret;
	ret.csrs = csrs;
	ret.numrib = num_ribbons;
	ret.ribwidth = ribbon_size;
	ret.m = arr.m;
	return ret;
}

float ** slice_x(float * arr, int n, int ribbon_size, int coalsz) {
	if(ribbon_size%coalsz != 0) {
		printf("ERROR: supplied ribbon width is not matched to coalescing size!\n");
	}
	int num_ribbons = n/ribbon_size;
	float ** ret = (float **)malloc(sizeof(float *)*(num_ribbons+1));
	int s = 0;
	// process full-sized slices
	for(int i=0; i<num_ribbons; i++) {
		float * slice = (float *)malloc(sizeof(float)*ribbon_size);
		while(s<i*ribbon_size) {
			slice[s-i*ribbon_size] = arr[s];
			s++;
		}
		ret[i] = slice;
	}
	// process last ribbon
	if(n%ribbon_size != 0) {
		int rem = n%ribbon_size;
		if(rem%coalsz != 0) {
			rem = coalsz*(rem/coalsz+1);
		}
		float * slice = (float *)malloc(sizeof(float)*rem);
		rem = n%ribbon_size;
		for(int i=0; i<rem; i++) {
			slice[i] = arr[s+i];
		}
		ret[num_ribbons] = slice;
	}
	return ret;
}

float * gen_rand_x(int size, float min, float max) {
	float * ret = (float *)malloc(sizeof(float)*size);
	float randnum;
	for(int i=0; i<size; i++) {
		randnum = ((float)rand()) / (float)RAND_MAX;
		ret[i] = min + randnum * (max-min);
	}
	return ret;
}

int * count_csr_rowlens(struct csr arr) {
	// allocate mem for result
	int * rowlens = (int *)malloc(arr.m*sizeof(int));
	rowlens[0] = arr.rowsts[0];
	rowlens[arr.m-1] = arr.nnz-arr.rowsts[arr.m-1];
	for(int i=1; i<arr.m-1; i++) {
		rowlens[i] = arr.rowsts[i]-arr.rowsts[i-1];
	}
	return rowlens;
}

void print_arr(float * arr, int m, int n) {
	for(int i=0; i<m; i++) {
		printf("[");
		for(int j=0; j<n-1; j++) {
			printf("%.2f, ", arr[i*n+j]);
		}
		printf("%.2f]\n", arr[(i+1)*n-1]);
	}
	printf("\n");
}

void print_int_arr(int * arr, int m, int n) {
	for(int i=0; i<m; i++) {
		printf("[");
		for(int j=0; j<n-1; j++) {
			printf("%d, ", arr[i*n+j]);
		}
		printf("%d]\n", arr[(i+1)*n-1]);
	}
	printf("\n");
}

void print_csr(struct csr arr) {
	printf("column indices:\n");
	print_int_arr(arr.cols, 1, arr.nnz);
	printf("vals: \n");
	print_arr(arr.vals, 1, arr.nnz);
	printf("row pointers:\n");
	print_int_arr(arr.rowsts, 1, arr.m);
}

void free_ell(struct ell arr) {
	free(arr.cols);
	free(arr.vals);
}

void free_coo(struct coo arr) {
	free(arr.rows);
	free(arr.cols);
	free(arr.vals);
}

void free_csr(struct csr arr) {
	free(arr.rowsts);
	free(arr.cols);
	free(arr.vals);
}

int arrs_are_same(float * a, float * b, int size, float maxerr) {
	for(int i=0; i<size; i++) {
		if (!(abs(a[i] /b[i]) > (1.0f-maxerr) && abs(a[i]/b[i]) < (1.0f+maxerr)) && !(a[i] == 0.0f && b[i] == 0.0f)) {
			printf("ERROR: value in array a not equal to in b:\na[%d] = %f\nb[%d] = %f\n", i, a[i], i, b[i], a[i]/b[i]);
			return 0;
		}
	}
	return 1;
}

int round_val(int val, int interval) {
	if(val%interval == 0) {
		return val;
	}
	else {
		return (val/interval+1)*interval;
	}
}

void print_coo(struct coo arr) {
	if(!coo_is_sorted(arr)) {
		printf("ERROR: tried to print COO array but it isn't sorted column-wise. aborting\n");
		return;
	}
	int m = arr.m;
	int n = arr.n;
	// can't print it column-wise so we'll need to allocate memory for a dense array, convert, print, and free
	float * dense = (float *)calloc(m*n, sizeof(float));
	for(int s=0; s<arr.nnz; s++) {
		int i = arr.rows[s];
		int j = arr.cols[s];
		dense[i*n+j] = arr.vals[s];
	}
	// now print it
	print_arr(dense, m, n);
}

void print_ricsr_stats(struct ricsr mtx) {
	printf("RICSR matrix has %d ribbons\n", mtx.numrib);
	printf("Ribbon matrix-width is %d\n", mtx.ribwidth);
	for(int r=0; r<mtx.numrib; r++) {
		struct csr rib = *mtx.csrs[r];
		printf("Ribbon %d: matrix dimensions %d rows x %d columns with %d nnz\n", r, rib.m, rib.n, rib.nnz);
		printf("                     ribbon sparsity: %.6f\n", float(rib.nnz)/float(rib.m*rib.n));
	}
}