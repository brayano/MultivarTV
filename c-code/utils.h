#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

/* FUNCTION DECLARATIONS */

// Product function: input array, output product of all elements in array

// Range function: input min, max as integers; return array of integers min, min+1, ..., max-1, max. 

void range(int min, int max, int arr[max-min+1]);

// Tensor-to-Vector Function: input number of covariates,p, and as arrays both a multi-index (arr[p] = [i_1,...,i_p]) and dimensions of mesh (arr[p] = [m_1,...,m_p])

int tensor2vector(int p, int multi_ind[p], int dims[p], int vec_ind); 

// Vector-to-Tensor Function: input number of covariates,p, and as arrays both a vector-index (int) and dimensions of mesh (arr[p] = [m_1,...,m_p])

int vector2tensor(int p, int vec_ind, int dims[p], int multi_ind[p]); 

// Decimal-to-Binary Function: Given a number, convert it to binary. Component of determining first-difference binaries, i.e. fd_binaries(). 

int *dec2binary(int n, int p);

// First-difference binaries into single array

int **fd_binaries(int p);

// Getting the values (1, -1) for (row,col)-index for differences designated in sparse matrix

typedef struct spinds {
	int val;
	int col_ind;
	int row_ind;
} spinds;

void get_spinds( spinds* buffer, int p, int dims[p], int direction);


