#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "utils.h"

#define EPS 0.01

/* FUNCTION DEFINITIONS */

void range(int min, int max, int arr[max-min+1]){
	int i;
	for (i=0; i<=max-min+1; i++){
		arr[i] = min + i;
	}
}

int tensor2vector(int p, int multi_ind[p], int dims[p], int vec_ind){
	int i,j, dims_prod;
	vec_ind = multi_ind[0];
	for (i=1; i<p; i++){
		dims_prod=1;
		for (j=1; j<= i; j++){
			dims_prod *= dims[j];
		}
		vec_ind += multi_ind[i]*dims_prod;
	}
	return vec_ind;
}

int vector2tensor(int p, int vec_ind, int dims[p], int multi_ind[p]){
	int i, j, dims_prod;
	float ind2 = vec_ind + 1.0;
	range(0,p-1,multi_ind);

	for (i=p; i>0; i--){
		dims_prod=1;
		for (j=1; j<= i-1; j++){
			dims_prod *= dims[j];
		}
		multi_ind[i-1] = ceil(ind2/(float)dims_prod) - 1;
		ind2 -= multi_ind[i-1]*dims_prod;
	}
}

int *dec2binary(int n, int p){
	int binaryNum_long[1000];	// array to store binary number, allow big enough for large n
		// counter for binary array
	int i = 0;
	while (n > 0) { 
		binaryNum_long[i] = n % 2;  // storing remainder in binary array
		n = n / 2;
		i++;
	}
	int* binaryNum = malloc(p*sizeof(int));
  // printing binary array in reverse order
	for (int j = p-1; j >= p - i - 1; j--)
		binaryNum[j] = binaryNum_long[p-1-j];
	for (int j = 0; j < p- i - 2; j++)
		binaryNum[j] = 0;
	return binaryNum;
}

int **fd_binaries(int p){
	int i,j;
	int alpha = 1<<p;
	printf("%i \n",alpha);
	//int (*fd_bins)[alpha-1][p];
	int** fd_bins = malloc(sizeof(int*)*(alpha-1));
	//fd_bins = (int**)malloc(sizeof(int[alpha-1][p]));
 	//fd_bins = (int**)malloc(sizeof(int)*(alpha-1));
	for (i=0; i < alpha - 1; i++){
		//printf("i= %i", i);
		fd_bins[i] = malloc(p*sizeof(int*));
		int *binnum = dec2binary(i+1,p);
		for (j=0; j<p; j++)
			//(*fd_bins)[i][j] = binnum[j];
			fd_bins[i][j] = binnum[j];
		free(binnum);
	}
	return fd_bins;
}

void get_spinds( spinds* buffer, int p, int dims[p], int direction){
	int ind[p];
	memset(ind, 0, sizeof ind);
	int row = 0;
	int i,j,vec_ind;
	int dims_prod = 1;
	for ( j=0; j< p; j++)
		dims_prod *= dims[j];
	for ( i=0; i < dims_prod; i++){
		if (ind[direction] + 1 < dims[direction])
			buffer[i].col_ind = tensor2vector(p, ind, dims, vec_ind);
			buffer[i].row_ind = row;
			buffer[i].val = 1;
			ind[direction] += 1;
			buffer[i].col_ind = tensor2vector(p, ind, dims, vec_ind);
			buffer[i].row_ind = row;
			buffer[i].val = -1;
		vector2tensor(p, i+1, dims, ind);
	}
}

