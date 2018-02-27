#include <iostream>
#include <armadillo>
#include <cmath>
#include <string>
#include <cstdlib>
#include <array>
#include "utils.hpp"

using namespace arma; 
typedef std::vector<int> VEC;

/* FUNCTION DEFINITIONS */

VEC range(int min, int max){
	VEC vec(max-min+1); // If you do not know size, then do not declare and use vec.push_back(min+i); 
	for (int i=0; i<max-min+1; i++){
		vec[i] = min + i;
	}
	return vec;
}

int tensor2vector(int p, VEC multi_ind, VEC dims){
	int vec_ind;
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

VEC vector2tensor(int p, int vec_ind, VEC dims){
	VEC multi_ind(p);
	int i, j, dims_prod;
	float ind2 = vec_ind + 1.0;
	multi_ind = range(0,p-1);

	for (i=p; i>0; i--){
		dims_prod=1;
		for (j=1; j<= i-1; j++){
			dims_prod *= dims[j];
		}
		multi_ind[i-1] = ceil(ind2/(float)dims_prod) - 1;
		ind2 -= multi_ind[i-1]*dims_prod;
	}
	return multi_ind;
}

VEC dec2binary(int n, int p){
	VEC binaryNum_long(1000);	// array to store binary number, allow big enough for large n
		// counter for binary array
	int i = 0;
	while (n > 0) { 
		binaryNum_long[i] = n % 2;  // storing remainder in binary array
		n = n / 2;
		i++;
	}
	VEC binaryNum(p);
  // printing binary array in reverse order
	for (int j = p-1; j >= p - i - 1; j--)
		binaryNum[j] = binaryNum_long[p-1-j];
	for (int j = 0; j < p- i - 2; j++)
		binaryNum[j] = 0;
	return binaryNum;
}

umat fd_binaries(int p){
	int i,j;
	int alpha = 1<<p;
	umat fd_bins(alpha-1, p);
	for (i=0; i < alpha - 1; i++){
		//printf("i= %i", i);
		VEC binnum = dec2binary(i+1,p);
		uvec abinnum = conv_to<uvec>::from(binnum);
		fd_bins.row(i) = trans(abinnum);
		//for (j=0; j<p; j++)
		//	fd_bins[i][j] = binnum[j];
		//free(binnum);
	}
	return fd_bins;
}

void get_spinds( int p, VEC dims, int direction, uvec &col_ind, uvec &row_ind, uvec &vals){
	VEC ind(p,0);
	int row = 0;
	int i,j,vec_ind;
	int dims_prod = 1;
	for ( j=0; j< p; j++)
		dims_prod *= dims[j];
	for ( i=0; i < dims_prod; i++){
		if (ind[direction] + 1 < dims[direction])
			col_ind[i] = tensor2vector(p, ind, dims);
			row_ind[i] = row;
			vals[i] = 1;
			ind[direction] += 1;
			col_ind[i] = tensor2vector(p, ind, dims);
			row_ind[i] = row;
			vals[i] = -1;
		ind = vector2tensor(p, i+1, dims);
	}
}

