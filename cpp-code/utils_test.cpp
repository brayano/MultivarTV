#include <iostream>
#include <armadillo>
#include <cmath>
#include <string>
#include <cstdlib>
#include "utils.hpp"
#include <vector>
using namespace arma;
 
#define N 4
#define M 3
#define P 3
#define T 100

/* EXECUTABLE CODE */

int main()
{
	//double mat1[N][N] = { {1, 1, 1, 1},
  //                  {2, 2, 2, 2},
  //                  {3, 3, 3, 3},
  //                  {4, 4, 4, 4}};
 
	//double mat2[N][N] = { {1, 1, 1, 1},
  //                 {2, 2, 2, 2},
  //                  {3, 3, 3, 3},
  //                  {4, 4, 4, 4}};
	//double mat2[N][1] = { {1},
  //                  {2},
  //                  {3},
  //                  {4}};
	
	//double asum[N][1]; // store sum
	//double result[N][1]; // store multiplication
	//double mat2t[1][N]; // store transpose
	//double desmat[N][2]; // store design matrix
	//double mat3[N][1];
	int i ,j;
 	VEC myints = range(1,N);
	int vec_ind, vec_ind2;
	VEC tensor_ind, tensor_ind2;
	VEC myind(P,0);
	VEC myind2(P,2);
	VEC dims(P,3);
	//range(1,N,myints);
	vec_ind = tensor2vector(P, myind,dims);
	vec_ind2 = tensor2vector(P, myind2,dims);
	tensor_ind = vector2tensor(P,0, dims);
	tensor_ind2 = vector2tensor(P,26,dims);

	std::cout << "Printing integers 1:4 using range(): \n";
	for (VEC::iterator it=myints.begin(); it != myints.end(); ++it){
		std::cout << ' ' << *it;
	}
	std::cout << '\n';
	
	// printf() still works in C++!!
	printf("Using tensor2vector() for (3,3,3)-mesh at (0,0,0), vec_ind = %d (0 is correct) \n", vec_ind);
	printf("Using tensor2vector() for (3,3,3)-mesh at (2,2,2), vec_ind = %d (27-1 is correct) \n", vec_ind2);

	printf("Using vector2tensor() for (3,3,3)-mesh at 0, ({0,0,0} is correct) multi_ind =  \n");
	for (i=0; i<P; i++){
		printf("%i ", tensor_ind[i]);
	}
	printf("\n");
	
	printf("Using vector2tensor() for (3,3,3)-mesh at 26, ({2,2,2} is correct) multi_ind =  \n");
	for (i=0; i<P; i++){
		printf("%i ", tensor_ind2[i]);
	}
	printf("\n");

	printf("Printing the 7 binaries for p=3: \n");
	for (i=1; i<8; i++){
		VEC binnum = dec2binary(i,P);
		for (int j=0; j<P; j++)
			printf("%i", binnum[j]);
		printf(" ");
	}
	printf("\n");
	fflush(stdout);

	umat fd_bins = fd_binaries(P);
	printf("fd_binaries array output: \n");
	int alpha = 1<<P;
	for ( i=0; i< alpha-1; i++){
		for (j=0; j<P; j++)
			printf("%llu", fd_bins(i,j) );
		printf("\n");
	}

	uvec col_ind(T), row_ind(T), vals(T);
	get_spinds(P, dims, 1, col_ind, row_ind, vals);

	return 0;
}
