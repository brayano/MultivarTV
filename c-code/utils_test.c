#include<stdio.h>
#include "utils.h"
#define N 4
#define M 3
#define P 3

/* EXECUTABLE CODE */

int main()
{
	double mat1[N][N] = { {1, 1, 1, 1},
                    {2, 2, 2, 2},
                    {3, 3, 3, 3},
                    {4, 4, 4, 4}};
 
	//double mat2[N][N] = { {1, 1, 1, 1},
  //                 {2, 2, 2, 2},
  //                  {3, 3, 3, 3},
  //                  {4, 4, 4, 4}};
	double mat2[N][1] = { {1},
                    {2},
                    {3},
                    {4}};
	
	//double asum[N][1]; // store sum
	//double result[N][1]; // store multiplication
	//double mat2t[1][N]; // store transpose
	//double desmat[N][2]; // store design matrix
	//double mat3[N][1];

	int i,j;
 	int myints[N];
	int vec_ind, vec_ind2;
	int tensor_ind[P], tensor_ind2[P];
	int myind[P] = {0,0,0};
	int myind2[P] = {2,2,2};
	int dims[P] = {3,3,3};

	range(1,N,myints);
	vec_ind = tensor2vector(P, myind,dims,vec_ind);
	vec_ind2 = tensor2vector(P, myind2,dims,vec_ind2);
	vector2tensor(P,0, dims,tensor_ind);
	vector2tensor(P,26,dims,tensor_ind2);

	printf("Printing integers 1:4 using range(): \n");
	for (i=0; i<N; i++){
		printf("%i ", myints[i]);
	}
	printf("\n");
	
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
		int *binnum = dec2binary(i,P);
		for (int j=0; j<P; j++)
			printf("%i", binnum[j]);
		printf(" ");
	}
	fflush(stdout);

	int **fd_bins = fd_binaries(P);
	printf("fd_binaries array output: \n");
	int alpha = 1<<P;
	for ( i=0; i< alpha-1; i++){
		for (j=0; j<P; j++)
			printf("%d", fd_bins[i][j]);
		printf("\n");
	}
	//free the memory
	for(i = 0; i < 1<<P - 1; i++)
		free(fd_bins[i]);
  free(fd_bins);

	spinds* spindst;
	get_spinds(spindst, P, dims, 1);

	return 0;
}
