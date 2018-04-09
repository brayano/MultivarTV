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
#define T 60

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
	VEC dims2(2,3);
	VEC specdims = {2,3};

	//range(1,N,myints);
	vec_ind = tensor2vector(P, myind,dims);
	vec_ind2 = tensor2vector(P, myind2,dims);
	tensor_ind = vector2tensor(P,0, dims);
	tensor_ind2 = vector2tensor(P,26,dims);
	int dims_prod = prod(P, myind2);
	std::cout << "Printing dims_prod =" << dims_prod; 
	std::cout << "\n";

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

	printf("Using vector2tensor() for (3,2,3)-mesh for 0-26 = \n");
	VEC altdims = {3,2,3};
	for (i=0; i<3; i++){
		VEC tenind = vector2tensor(P,i,altdims);
		//printf("At %i : \n", i);
		for (j=0; j<P; j++){
			printf("%i ", tenind[j]);
		}
		printf("\n");
	}

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

	//uvec col_ind = get_col_inds(P,dims,0);
	uvec col_ind_spec = get_col_inds(2,specdims,1);

	printf("ONE \n");
	//uvec row_ind = get_row_inds(col_ind);
	printf("TWO \n");
	//vec vals = get_vals(col_ind);
	//get_spinds(P, dims, 1, col_ind, row_ind, vals);
	//std::cout << "Printing row_ind: " << row_ind.n_rows << "\n";
	//for (i=0; i < row_ind.size(); ++i){
	//	printf("%llu ", row_ind[i]);
	//}
	//std::cout << '\n';

	std::cout << "Printing col_ind_spec: " << col_ind_spec.n_rows << "\n";
	for (i=0; i < col_ind_spec.size(); ++i){
		printf("%llu ", col_ind_spec[i]);
	}
	std::cout << '\n';

	//std::cout << "Printing vals: " << vals.n_rows << "\n";
	//for (i=0; i < vals.size(); ++i){
	//	printf("%f ", vals[i]);
	//}
	//std::cout << '\n';

	//sp_mat D = build_diffmat(P,dims,1);
	//sp_mat D = build_diffmat(2,dims2,0);
	//D.print("D:");

	//VEC mixedbinary = {0,1,1};
	//uvec binary = conv_to<uvec>::from(mixedbinary);
	//sp_mat Dmix = mixedpartial(P, dims, binary);
	//sp_mat Dmix2 = binary2diffmat(P, dims, binary);
	
	//sp_mat Dmix = mixedpartial(2, dims2, binary);
	//sp_mat Dmix2 = binary2diffmat(2, dims2, binary);

	//VEC abinary = {1,0,0};
	//uvec binary0 = conv_to<uvec>::from(abinary);
	//sp_mat D2 = binary2diffmat(P, dims, binary0);
	
	vec deltas = {0.1,0.1,0.1};
	sp_mat Dmat = create_D(P,dims,deltas);
	//Dmat.print("Dmat: ");
	//sp_mat Dmat = create_D(2,dims2,deltas);
	//Dmat.print("Dmat:");
	mat X = randu<mat>(100,2);
	vec y = randu<vec>(100,1);
	//y.print("Y= ");

	VEC meshdims = {3,3};
	mat mesh = create_mesh(X, meshdims);
	//mesh.print("mesh: ");
	vec delts = create_deltas(X,meshdims);
	//delts.print("deltas = ");

	vec target = {0.028,0.028};
	printf("past \n");
	uword nint = nearest1_unit(target,mesh);
	printf("nint = %llu \n", nint);

	uvec nints = nearest1(X,mesh);
	//nints.print("nints = ");

	sp_mat O = nearest_interp_matrix(X,mesh);
	//O.print("O = ");
	sp_mat D = create_D(2,meshdims,delts);
	vec Oty = O.t()*y;
	//vec b = mypinv(D, Oty);
	//b.print("b=");
	double lammax = lam_max_pinv(D, Oty);
	printf("lammax = %f \n", lammax);

	arma::arma_version ver;
	std::cout << "ARMA version: "<< ver.as_string() << std::endl;
	return 0;
}
