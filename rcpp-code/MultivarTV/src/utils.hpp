#ifndef UTILS_HPP
#define UTILS_HPP
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// tell armadillo to use superlu for spsolve()
#define ARMA_USE_SUPERLU
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#define EPS 0.0001
using namespace arma;
typedef std::vector<int> VEC;
typedef arma::mat MAT;

// FUNCTION DECLARATIONS

// Product function

int prod(int p, VEC vec);

double prodd(vec avec);

// Range function: input min, max as integers; return array of integers min, min+1, ..., max-1, max. 

VEC range(int min, int max);

// Tensor-to-Vector Function: input number of covariates,p, and as arrays both a multi-index (arr[p] = [i_1,...,i_p]) and dimensions of mesh (arr[p] = [m_1,...,m_p])

int tensor2vector(int p, VEC multi_ind, VEC dims); 

// Vector-to-Tensor Function: input number of covariates,p, and as arrays both a vector-index (int) and dimensions of mesh (arr[p] = [m_1,...,m_p])

VEC vector2tensor(int p, int vec_ind, VEC dims); 

// Decimal-to-Binary Function: Given a number, convert it to binary. Component of determining first-difference binaries, i.e. fd_binaries(). 

VEC dec2binary(int n, int p);

// First-difference binaries into single array

umat fd_binaries(int p);

// Getting the values (1, -1) for (row,col)-index for differences designated in sparse matrix

uvec get_col_inds( int p, VEC dims, int direction);

uvec get_row_inds(uvec col_inds);

vec get_vals(uvec col_inds);

// For any particular set of partial derivatives we want, we aim to approximate them using differences. 
// We need to create a function that returns a difference matrix for differences along a particular dimension of the array
// i.e. if direction = 0, then first differences along the rows of the matrix
// if direction = 1, then along the columns

sp_mat build_diffmat(int p, VEC dims, int direction);

//typedef struct spmats {
//	sp_mat M;
//} spmats;

// Use build_diffmat() to create a difference matrix corresponding to a binary. We separate pure partials from mixed partials. 

sp_mat mixedpartial(int p, VEC dims, uvec binary);

sp_mat binary2diffmat(int p, VEC dims, uvec binary);

// Use binary2diffmat() to build a single difference: a stack of matrices corresponding to each binary
vec ew_power(vec x, vec pows);

sp_mat create_D(int p, vec dims, vec deltas);

// Given data with column features, generate a (m_0 x ... x m_p-1)-mesh tensor flattened by column. Also, calculate deltas\equiv mesh widths. 

MAT create_mesh(mat data, vec dims);
vec create_deltas(mat data, vec dims);

// For given observation, find nearest mesh value

uword nearest1_unit(rowvec target, MAT choices);

// For matrix of observations, find nearest mesh values for each

uvec nearest1( mat data, MAT mesh);

// Create interpolation matrix O, i.e. for each x_i, O_ij has a 1.0, where the jth mesh point is nearest x_i

sp_mat nearest_interp_matrix(mat data, MAT mesh);

// Conjugate Gradient

vec cg(sp_mat A, vec b);

// Perform pseudo-inverse of 

vec mypinv(sp_mat a, vec Oty, int d);

double lam_max_pinv(arma::sp_mat a, arma::vec Oty, int d);

// Row mean function

vec rowmean(mat A);

// CROSS-VALIDATION FUNCTIONS/STRUCTS

//typedef struct kfolds {
//	std::vector<mat> Xtrain;
//	std::vector<mat> Xtest;
//	std::vector<vec> Ytrain;
//	std::vector<vec> Ytest;
//} kfolds;

Rcpp::List kfold(int k, arma::mat data, arma::vec y);

#endif