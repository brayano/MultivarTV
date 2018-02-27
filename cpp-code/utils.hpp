#include <iostream>
#include <armadillo>
#include <cmath>
#include <string>
#include <cstdlib>
#include <array>
#include <vector>

using namespace arma;
typedef std::vector<int> VEC;

/* FUNCTION DECLARATIONS */

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

void get_spinds( int p, VEC dims, int direction, uvec &col_ind, uvec &row_ind, uvec &vals);

