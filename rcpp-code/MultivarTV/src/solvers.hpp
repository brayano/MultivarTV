#ifndef SOLVERS_HPP
#define SOLVERS_HPP
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// tell armadillo to use superlu for spsolve()
#define ARMA_USE_SUPERLU
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
#include "utils.hpp"

using namespace arma;
typedef std::vector<int> VEC;
typedef arma::mat MAT;
#define EPS 0.0001
#define TOL 0.01

// FUNCTION DECLARATION 

// Basic utility functions

vec pointmax(vec a, double b);

vec softthresh(vec z , double lam);

// Structs to initialize objects oused within functions
typedef struct mbs_cache {
	sp_mat sp_crosses; // Product of sparse difference matrices
	vec Oty; // product of interpolation matrix and response
	sp_mat D; // sparse difference matrix
	sp_mat Dt; // transpose of difference matrix
	int rowsD; // n_rows in difference matrix
	sp_mat O; // interpolation matrix
	sp_mat Ot; // transpose of interpolation matrix
	int ntheta; // number of mesh points
	sp_mat crossO;
	sp_mat crossD;
} mbs_cache;

typedef struct mbs_one_inits {
	sp_mat O; sp_mat Ot; sp_mat D; sp_mat Dt; int rowsD; vec Oty; 
	int ntheta;
	vec deltas; // vector of mesh widths
	sp_mat crossD; // D.t()*D;
	sp_mat crossO; 	// O.t()*O;
	sp_mat sp_crosses; // crossO + rho*crossD;
} mbs_one_inits;

typedef struct mbs_one_object {
	MAT mesh; // the mesh
	vec theta_hat;
	vec fitted;
	mat data;
	vec y;
	vec m;
} mbs_one_object;

typedef std::vector<mbs_one_object> MBSVEC;

typedef struct mbs_object {
	mbs_one_object minmse_model;
	MBSVEC models;
	double minmse;
	double minmse_lambda;
	vec mses;
	// add lambdas
} mbs_object;

// Functions to handle cache assignment/creation for mbs_one() and mbs()

void create_cache(mat data, MAT mesh, mbs_one_inits & inits);

void use_cache(mbs_cache* cache, mbs_one_inits & inits);

void fill_cache(mbs_cache*& cache, mbs_one_inits inits);

void fill_output_mbs_one(mbs_one_object& output, mat data, vec y, MAT mesh, vec theta, mbs_one_inits inits, vec m);

// Adaptive step size function

typedef struct adaptstep {
	double rho_next;
	vec u_next;
} adaptstep;

void adapt_step(vec r_current, vec s_current, double rho_current, vec u_current, adaptstep &object);

// ADMM Update Function
vec admm_update(vec y, mbs_one_inits inits, vec* theta_init, double lambda, bool verbose);

// Mesh-based Solution to Total Variation Problem with Approximated Penalty at One tuning parameter. Output is theta.hat

void mbs_one(mat data, vec y, vec m, mbs_one_object & output , MAT mesh ,  vec* theta_init = NULL, double lambda = 1.0 , mbs_cache* cache = NULL , bool verbose = true);

// Given data, find fitted values using estimates in mbs_one_object

vec mbs_predict(mbs_one_object model, mat data);

// Calculate mean squared error for mbs_one_object
double mse(vec fits, vec y);
double mbs_mse(mbs_one_object model, vec y);

// Functions to handle tuning parameter and output creation 

void fill_output(mbs_object &output, vec mse, vec lambdas);

// CREATE VECTOR OF LAMBDAS

arma::vec create_lambdas(int n_lambda, mbs_one_object inits, Rcpp::Nullable<arma::vec> lambdas=R_NilValue, bool verbose = true, int d = 1);

// GENERATE SOLUTION TO VECTOR OF LAMBDAS

void mbs_path(mat data, vec y, vec m, MAT mesh, int n_lambda, vec lambdas, vec ftrue, mbs_object &output, mbs_one_inits inits, mbs_cache* cache, bool verbose);

// GENERATE MESH GIVEN EMPTY/NON-EMPTY POINTER

arma::mat gen_mesh(arma::mat data, arma::vec m, Rcpp::Nullable<arma::mat> mesh = R_NilValue);

// GENERATE RESPONSE GIVEN EMPTY/NON-EMPTY POINTER

arma::vec gen_ftrue(arma::vec y, Rcpp::Nullable<arma::vec> ftrue= R_NilValue);

// USE MBS_PATH() OUTPUT TO FIND OPTIMAL MODEL

void mbs_fit_optimal(mat data, vec y, vec m, mbs_one_object &best_model, MAT mesh, vec lambdas, mat mse_mat, mbs_cache* cache, mbs_one_inits inits, bool verbose);

// Using solutions, generate predictions for test data

arma::vec test_mse(arma::mat data, arma::vec y, mbs_object path_object, int n_lambda);

// Create list of models in fitted path

Rcpp::List listPATH(mbs_object pob, arma::vec lambdas);

// Perform mbs_one over range of lambdas

Rcpp::List mbs_impl(arma::mat data, arma::vec y, arma::vec m, Rcpp::Nullable<arma::mat> mesh = R_NilValue, int n_lambda = 100, Rcpp::Nullable<arma::vec> ftrue = R_NilValue, Rcpp::Nullable<arma::vec> lambdas = R_NilValue, int folds =5, bool verbose = true); 

#endif