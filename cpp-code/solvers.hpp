#include <iostream>
#include <armadillo>
#include <cmath>
#include <string>
#include <cstdlib>
#include <array>
#include <vector>
#include <stdexcept>
#include "utils.hpp"
using namespace arma;
typedef std::vector<int> VEC;
typedef fmat MAT;
#define EPS 0.01
#define TOL 0.001

/* FUNCTION DECLARATION */

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

void create_cache(mat data, MAT mesh, mbs_one_inits &inits);

void use_cache(mbs_cache* cache, mbs_one_inits &inits);

void fill_cache(mbs_cache*& cache, mbs_one_inits inits);

void fill_output_mbs_one(mbs_one_object& output, mat data, vec y, MAT mesh, vec theta, mbs_one_inits inits, vec m);

// Adaptive step size function

typedef struct adaptstep {
	double rho_next;
	vec u_next;
} adaptstep;

void adapt_step(vec r_current, vec s_current, double rho_current, vec u_current, adaptstep &object);

// ADMM Update Function
vec admm_update(vec y, mbs_one_inits inits, vec* theta_init, double lambda);

// Mesh-based Solution to Total Variation Problem with Approximated Penalty at One tuning parameter. Output is theta.hat

void mbs_one(mat data, vec y, vec m, mbs_one_object & output , MAT mesh ,  vec* theta_init = NULL, double lambda = 1.0 , mbs_cache* cache = NULL );

// Given data, find fitted values using estimates in mbs_one_object

vec mbs_predict(mbs_one_object model, mat data);

// Calculate mean squared error for mbs_one_object
double mse(vec fits, vec y);
double mbs_mse(mbs_one_object model, vec y);

// Functions to handle tuning parameter and output creation 

void fill_output(mbs_object &output, vec mse, vec lambdas);

// CREATE VECTOR OF LAMBDAS

vec create_lambdas(int n_lambda, mbs_one_object inits, vec* lambdas);

// GENERATE SOLUTION TO VECTOR OF LAMBDAS

void mbs_path(mat data, vec y, vec m, MAT mesh, int n_lambda, vec lambdas, vec ftrue, mbs_object &output, mbs_one_inits inits, mbs_cache* cache);

// GENERATE MESH GIVEN EMPTY/NON-EMPTY POINTER

MAT gen_mesh(mat data, vec m, MAT* mesh);

// GENERATE RESPONSE GIVEN EMPTY/NON-EMPTY POINTER

vec gen_ftrue(vec y, vec* ftrue);

// USE MBS_PATH() OUTPUT TO FIND OPTIMAL MODEL

void mbs_fit_optimal(mat data, vec y, vec m, mbs_one_object &best_model, MAT mesh, vec lambdas, mat mse_mat, mbs_cache* cache, mbs_object path_object, mbs_one_inits inits);

// Using solutions, generate predictions for test data

vec test_mse(mat data, vec y, mbs_object path_object, int n_lambda);

// Perform mbs_one over range of lambdas

void mbs(mat data, vec y, vec m, mbs_one_object &output,  MAT* mesh=NULL, int n_lambda = 100, vec* ftrue=NULL, vec* lambdas = NULL, int folds =5); 

