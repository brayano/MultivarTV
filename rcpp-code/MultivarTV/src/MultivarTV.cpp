// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// tell armadillo to use superlu for spsolve()
#define ARMA_USE_SUPERLU
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "Rcpp.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
#include "solvers.hpp"
//#include "tvR_types.hpp"

#define EPS 0.0001
using namespace arma; 
typedef std::vector<int> VEC;
typedef arma::mat MAT;

// MVTV Cross-Validation Procedure
//' @title Default Multivariate Total Variation Denoising Solver for use by S3 Generic
//' @description Create a mesh and find cross-validated best 
//' approximation to total variation denoising problem. 
//' @name mvtv_default
//' @param data n by p matrix of inputs
//' @param y response column vector
//' @param m vector of number of mesh points per predictor 
//' @param mesh user can supply or NULL for regularly spaced mesh, which will be returned
//' @param n_lambda number of logarithmically spaced tuning parameters
//' @param ftrue prediction target. If NULL, use observed data. 
//' @param lambdas user can supply vector of lambdas to be solved over. If NULL,
//' function generates n_lambda logarithmically spaced lambdas from 0.00001*lambda_max
//' and lambda_max, where lambda_max is our approximation of smallest lambda where
//' regularization ends. 
//' @param folds number of folds for cross-validation
//' @param verbose Default: true, prints out current working penalty and 
//' number of iters to solve.
//' @export
// [[Rcpp::export(name="mvtv_default")]]
Rcpp::List mbs(arma::mat data, arma::vec y, arma::vec m, Rcpp::Nullable<arma::mat> mesh = R_NilValue, int n_lambda = 100, Rcpp::Nullable<arma::vec> ftrue = R_NilValue, Rcpp::Nullable<arma::vec> lambdas = R_NilValue, int folds =5, bool verbose = true){
  Rcpp::List mbsout = mbs_impl(data, y, m, mesh, n_lambda, ftrue, lambdas, folds, verbose);
  return mbsout;
}

// MVTV Prediction Function
//' @title MVTV Predict for use by S3 Generic Function
//' @description Use fitted 'mvtv' object to predict new data.
//' @name predict_mvtv
//' @param mvtvobject object produced by mbtv.default
//' @param data n by p matrix of inputs
//' @param mesh m by p mesh used by fitting function mvtv
//' @export
// [[Rcpp::export(name="predict_mvtv")]]
arma::vec mbspredict(Rcpp::List mvtvobject, Rcpp::Nullable<arma::mat> data = R_NilValue, Rcpp::Nullable<arma::mat> mesh = R_NilValue){
  //if (! mvtvobject.inherits("mvtv")) Rcpp::stop("Input must be a mvtv.default() model object.");
  if (data.isNull()) return mvtvobject["fitted"];
  
  arma::mat Data = Rcpp::as<arma::mat>(data);
  arma::sp_mat O;
  if (mesh.isNull()) O = nearest_interp_matrix(Data,mvtvobject["mesh"]);
  else{
    arma::mat Mesh = Rcpp::as<arma::mat>(mesh);
    O = nearest_interp_matrix(Data,Mesh);
  } 
  //Mesh = Rcpp::as<arma::mat>(mesh);
  //arma::sp_mat O = nearest_interp_matrix(Data, Mesh);
  arma::vec thetahat = Rcpp::as<arma::vec>(mvtvobject["theta_hat"]);
  arma::vec fits = O*thetahat;
  return fits;
}  
