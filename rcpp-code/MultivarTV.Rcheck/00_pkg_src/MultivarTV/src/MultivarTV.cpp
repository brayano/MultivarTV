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
//' @title Default Multivariate Total Variation Denoising Solver
//' @description Create a mesh and find cross-validated best 
//' approximation to total variation denoising problem. 
//' @name mvtv.default
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
//' @examples
//' # Approximating Bivariate Fused Lasso for Uniform Data 
//' ## Generate Data
//' set.seed(117)
//' x <- matrix(runif(100),ncol = 2)
//' y <- matrix(runif(50),ncol=1)
//' m <- matrix(c(3,3))
//' 
//' ## Find Total Variation Solution over range of lambdas and whole data set
//' mvtv_fold1 <- mvtv(x,y,m,folds=1, verbose = FALSE)
//' 
//' ## Find 5-fold validated MVTV Model over range of lambdas
//' mvtv_fold5 <- mvtv(x,y,m,folds=5, verbose = FALSE)
//' 
//' @export
// [[Rcpp::export(name="mvtv.default")]]
Rcpp::List mbs(arma::mat data, arma::vec y, arma::vec m, Rcpp::Nullable<arma::mat> mesh = R_NilValue, int n_lambda = 100, Rcpp::Nullable<arma::vec> ftrue = R_NilValue, Rcpp::Nullable<arma::vec> lambdas = R_NilValue, int folds =5, bool verbose = true){
  Rcpp::List mbsout = mbs_impl(data, y, m, mesh, n_lambda, ftrue, lambdas, folds, verbose);
  
  mbsout.attr("class") = "mvtv";
  return mbsout;
}

// MVTV Prediction Function
//' @title MVTV Predict Function
//' @description Use fitted 'mvtv' object to predict new data.
//' @name predict.mvtv
//' @param mvtvobject object produced by mbtv.default
//' @param data n by p matrix of inputs
//' @examples
//' # Approximating Bivariate Fused Lasso for Uniform Data 
//' ## Generate Data
//' set.seed(117)
//' x <- matrix(runif(100),ncol = 2)
//' y <- matrix(runif(50),ncol=1)
//' m <- matrix(c(3,3))
//' 
//' ## Find 5-fold validated MBS Model over range of lambdas
//' mbs_fold5 <- mvtv(x,y,m,folds=5,verbose=FALSE)
//' 
//' # Access fitted values of training data; equivalent to mbs_fold5$fitted
//' fitted.values <- predict(mbs_fold5) 
//' newdata <- matrix( runif(50), ncol = 2) # Generate new data
//' newfits <- predict(mbs_fold5, newdata) # Fit new data
//' @export
// [[Rcpp::export(name="predict.mvtv")]]
arma::vec mbspredict(Rcpp::List mvtvobject, Rcpp::Nullable<arma::mat> data = R_NilValue){
  if (! mvtvobject.inherits("mvtv")) Rcpp::stop("Input must be a mvtv.default() model object.");
  if (data.isNull()) return mvtvobject["fitted"];
  
  arma::mat Data = Rcpp::as<arma::mat>(data);
  arma::sp_mat O = nearest_interp_matrix(Data, mvtvobject["mesh"]);
  arma::vec thetahat = Rcpp::as<arma::vec>(mvtvobject["theta_hat"]);
  arma::vec fits = O*thetahat;
  return fits;
}  
