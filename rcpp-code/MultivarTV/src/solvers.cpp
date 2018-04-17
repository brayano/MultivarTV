// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// tell armadillo to use superlu for spsolve()
#define ARMA_USE_SUPERLU
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

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

vec pointmax(vec a, double b){
	int i;
	vec amax(a.size());
	for (i=0; i<a.size(); i++){
		amax[i] = std::max(a[i],b);
	}
	return amax;
}

vec softthresh(vec z , double lam){
	vec zsign = sign(z);
	vec pmax = pointmax(abs(z)-lam,0.0);
	vec zp = zsign % pmax;
	return zp;
}

void create_cache_objects(mat data, vec y, MAT mesh, vec meshdims, mbs_one_inits &inits){
	inits.O = nearest_interp_matrix(data,mesh); 
	inits.Ot = inits.O.t();
	inits.D = create_D(data.n_cols,meshdims,inits.deltas); 
	inits.Dt = inits.D.t();
	inits.crossD = inits.Dt*inits.D; inits.crossO = inits.Ot*inits.O;
	inits.rowsD = inits.D.n_rows;
	inits.Oty = inits.Ot*y;
}

void use_cache(mbs_cache* cache, mbs_one_inits &inits){
	inits.sp_crosses = cache -> sp_crosses;
	inits.Oty = cache -> Oty;
	inits.D = cache -> D;
	inits.Dt = cache -> Dt;
	inits.rowsD = cache -> rowsD;
	inits.O = cache -> O;
	inits.Ot = cache -> Ot;
	inits.ntheta = cache -> ntheta;
	inits.crossD = cache -> crossD;
	inits.crossO = cache -> crossO;
}

void fill_cache(mbs_cache*& cache, mbs_one_inits inits){
	cache -> O = inits.O;
	cache -> Ot = inits.Ot;
	cache ->D = inits.D;
	cache ->Dt = inits.Dt;
	cache ->rowsD = inits.rowsD;
	cache ->Oty = inits.Oty;
	cache -> ntheta = inits.ntheta;
	cache -> crossO = inits.crossO;
	cache -> crossD = inits.crossD;
}

void fill_output_mbs_one(mbs_one_object& output, mat data, vec y, MAT mesh, admm_out out, mbs_one_inits inits, vec m){
	output.mesh = mesh; output.theta_hat = out.theta; output.uhat = out.u; output.rhohat = out.rho;
	output.fitted = inits.O*out.theta; output.data = data; 
	output.y = y; output.m = m; 
}

void adapt_step(vec r_current, vec s_current, double rho_current, vec u_current, adaptstep &object){
	double rho_next;
	double r_norm = norm(r_current);
	double s_norm = norm(s_current);
	double tau = 2.0;
	if (r_norm > 10*s_norm){
		object.rho_next = tau*rho_current;
		object.u_next = 1.0/tau*u_current;
	}
	else if ( s_norm > 10*r_norm){
		object.rho_next = 1.0/tau*rho_current;
		object.u_next = tau*u_current;
	}
	else {
		object.rho_next = rho_current;
		object.u_next = u_current;
	}
}

void admm_update(vec y, mbs_one_inits inits, vec & theta_init, double lambda, bool verbose, vec & u_init, double & rho_init, admm_out & out){
	// Set-up ADMM: Initialize theta, u and rho. 
	vec theta(inits.ntheta); theta = theta_init;
  vec u(inits.rowsD); u = u_init;
  double rho; rho = rho_init;
  
	vec alpha = inits.D*theta;
	int counter = 1;
	int max_counter = 3000;
	vec b, uold;
	vec primal_residual(inits.rowsD); vec dual_residual(inits.ntheta);
	adaptstep stepobject;
	arma::sp_mat spcrosses = inits.sp_crosses;
	double dual_norm = 1, primal_norm = 1;
	double eps_dual = TOL, eps_primal = TOL;
	while ( dual_norm > eps_dual or primal_norm > eps_primal ) {
	  uold = u;
		b = inits.Oty + rho*inits.Dt*(alpha+u);
		theta = spsolve( spcrosses , b , "superlu" );
		alpha = softthresh(inits.D*theta-u, lambda/rho);
		primal_residual = alpha - inits.D*theta;
		u +=  primal_residual;
		dual_residual = rho*inits.Dt*(u - uold);
		// Stopping criterions
		dual_norm = norm(dual_residual);
		primal_norm = norm(primal_residual);
		eps_dual = TOL*(sqrt(inits.ntheta) + norm(inits.Dt*u));
		eps_primal = TOL*(sqrt(inits.rowsD) + std::max(norm(inits.D*theta),norm(alpha)));

		counter += 1;
		if (counter>max_counter){
			throw std::invalid_argument("Failed to converge!");
		}
		adapt_step(primal_residual, dual_residual, rho,u, stepobject);
		rho = stepobject.rho_next; u = stepobject.u_next;
		spcrosses = inits.crossO + rho * inits.crossD; 
	}
	if (verbose) Rcpp::Rcout << "Lambda= " << lambda << ", Counter = " << counter << std::endl;
	out.theta = theta; out.rho = rho; out.u = u;
}

// MBS Solution for single lambda

void mbs_one(mat data, vec y, vec m, mbs_one_object & output, MAT mesh, vec & u, double & rho, vec & theta_init, double lambda, mbs_cache* cache, bool verbose){
	// MESH MUST BE SUPPLIED TO SINGLE LAMBDA SOLVER!
	// CALL MBS(..., LAMBDAS = LAMBDA) FOR SINGLE SOLVE IN PRACTICE.

	// Initialize objects within a struct
	mbs_one_inits inits;

	if (cache==NULL){
		inits.ntheta = prodd(m);
		create_cache_objects(data,y,mesh, m, inits);
		inits.sp_crosses = inits.crossO + lambda*inits.crossD;
	}
	else{
		use_cache(cache, inits);
	}
	// Get ADMM estimate of theta
	admm_out out;
	admm_update(y, inits, theta_init, lambda, verbose, u, rho, out);
	fill_output_mbs_one(output, data, y, mesh, out, inits, m);
}

vec mbs_predict(mbs_one_object model, mat data){
	sp_mat O = nearest_interp_matrix(data, model.mesh);
	vec fits = O*model.theta_hat;
	return fits;
}

double mse(vec fits, vec y){
  double mseval = pow(norm(fits - y),2)/ y.n_rows;
	return mseval;
}

double mbs_mse(mbs_one_object model, vec y){
	double mseval = mse(model.fitted, y);
	return mseval;
}

void fill_output_mbs(mbs_object &output, vec mse, vec lambdas){
// In the event of a tie, we use the first instance of lowestMSE or  put another way, the largest value of lambda with that lowestMSE
	uword lowestMSE = mse.index_min();
	output.minmse_model = output.models[lowestMSE];
	output.minmse = mse[lowestMSE]; 
	output.minmse_lambda = lambdas[lowestMSE];
	output.mses = mse;
}

arma::vec create_lambdas(int n_lambda, mbs_one_inits inits, Rcpp::Nullable<arma::vec> lambdas, bool verbose){
	double lambda_max;
	arma::vec lambdavec;
	if (lambdas.isNull()){
		lambda_max = lam_max_pinv(inits.D, inits.Oty);
		lambdavec = flipud(exp(linspace<vec>(log(lambda_max*0.0001), log(lambda_max),n_lambda)));
		
		if (verbose) Rcpp::Rcout << "Lambda_max = " << lambda_max << std::endl;
	}
	else{
		lambdavec = Rcpp::as<arma::vec>(lambdas);
	}
	return lambdavec;
}

// MBS solutions for a path of lambdas

void mbs_path(mat data, vec y, vec m, MAT mesh, int n_lambda, vec lambdas, vec ftrue, mbs_object &output, mbs_one_inits inits, mbs_cache* cache, bool verbose){
	mbs_one_object tempmodel; 
	vec MSEs(n_lambda);
	vec theta_init(inits.ntheta); theta_init.fill(mean(y));
	vec u_init(inits.rowsD); u_init.fill(0.0); 
	double rho_init = lambdas[0]/5.0;
	// Gradient-step size
	int i;
	for (i=0; i<n_lambda; i++){
		sp_mat sp_crosses = inits.crossO+rho_init*inits.crossD; cache -> sp_crosses = sp_crosses; // Create/store sum of cross products
		mbs_one(data,y,m, tempmodel, mesh, u_init, rho_init, theta_init, lambdas[i], cache, verbose);
		output.models.push_back(tempmodel);
		MSEs[i] = mbs_mse(tempmodel, ftrue);
		theta_init = tempmodel.theta_hat; // theta_init points to thetainit
		rho_init = tempmodel.rhohat;
		u_init = tempmodel.uhat;
	}
	fill_output_mbs(output, MSEs, lambdas);
}

// Single function to handle mesh creation
//' @title Generate a mesh
//' @description Single function to handle creating a mesh regularly
//' across domain of predictors. Mesh created is a convex hull of predictor space. 
//' @name gen_mesh
//' @param data n by p matrix of inputs
//' @param m vector of length p with number of knots desired for each predictor
//' @param mesh NULL; otherwise, takes user defined mesh. 
//' @export
// [[Rcpp::export]]
arma::mat gen_mesh(arma::mat data, arma::vec m, Rcpp::Nullable<arma::mat> mesh){
	// IF MISSING, CREATE MESH, I.E. FIND EVENLY SPACED POINTS ACROSS EACH PREDICTOR:
	arma::mat MESH;
	if (mesh.isNull()){
		MESH = create_mesh(data,m);
	}
	else{
		MESH = Rcpp::as<arma::mat>(mesh);
	}
	return MESH;
}

// DATA WE WILL CALCULATE MSE FOR: 

arma::vec gen_ftrue(vec y, Rcpp::Nullable<arma::vec> ftrue){
	arma::vec Y;
	if (ftrue.isNull()){
		Y = y;
	}
	else{
		Y = Rcpp::as<arma::vec>(ftrue);
	}
	return Y;
}

// Given matrix of MSEs with columns as lambdas and rows as realizations, fit MBS with best lambda

void mbs_fit_optimal(mat data, vec y, vec m, mbs_one_object &best_model, MAT mesh, vec lambdas, mat mse_mat, mbs_cache* cache, mbs_one_inits inits, bool verbose){
	vec mean_mses = rowmean(mse_mat);
	// FIT OPTIMAL LAMBDA OVER TRAINING SET
	uword lowestMSE = mean_mses.index_min();
	// MBS_ONE() requires theta_init as pointer
	vec theta_init(inits.ntheta); theta_init.fill(mean(y));
	vec u_init(inits.rowsD); u_init.fill(0.0); 
	double rho_init = lambdas[0]/5.0;
	double best_lambda = lambdas[lowestMSE];
	
	if (verbose) Rcpp::Rcout << "Best lambda = " << best_lambda << std::endl;
	
	mbs_one(data,y,m,best_model, mesh, u_init, rho_init, theta_init, best_lambda, cache, verbose);
}

// GIVEN SOLUTIONS TO LAMBDAS, PREDICT TESTING DATA AND MEASURE ACCURACY

vec test_mse(mat data, vec y, mbs_object path_object, int n_lambda){
	int i;
	vec mses(n_lambda);
	vec fits(data.n_rows);
	for (i=0; i<n_lambda; i++){
		fits = mbs_predict(path_object.models[i], data);
		mses[i] = mse(fits, y);
    //Rcpp::Rcout << "test_mse = " << mses[i] << std::endl;
	}
	return mses;
}

// Conversion function

Rcpp::List listPATH(mbs_object pob, arma::vec lambdas){
  // pob \equiv path_object
  Rcpp::List out;
  int i;
  for (i=0; i< lambdas.n_rows; i++){
    out.push_back(Rcpp::List::create(Rcpp::Named("lambda", lambdas[i]),Rcpp::Named("mse", pob.mses[i]),
                                     Rcpp::Named("theta_hat",pob.models[i].theta_hat),
                                     Rcpp::Named("fitted",pob.models[i].fitted)));
  }
  return out;
}

// MBS Cross-Validation Procedure
Rcpp::List mbs_impl(const arma::mat data, const arma::vec y, arma::vec m, Rcpp::Nullable<arma::mat> mesh, int n_lambda, Rcpp::Nullable<arma::vec> ftrue, Rcpp::Nullable<arma::vec> lambdas, int folds, bool verbose){
	// Initialize objects
	mbs_one_object best_model;
	mbs_one_inits inits;
	inits.ntheta = prodd(m);
	inits.deltas = create_deltas(data,m);
	MAT MESH = gen_mesh(data,m,mesh);
	if (verbose) Rcpp::Rcout << "MBS Begins: ntheta =  " << inits.ntheta << std::endl;

	// Create cache struct
	mbs_cache* cache = new mbs_cache();
	cache -> ntheta = inits.ntheta;
	create_cache_objects(data,y,MESH, m, inits); // Create inits
	fill_cache(cache,inits); // Create cache using inits
	
	// Create tuning parameters
	vec LAMBDAS = create_lambdas(n_lambda, inits, lambdas, verbose);
	vec FTRUE = gen_ftrue(y,ftrue);
	int i;
	mbs_object final_path;
	mat mse_mat(n_lambda, folds);
	vec mean_mses(n_lambda);
	uword bestModelind;
	if (folds == 1){
	  mbs_path(data, y, m, MESH, n_lambda, LAMBDAS, FTRUE, final_path, inits, cache,verbose);
	  mse_mat.col(0) = test_mse(data, y, final_path, n_lambda);
	  mbs_fit_optimal(data,y,m,best_model,MESH,LAMBDAS, mse_mat, cache, inits,verbose);
	  uword lowestMSE = mse_mat.index_min();
	  bestModelind = lowestMSE;
	  mean_mses = mse_mat;
	}
	else{
	  // CROSS-VALIDATION ON TRAINING DATA: FIND LAMBDA THAT MINIMIZES MSE
	  arma::vec foldinds = kfoldinds(data.n_rows,folds);

	  for (i=0; i<folds; i++){
	    mbs_object path_object;
		  arma::uvec train_ids = find(foldinds != i); // Find training indices
	    arma::uvec test_ids = find(foldinds == i); // Find testing indices
      arma::mat train_x = data.rows(train_ids); arma::vec train_y = y.rows(train_ids);
      arma::mat test_x = data.rows(test_ids); arma::vec test_y = y.rows(test_ids);

      create_cache_objects(train_x,train_y,MESH, m, inits); // Create inits
      fill_cache(cache,inits); // Create cache using inits
      
		  mbs_path(train_x, train_y, m, MESH, n_lambda, LAMBDAS, train_y, path_object, inits, cache, verbose);
	    if (verbose) Rcpp::Rcout << "Fold Complete: " << i << std::endl;
		  mse_mat.col(i) = test_mse(test_x, test_y, path_object, n_lambda);
	  }
	  // FIT OPTIMAL SOLUTION
	  create_cache_objects(data,y,MESH, m, inits); // Create inits
	  fill_cache(cache,inits); // Create cache using inits
	  
	  mbs_path(data, y, m, MESH, n_lambda, LAMBDAS, y, final_path, inits, cache, verbose);
	  mean_mses = rowmean(mse_mat);
	  uword lowestMSE = mean_mses.index_min();
	  bestModelind = lowestMSE;
	  best_model = final_path.models[bestModelind];
	}
	// Create residuals
	arma::vec residuals = y - best_model.fitted;
	Rcpp::List models = listPATH(final_path,LAMBDAS);
	
	return Rcpp::List::create(Rcpp::Named("data", best_model.data),Rcpp::Named("fitted", best_model.fitted),
                           Rcpp::Named("m", best_model.m),Rcpp::Named("mesh", best_model.mesh),
                           Rcpp::Named("theta_hat", best_model.theta_hat),
                           Rcpp::Named("y", best_model.y), Rcpp::Named("residuals",residuals),
                             Rcpp::Named("models", models), Rcpp::Named("lambda_minmse_ind",bestModelind+1),
                             Rcpp::Named("cv.mses", mean_mses));
	// Remove allocated memory for cache. NOTE: ARMADILLO OBJECTS ARE DEALLOCATED ONCE OUT OF SCOPE (don't worry about them out of function)
	delete cache;
}
