#include <iostream>
#include <armadillo>
#include <cmath>
#include <string>
#include <cstdlib>
#include <array>
#include <stdexcept>
#include "solvers.hpp"

#define EPS 0.01
using namespace arma; 
typedef std::vector<int> VEC;
typedef fmat MAT;

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
	printf("Created O and D \n");
	inits.crossD = inits.Dt*inits.D; inits.crossO = inits.Ot*inits.O;
	inits.rowsD = inits.D.n_rows;
	printf("Ot*y \n");
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
}

void fill_cache(mbs_cache*& cache, mbs_one_inits inits){
	cache -> O = inits.O;
	cache -> Ot = inits.Ot;
	cache ->D = inits.D;
	cache ->Dt = inits.Dt;
	cache ->rowsD = inits.rowsD;
	cache ->Oty = inits.Oty;
	cache -> ntheta = inits.ntheta;
}

void fill_output_mbs_one(mbs_one_object& output, mat data, vec y, MAT mesh, vec theta, mbs_one_inits inits, vec m){
	output.mesh = mesh; output.theta_hat = theta;
	output.fitted = inits.O*theta; output.data = data; 
	output.y = y; output.m = m; 
}

void adapt_step(vec r_current, vec s_current, double rho_current, vec u_current, adaptstep &object){
	double rho_next;
	double r_norm = sqrt(dot(r_current.t(),r_current));
	double s_norm = sqrt(dot(s_current.t(), s_current));
	//printf("RNORM = %f, SNORM = %f		", r_norm, s_norm);
	if (r_norm > 20*s_norm){
		//printf("div = %f  ", r_norm/s_norm);
		object.rho_next = 20*rho_current;
		object.u_next = 0.05*u_current;
	}
	else if ( s_norm > 20*r_norm){
		object.rho_next = 0.1*rho_current;
		object.u_next = 10*u_current;
	}
	else {
		object.rho_next = rho_current;
		object.u_next = u_current;
	}
}

vec admm_update(vec y, mbs_one_inits inits, vec* theta_init, double lambda){
	// Set-up ADMM
	vec theta(inits.ntheta);
	/// Initialize theta
	if (theta_init == NULL){
		theta.fill(mean(y));
	}
	else{
		theta = *theta_init;
	}
	vec alpha = inits.D*theta;
	vec u(inits.rowsD); u.fill(1/lambda);

	double initval = mean(y)-0.1;
	vec thetaold(inits.ntheta); thetaold.fill(initval);

	int counter = 1;
	int max_counter = 2000;
	int rho = lambda; 
	int rhoold;
	vec alphaold, uold, dual_residual, primal_residual;
	adaptstep stepobject;

	while (any(abs(theta-thetaold) > TOL ) ){
		thetaold = theta; alphaold = alpha; rhoold = rho; uold = u;
		vec b = inits.Oty + rho*inits.Dt*(alpha+u);
		theta = spsolve(inits.sp_crosses,b);
		alpha = softthresh(inits.D*theta-u, lambda/rho);
		dual_residual = rho*inits.Dt*(alpha +u); // -alphaold
		primal_residual = alpha - inits.D*theta;
		u +=  primal_residual;
		counter += 1;
		if (counter>max_counter){
			throw std::invalid_argument("Failed to converge!");
		}
		adapt_step(primal_residual, dual_residual, rho,u, stepobject);
		rho = stepobject.rho_next; u = stepobject.u_next;
	}
	printf("Lambda = %f, Counter = %i \n", lambda, counter);
	return theta;
}

// MBS Solution for single lambda

void mbs_one(mat data, vec y, vec m, mbs_one_object & output, MAT mesh, vec* theta_init /*NULL*/, double lambda /*= 1.0*/, mbs_cache* cache /*= NULL*/){
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
	vec theta = admm_update(y, inits, theta_init, lambda);
	fill_output_mbs_one(output, data, y, mesh, theta, inits, m);
}

vec mbs_predict(mbs_one_object model, mat data){
	sp_mat O = nearest_interp_matrix(data, model.mesh);
	vec fits = O*model.theta_hat;
	return fits;
}

double mse(vec fits, vec y){
	double mseval = sum(pow(fits - y,2))/ y.size();
	return mseval;
}

double mbs_mse(mbs_one_object model, vec y){
	double mseval = mse(model.fitted, y);
	return mseval;
}

void fill_output_mbs(mbs_object &output, vec mse, vec lambdas){
// In the event of a tie, we use the first instance of lowestMSE or  put another way, the largest value of lambda with that lowestMSE
	uvec lowestMSE = find( (mse - min(mse) )== 0);
	output.minmse_model = output.models[lowestMSE[0]];
	output.minmse = mse[lowestMSE[0]]; 
	output.minmse_lambda = lambdas[lowestMSE[0]];
	output.mses = mse;
}

vec create_lambdas(int n_lambda, mbs_one_inits inits, vec* lambdas){
	double lambda_max;
	vec lambdavec;
	if (lambdas == NULL){
		//double deltaprod = prodd(deltas);
		lambda_max = lam_max_pinv(inits.D, inits.Oty);
		lambdavec = flipud(exp(linspace<vec>(log(lambda_max*0.00001), log(lambda_max),n_lambda)));
		printf("lambda_max = %f ", lambda_max);
	}
	else{
		lambdavec = *lambdas;
	}
	return lambdavec;
}

// MBS solutions for a path of lambdas

void mbs_path(mat data, vec y, vec m, MAT mesh, int n_lambda, vec lambdas, vec ftrue, mbs_object &output, mbs_one_inits inits, mbs_cache* cache){
	mbs_one_object tempmodel; 
	vec MSEs(n_lambda);
	vec thetainit(inits.ntheta); thetainit.fill(mean(y));
	
	vec* theta_init; theta_init = new vec(inits.ntheta);
	theta_init = &thetainit; *theta_init = thetainit;

	// Gradient-step size
	double rho; // initialize at first lambda val
	int i;
	for (i=0; i<n_lambda; i++){
		rho = lambdas[i];
		sp_mat sp_crosses = inits.crossO+rho*inits.crossD; cache -> sp_crosses = sp_crosses; // Create/store sum of cross products
		mbs_one(data,y,m, tempmodel,mesh,theta_init,rho,cache);
		output.models.push_back(tempmodel);
		MSEs[i] = mbs_mse(tempmodel, ftrue);
		thetainit = tempmodel.theta_hat; // theta_init points to thetainit
	}
	//printf("PAST N_LAMBDA SOLUTIONS \n");
	fill_output_mbs(output, MSEs, lambdas);
}

// Single function to handle mesh creation

MAT gen_mesh(mat data, vec m, MAT* mesh){
	// IF MISSING, CREATE MESH, I.E. FIND EVENLY SPACED POINTS ACROSS EACH PREDICTOR:
	MAT MESH;
	if (mesh == NULL){
		MESH = create_mesh(data,m);
	}
	else{
		MESH = *mesh;
	}
	return MESH;
}

// DATA WE WILL CALCULATE MSE FOR: 

vec gen_ftrue(vec y, vec* ftrue){
	vec Y;
	if (ftrue == NULL){
		Y = y;
	}
	else{
		Y = *ftrue;
	}
	return Y;
}

// Given matrix of MSEs with columns as lambdas and rows as realizations, fit MBS with best lambda

void mbs_fit_optimal(mat data, vec y, vec m, mbs_one_object &best_model, MAT mesh, vec lambdas, mat mse_mat, mbs_cache* cache, mbs_object path_object, mbs_one_inits inits){
	vec mean_mses = rowmean(mse_mat);
	// FIT OPTIMAL LAMBDA OVER TRAINING SET
	uvec lowestMSE = find( (mean_mses - min(mean_mses) )== 0);
	// MBS_ONE() requires theta_init as pointer
	vec thetainit = path_object.models[lowestMSE[0]].theta_hat;
	vec* theta_init; theta_init = new vec(inits.ntheta);
	theta_init = &thetainit; *theta_init = thetainit;

	double best_lambda = lambdas[lowestMSE[0]];
	
	mbs_one(data,y,m,best_model, mesh, theta_init, best_lambda, cache);
}

// GIVEN SOLUTIONS TO LAMBDAS, PREDICT TESTING DATA AND MEASURE ACCURACY

vec test_mse(mat data, vec y, mbs_object path_object, int n_lambda){
	int i;
	vec mses(n_lambda);
	vec fits(data.n_rows);
	for (i=0; i<n_lambda; i++){
		fits = mbs_predict(path_object.models[i], data);
		mses[i] = mse(fits, y);
	}
	return mses;
}

// MBS Cross-Validation Procedure

void mbs(mat data, vec y, vec m, mbs_one_object &best_model, MAT* mesh /*=NULL*/, int n_lambda /*= 50*/, vec* ftrue/*=NULL*/, vec* lambdas /*= NULL*/, int folds /* =10 */){
	// Initialize objects
	mbs_one_inits inits;
	inits.ntheta = prodd(m);
	inits.deltas = create_deltas(data,m);
	MAT MESH = gen_mesh(data,m,mesh);
	printf("MBS BEGINS: ntheta = %i \n", inits.ntheta);

	// Create cache struct
	mbs_cache* cache = new mbs_cache();
	cache -> ntheta = inits.ntheta;
	create_cache_objects(data,y,MESH, m, inits); // Create inits
	fill_cache(cache,inits); // Create cache using inits
	
	// Create tuning parameters
	vec LAMBDAS = create_lambdas(n_lambda, inits, lambdas);
	vec FTRUE = gen_ftrue(y,ftrue);
	int i;
	mbs_object path_object;
	mat mse_mat(n_lambda, folds);
	kfolds datafolded;
	kfold(folds, data,y,datafolded);
	// CROSS-VALIDATION ON TRAINING DATA: FIND LAMBDA THAT MINIMIZES MSE
	for (i=0; i<folds; i++){
		mbs_path(datafolded.Xtrain[i], datafolded.Ytrain[i], m, MESH, n_lambda, LAMBDAS, FTRUE, path_object, inits, cache);
		printf("Fold = %i Done \n", i+1);
		mse_mat.col(i) = test_mse(datafolded.Xtest[i], datafolded.Ytest[i], path_object, n_lambda);
	}
	// FIT OPTIMAL SOLUTION
	mbs_fit_optimal(data,y,m,best_model,MESH,LAMBDAS, mse_mat, cache, path_object,inits);
	
	// Remove allocated memory for cache. NOTE: ARMADILLO OBJECTS ARE DEALLOCATED ONCE OUT OF SCOPE (don't worry about them out of function)
	delete cache;
}
