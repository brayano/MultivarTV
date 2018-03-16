#include <iostream>
#include <armadillo>
#include <cmath>
#include <ctime>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include "solvers.hpp"
#include <vector>
using namespace arma;

/* EXECUTABLE CODE */

int main()
{
	mat X = randu<mat>(10000,2);
	vec y = randu<vec>(10000,1);
	//y.print("y=");
	double lam = 0.9;
	
	vec softy = softthresh(y, lam);
	//softy.print("softy = ");

	vec m = {20,20};
	//mbs_one_object model1;
	//mbs_one(X,y,m, model1); // Everything else goes to default
	//model1.theta_hat.print("thetahat = ");
	//MAT regmesh = create_mesh(X,m);
	//double model1_mse = mbs_mse(model1,y);
	//printf("Model MSE = %f \n", model1_mse);

	mbs_one_object model_tuned;
	std::clock_t start;
	start = std::clock();

	mbs(X,y,m,model_tuned);
	std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
	double train_mse = mse(model_tuned.fitted, y);
	printf("Tuned model Training MSE = %f \n", train_mse);
	//printf("Tuned model MSE = %f \n", model_tuned.minmse);
	//printf("Tuned model Lambda = %f \n", model_tuned.minmse_lambda);
	return 0;
}
