MultivarTV
=======
In this directory, we build an R package, "MultivarTV," that implements efficient procedures written in C++ for fitting approximate solutions to multivariate total variation denoising problems. The algorithm uses the alternating direction method of multipliers (ADMM), as described by Boyd et al. (2011).

Package is still in development. Documentation can be accessed without installing R package: MultivarTV.Rcheck/ contains MultivarTV-manual.pdf and MultivarTV-ex.pdf.

In R, "MultivarTV" can do the following (with accompanying caveats):

1. Fit cross-validated solution to multivariate total variation denoising problem via mvtv(). We solve a finite approximation to the total variation problem, which allows for efficient computation while maintaining theoretical guarantees. For sample size n and predictors p, iterations of ADMM solve in linear time, $O(sqrt(n)^p)$. Note that this suggests memory issues for large dimension problems.
2. Generate residuals plot for solution to multivariate total variation problem via plotResiduals(). 
3. Plot fitted surface for solution to univariate total variation problem via plotFits(). 

To-Do List:

1. Expand plotFits() to also plot bivariate surface. 
2. Create vignette demonstrating novel applications. 

## Installing "MultivarTV"

Users need an installation of [SuperLU 5.2](http://crd-legacy.lbl.gov/~xiaoye/SuperLU/#superlu) (or latest version) present on machine. 

Provided SuperLU is present, then in R:

	require(Rcpp) # Install all dependencies
	require(RcppArmadillo) # Install all dependencies
	install.packages("MultivarTV_1.0.tar.gz",repos=NULL,type="source")

Note that user does not need to install Armadillo >7.5, since RcppArmadillo includes headers for Armadillo within installation. 
