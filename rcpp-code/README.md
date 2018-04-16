MultivarTV
=======
In this directory, we build an R package, "MultivarTV," that implements efficient procedures written in C++ for fitting approximate solutions to multivariate total variation denoising problems. The algorithm uses the alternating direction method of multipliers (ADMM), as described by Boyd et al. (2011).

Package is still in development. Documentation can be accessed without installing R package: MultivarTV.Rcheck/ contains MultivarTV-manual.pdf and MultivarTV-ex.pdf.

In R, "MultivarTV" can do the following (with accompanying caveats):

1. Fit cross-validated solution to multivariate total variation denoising problem via mvtv(). We solve a finite approximation to the total variation problem, which allows for efficient computation while maintaining theoretical guarantees. For sample size n and predictors p, algorithm runs roughly in linear time in the number of optimization parameters, <a href="https://www.codecogs.com/eqnedit.php?latex=\sim&space;n^{\frac{p}{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sim&space;n^{\frac{p}{2}}" title="\sim n^{\frac{p}{2}}" /></a>. Note that this suggests memory issues for large dimension problems.
2. Generate residuals plot for solution to multivariate total variation problem via plotResiduals(). 
3. Plot fitted surface for solution to univariate/bivariate total variation problem via plotFits() (importFrom(scatterplot3d)). 

To-Do List:

1. Use ggplot2 for plotting functions.
2. Create vignette demonstrating novel applications. 

## Installing "MultivarTV"

Users need an installation of [SuperLU 5.2](http://crd-legacy.lbl.gov/~xiaoye/SuperLU/#superlu) (or latest version) present on machine. 

Provided SuperLU is present, then in R:

	require(Rcpp) # Install all dependencies
	require(RcppArmadillo) # Install all dependencies
	install.packages("MultivarTV_1.0.tar.gz",repos=NULL,type="source")

Note that user does not need to install Armadillo >7.5, since RcppArmadillo includes headers for Armadillo within installation. 
