#' @title MVTV Generic Class
#' @description Defining MVTV Generic Class
#' @name mvtv
#' @param data n by p matrix of data
#' @param ... ignore
#' @export
mvtv <- function(data, ...) UseMethod("mvtv") 

# MVTV Cross-Validation Procedure
#' @title Default Multivariate Total Variation Denoising Solver
#' @description Create a mesh and find cross-validated best 
#' approximation to total variation denoising problem. 
#' @name mvtv.default
#' @param data n by p matrix of inputs
#' @param y response column vector
#' @param ... ignore
#' @param m vector of number of mesh points per predictor 
#' @param mesh user can supply or NULL for regularly spaced mesh, which will be returned
#' @param n_lambda number of logarithmically spaced tuning parameters
#' @param ftrue prediction target. If NULL, use observed data. 
#' @param lambdas user can supply vector of lambdas to be solved over. If NULL,
#' function generates n_lambda logarithmically spaced lambdas from 0.00001*lambda_max
#' and lambda_max, where lambda_max is our approximation of smallest lambda where
#' regularization ends. 
#' @param folds number of folds for cross-validation
#' @param verbose Default: true, prints out current working penalty and 
#' number of iters to solve.
#' @examples
#' # Approximating Bivariate Fused Lasso for Uniform Data 
#' ## Generate Data
#' set.seed(117)
#' x <- matrix(runif(100),ncol = 2)
#' y <- matrix(runif(50),ncol=1)
#' m <- matrix(c(3,3),ncol=1)
#' 
#' ## Find Total Variation Solution over range of lambdas and whole data set
#' mvtv_fold1 <- mvtv(x,y,m,folds=1, verbose = FALSE)
#' 
#' ## Find 5-fold validated MVTV Model over range of lambdas
#' mvtv_fold5 <- mvtv(x,y,m,folds=5, verbose = FALSE)
#' 
#' @export
mvtv.default <- function(data, y, m= NULL, ... , mesh = NULL, n_lambda = 100, ftrue = NULL, lambdas = NULL, folds =5, verbose = TRUE){
  if (is.null(m)){
    mknots <- floor(sqrt(length(y)))
    p <- dim(data)[2]
    m <- matrix(rep(mknots,p), ncol = 1)
  } 
  model <- mvtv_default(data = data, y=y, m=m, mesh=mesh, n_lambda = n_lambda, ftrue=ftrue, lambdas =lambdas, folds=folds, verbose=verbose)
  
  outlambdas <- numeric(n_lambda)
  for (i in 1:n_lambda) outlambdas[i] <- model$models[[i]]$lambda
  model$lambdas <- outlambdas
  
  class(model) <- "mvtv"
  return(model)
}

#' @title Plotting Residuals 
#' @description Plotting residuals for an 'mvtv' Object
#' @name plotResiduals
#' @param mvtvmodel object of class 'mvtv'
#' @export
plotResiduals <- function(mvtvmodel) {
  if (class(mvtvmodel) != "mvtv") stop("Input must be a 'mvtv' model object.")
  
  plot(mvtvmodel$fitted,mvtvmodel$residuals,xlab="Fitted",ylab="Residuals",pch=1,cex=0.5)
  loessob <- loess(mvtvmodel$residuals~mvtvmodel$fitted)
  lines(mvtvmodel$fitted,predict(loessob),col="blue",lwd=1)
  abline(h=0,lty=2,lwd=1)
  
  #print(apply(data,2,mean))
}

#' @title Plotting Fitted Surface, p=1 
#' @description Plotting fitted values for an 'mvtv' Object
#' @name plot.mvtv
#' @param x object of class 'mvtv.'
#' @param ... ignore.
#' @param addmesh If TRUE, vertical grey lines plotted along x-axis value of mesh.
#' @param adddata If TRUE, observed data is plotted.
#' @param lambda Plot at specified lambda. If NULL, plot fit at lambda with smalled cross-validated MSE.
#' @export
plot.mvtv <- function(x, ..., addmesh = FALSE, adddata = TRUE, lambda = NULL) {
  mvtvmodel <- x
  if (class(mvtvmodel) != "mvtv") stop("Input must be a 'mvtv' model object.")
  
  if (!is.null(lambda)){
    nlam <- length(mvtvmodel$models)
    lambdas <- numeric(nlam)
    for (i in 1:nlam) lambdas[nlam-i+1] <- mvtvmodel$lambdas[i]
    if (lambda %in% lambdas){
      ind <- which(lambda==lambdas); mod2plt <- mvtvmodel$models[[ind]]
      mod2plt$mesh <- mvtvmodel$mesh
    }
    else{
      ind <- findInterval(lambda,lambdas) # works for increasing vector
      if (ind == 0) mod2plt <- mvtvmodel$models[[nlam]] # have models in decreasing lambda order
      else if (ind == nlam) mod2plt <- mvtvmodel$models[[1]]
      else{
        mod1 <- mvtvmodel$models[[nlam-ind]]
        mod2 <- mvtvmodel$models[[nlam-ind+1]]
        mod2plt <- list()
        mod2plt$fitted <- rowMeans(cbind(mod1$fitted,mod2$fitted))
        mod2plt$theta_hat <- rowMeans(cbind(mod1$theta_hat,mod2$theta_hat))
      }
      mod2plt$mesh <- mvtvmodel$mesh
    }
  } else{
    mod2plt <- mvtvmodel
  }

  if (dim(mvtvmodel$data)[2] == 1){
    newData <- matrix(seq(min(mvtvmodel$mesh),max(mvtvmodel$mesh),length.out = length(mvtvmodel$y)*10),ncol=1)
    fitData <- predict_mvtv(mvtvobject = mod2plt,data = as.matrix(newData), mesh = mvtvmodel$mesh)
    delta <- mvtvmodel$mesh[2] - mvtvmodel$mesh[1]
    
    plot(fitData~newData,type='l',col="blue", lwd= 2, ylab="y",xlab="x", ylim=c(min(mvtvmodel$y),max(mvtvmodel$y)))
    if (adddata==TRUE) points(mvtvmodel$y~mvtvmodel$data,pch=16,cex=0.5)
    if (addmesh==TRUE) abline(v = mvtvmodel$mesh[-mvtvmodel$m] + delta/2,col="grey",lwd=0.5,lty=2)
  }
  else if (dim(mvtvmodel$data)[2] == 2) {
    newM <- floor(sqrt(length(mvtvmodel$y)))*3
    x1 <- seq(min(mvtvmodel$mesh[,1]),max(mvtvmodel$mesh[,1]),length.out=newM)
    x2 <- seq(min(mvtvmodel$mesh[,2]),max(mvtvmodel$mesh[,2]),length.out=newM)
    gridData <- as.matrix(expand.grid(x1,x2))
    z <- matrix(predict_mvtv(mod2plt, gridData),nrow=newM,ncol=newM)
    if (adddata==FALSE){
      persp3D(x1,x2,z, xlab="x1",ylab="x2",zlab="y",theta=30,phi=30, zlim=c(min(z)-0.1,1.25*max(z)))
    }
    else{
      
      scatter3D(mvtvmodel$data[,1],mvtvmodel$data[,2],mvtvmodel$y,theta=30,phi=30,xlab="x1",ylab="x2",zlab="y",cex=1,pch=20,
                surf= list(x = x1,y = x2,z = z,facets=NA,fit = mod2plt$fitted))
    }
  }
  else {
    stop("Only univariate and bivariate fitting supported.")
  }
}

# MVTV Prediction Function
#' @title MVTV Predict for Fitting Observed/New Data
#' @description Use fitted 'mvtv' object to predict new data.
#' @name predict.mvtv
#' @param object object produced by mvtv.default
#' @param data n by p matrix of inputs
#' @param mesh m by p mesh used by fitting function mvtv
#' @param ... ignore
#' @examples
#' # Approximating Bivariate Fused Lasso for Uniform Data 
#' ## Generate Data
#' set.seed(117)
#' x <- matrix(runif(100),ncol = 2)
#' y <- matrix(runif(50),ncol=1)
#' m <- matrix(c(3,3))
#' 
#' ## Find 5-fold validated MBS Model over range of lambdas
#' mbs_fold5 <- mvtv(x,y,m,folds=5,verbose=FALSE)
#' 
#' # Access fitted values of training data; equivalent to mbs_fold5$fitted
#' fitted.values <- predict(mbs_fold5) 
#' newdata <- matrix( runif(50), ncol = 2) # Generate new data
#' newfits <- predict(mbs_fold5, newdata) # Fit new data
#' @export
predict.mvtv <- function(object, data=NULL, mesh=NULL, ...){
  predicted <- predict_mvtv(mvtvobject=object, data=data,mesh=mesh)
  return(predicted)
}

#' @useDynLib MultivarTV
#' @importFrom Rcpp sourceCpp 
#' @importFrom graphics abline lines plot points
#' @importFrom stats loess predict
#' @importFrom plot3D scatter3D persp3D
NULL