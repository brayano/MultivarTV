#' @title MVTV Generic Class
#' @description Defining MVTV Generic Class
#' @name mvtv
#' @param data n by p matrix of data
#' @param ... ignore
#' @export
mvtv <- function(data,...) UseMethod("mvtv") 

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
#' @name plotFits
#' @param mvtvmodel object of class 'mvtv'
#' @param addmesh If TRUE, plot has vertical grey lines along x-axis value of mesh
#' @export
plotFits <- function(mvtvmodel, addmesh = FALSE) {
  if (class(mvtvmodel) != "mvtv") stop("Input must be a 'mvtv' model object.")
  
  newData <- matrix(seq(min(mvtvmodel$mesh),max(mvtvmodel$mesh),length.out = 1000),ncol=1)
  fitData <- predict(mvtvobject = mvtvmodel,data = as.matrix(newData), mesh = mvtvmodel$mesh)
  delta <- mvtvmodel$mesh[2] - mvtvmodel$mesh[1]
  
  plot(mvtvmodel$y~mvtvmodel$data,ylab="y",xlab="x",pch=16,cex=0.5)
  lines(fitData~newData,col="blue", lwd= 2)
  if (addmesh==TRUE) abline(v = mvtvmodel$mesh[-mvtvmodel$m] + delta/2,col="grey",lwd=0.5,lty=2)
}

#' @useDynLib MultivarTV
#' @importFrom Rcpp sourceCpp 
#' @importFrom graphics abline lines plot
#' @importFrom stats loess predict
NULL