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
#' @param mvtvmodel object of class 'mvtv.'
#' @param addmesh If TRUE, vertical grey lines plotted along x-axis value of mesh.
#' @param adddata If TRUE, observed data is plotted.
#' @export
plotFits <- function(mvtvmodel, addmesh = FALSE, adddata = TRUE) {
  if (class(mvtvmodel) != "mvtv") stop("Input must be a 'mvtv' model object.")
  if (dim(mvtvmodel$data)[2] == 1){
    newData <- matrix(seq(min(mvtvmodel$mesh),max(mvtvmodel$mesh),length.out = length(mvtvmodel$y)*10),ncol=1)
    fitData <- predict(mvtvobject = mvtvmodel,data = as.matrix(newData), mesh = mvtvmodel$mesh)
    delta <- mvtvmodel$mesh[2] - mvtvmodel$mesh[1]
    
    plot(fitData~newData,type='l',col="blue", lwd= 2)
    if (adddata==TRUE) points(mvtvmodel$y~mvtvmodel$data,ylab="y",xlab="x",pch=16,cex=0.5)
    if (addmesh==TRUE) abline(v = mvtvmodel$mesh[-mvtvmodel$m] + delta/2,col="grey",lwd=0.5,lty=2)
  }
  else if (dim(mvtvmodel$data)[2] == 2) {
    newM <- sqrt(length(mvtvmodel$y))*3
    newData <- gen_mesh(data = mvtvmodel$data,m = matrix(rep(newM,2)),mesh = NULL)
    newZ <- predict(mvtvmodel, newData)
    s3d <- scatterplot3d(newData[,1], newData[,2], newZ, highlight.3d = TRUE, 
                         pch=20,xlim = c(min(mvtvmodel$mesh[,1]),max(mvtvmodel$mesh[,1])),
                         ylim=c(min(mvtvmodel$mesh[,2]),max(mvtvmodel$mesh[,2])),
                         zlim=c(min(mvtvmodel$y),max(mvtvmodel$y)),
                         xlab = "x1",ylab = "x2",zlab = "y")
    if (adddata==TRUE) s3d$points3d(x = mvtvmodel$data[,1],y = mvtvmodel$data[,2],z = mvtvmodel$y, pch = 16)
  }
  else {
    stop("Only univariate and bivariate fitting supported.")
  }
}

#' @useDynLib MultivarTV
#' @importFrom Rcpp sourceCpp 
#' @importFrom graphics abline lines plot points
#' @importFrom stats loess predict
#' @importFrom scatterplot3d scatterplot3d
NULL