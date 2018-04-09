library(Rcpp)
library(RcppArmadillo)
# From scratch
setwd("~/Dropbox/projects/mbs-tv/rcpp-code")
# Uncomment following line to create new package
#RcppArmadillo::RcppArmadillo.package.skeleton("MultivarTV")

# Did you change code?
Rcpp::compileAttributes("MultivarTV")
# Did you add oxygen?
devtools::document("MultivarTV")
# Check package (this sweaves the manual)
system("R CMD check MultivarTV")
# Toy with code locally
Sys.setenv("PKG_LIBS"="-lsuperlu")
#Rcpp::sourceCpp("~/Dropbox/projects/mbs-tv/rcpp-code/tvR/src/solvers.cpp",verbose=TRUE,rebuild=TRUE)
#Rcpp::sourceCpp("~/Dropbox/projects/mbs-tv/rcpp-code/tvR/src/tvR.cpp",verbose=TRUE,rebuild=TRUE)
Rcpp::sourceCpp("~/Dropbox/projects/mbs-tv/rcpp-code/MultivarTV/src/MultivarTV.cpp",verbose=TRUE,rebuild=TRUE)

#remove.packages("Rmbs2tv")

pcf <- function(x){
  if (x<0.1) return(0.5)
  else if (x>= 0.1 & x<0.6) return(0.9)
  else return(0.1)
}
vecpcf <- function(vecx){
  y <- numeric(length(vecx))
  for (i in 1:length(vecx)){
    y[i] <- pcf(vecx[i])
  }
  return(y)
}

## Testing Code ##
# I need help to understand why CG would fail in univariate
set.seed(117)
n <- 50
p <- 2
m <- 3
data <- matrix(runif(n*p),ncol = p)
#data <- matrix(sort(data), ncol = p)
#y <- vecpcf(data) + rnorm(n,sd = 0.01)
y <- matrix(runif(n),ncol=1)
m <- matrix(rep(m,p))
mesh <- gen_mesh(data,m,NULL)
#mymbs <- mbs_impl(data,y,m,NULL,100,NULL,NULL,5)
set.seed(123) # cvfold will yield different stuff without seed
mvtv_fold1 <- mvtv(data,y,m,folds=1, verbose = FALSE)
mvtv_fold1_nlam1 <- mvtv(data,y,m, n_lambda = 3, lambdas = c(0.2,1.0,1.5), folds=1)
mvtv_fold1_nlam1 <- mvtv(data,y,m, n_lambda = 1, lambdas = 1.0, folds=1)

mymbs <- mvtv(data,y,m,mesh,verbose = TRUE)
#mbsone <- mvtv(data,y,m,mesh,n_lambda = 1,lambdas = 5.0,folds=1,verbose=FALSE)
#mbspath <- mvtv(data,y,m,mesh,folds=1)
#newdata <- matrix(runif(50),ncol=1)
#newdata <- matrix(seq(0,1,0.001),ncol=1)
#fits <- predict(mymbs)
#newfits <- predict.mvtv(mvtvobject = mymbs,data = newdata)
#plot(data,y,pch=16,cex=0.3)
#lines(newdata,newfits,col="blue",lwd=2)
#abline(v = c(0.25,0.75),lty=2,col='grey',lwd=0.1)
plotResiduals(mymbs)
plotFits(mymbs,addmesh = TRUE)
#delta = mymbs$mesh[2] - mymbs$mesh[1]
#abline(v = mymbs$mesh -delta/2)
#plot(mymbs$fitted,mymbs$residuals,xlab="Fitted",ylab="Residuals",pch=1,cex=0.5)
#loessob <- loess(mymbs$residuals~mymbs$fitted)
#lines(mymbs$fitted,predict(loessob),col="blue",lwd=1)
#abline(h=0,lty=2,lwd=1)
## END Testing Code ##

# ONCE CONFIDENT! THAT CODE IS AT A SAFE PLACE
#library(tools)
#package_native_routine_registration_skeleton(dir = "~/Dropbox/projects/mbs-tv/rcpp-code/Rmbs2tv",character_only = FALSE)
# if you change code EVER AGAIN take special care of init.c... 

# NOW BUILD
system("R CMD build MultivarTV")

system("R CMD check --as-cran MultivarTV_1.0.tar.gz")
install.packages("MultivarTV_1.0.tar.gz",repos=NULL,type="source")
library(MultivarTV)
library(RcppArmadillo)
library(Rcpp)
## Testing Package ##
set.seed(117)
n <- 50
p <- 3
data <- matrix(runif(p*n),ncol = p)
data <- as.matrix(data)
y <- matrix(runif(n),ncol=1)
y <- as.numeric(y)
m <- matrix(rep(3,p)); m <- as.numeric(m)
mesh <- gen_mesh(data,m,NULL)
#mymbs <- mbs_impl(data,y,m,NULL,100,NULL,NULL,5)
set.seed(123) # cvfold will yield different stuff without seed
mymbs <- mbs.default(data,y,m,mesh)
mbsone <- mbs.default(data,y,m,mesh,n_lambda = 1,lambdas = 5.0,folds=1)
mbspath <- mbs.default(data,y,m,mesh,folds=1)
newdata <- matrix(runif(50),ncol=2)
fits <- mbs.predict(mymbs)
newfits <- mbs.predict(mymbs,newdata)

set.seed(117)
n <- 1000
p <- 1
m <- 50
data <- matrix(runif(n*p),ncol = p)
data <- matrix(sort(data), ncol = p)
y <- vecpcf(data) + rnorm(n,sd = 0.01)
#y <- matrix(runif(n),ncol=1)
m <- matrix(rep(m,p))
mesh <- gen_mesh(data,m,NULL)
#mymbs <- mbs_impl(data,y,m,NULL,100,NULL,NULL,5)
set.seed(123) # cvfold will yield different stuff without seed
mymbs <- mvtv(data,y,m,mesh,folds = 10,verbose = TRUE)
plotResiduals(mymbs)
# Only if univariate
plotFits(mymbs,addmesh = TRUE)
?mvtv.default
?predict.mvtv
?gen_mesh
## END Testing Package ##
