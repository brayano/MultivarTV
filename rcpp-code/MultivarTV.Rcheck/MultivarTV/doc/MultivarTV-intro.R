## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----libsetup, include = FALSE, echo=FALSE-------------------------------
library(plot3D)
library(fields)
library(MultivarTV)
# Towers function
towers <- function(x1,x2){
	n=length(x1)
	y = numeric(n)
	for (i in 1:n){ 
		y[i] = 1*(x1[i]>0.8)*1*(x2[i]>0.8)+0.5*1*(x1[i]>0.8)*1*(x2[i]<0.2)+1*1*(x1[i]<0.2)*1*(x2[i]<0.2)+0.5*1*(x1[i]<0.2)*1*(x2[i]>0.8)
	}
	return(y)
}

pcf <- function(x){
  if (x<0.1) return(0.5)
  else if (x>= 0.1 & x<0.6) return(3)
  else if (x>= 0.6 & x<0.8) return(0.5)
  else return(2)
}
pcwise <- function(vecx){
  y <- numeric(length(vecx))
  for (i in 1:length(vecx)){
    y[i] <- pcf(vecx[i])
  }
  return(y)
}

## ----datagen-------------------------------------------------------------
set.seed(123)
N <- 100
x <- matrix(runif(N),ncol=1)
y <- matrix(pcwise(x) + rnorm(N,0,0.1),ncol=1)

## ----mvtv1, echo=FALSE---------------------------------------------------
set.seed(123) # Setting a seed for cross-validation procedure
m <- matrix(20,ncol=1)
fit <- mvtv(x,y,m,verbose=FALSE) # Verbose = FALSE to suppress algorithm details

## ----fig1, fig.cap = "Total variation solution for piecewise constant function at best cross-validated model. "----
plot(fit)

## ----fig2, fig.cap = "Total variation solution for piecewise constant function at another solution."----
plot(fit, lambda = 2.0)

## ----fig3, fig.cap = "Fused lasso.", echo=FALSE--------------------------
set.seed(123) # Setting a seed for cross-validation procedure
m <- matrix(N,ncol=1)
mesh <- x
fl_mvtv <- mvtv(x,y,m,mesh = mesh, verbose=FALSE) # Verbose = FALSE to suppress algorithm details
plot(fl_mvtv)

library(genlasso)
fl_genlasso <- fusedlasso1d(y)
plot(fl_genlasso)

## ----fig4, echo=FALSE, fig.cap = "Figure 4. Plot of N=100 points drawn from Towers function (drawn as mesh) with noise.", fig.width=4, fig.height = 4----
x1 <- seq(0,1,length.out=40); x2 <- x1
x1x2 <- expand.grid(x1,x2)
y <- matrix(towers(x1x2[,1],x1x2[,2]),nrow=40,ncol = 40)

set.seed(117)
z1 <- runif(100)
z2 <- runif(100)
z3 <- towers(z1,z2)
ynoisy <- z3 + rnorm(length(z3),0,0.5)

scatter3D(z1,z2,ynoisy,theta=30,phi=30,xlab="x1",ylab="x2",zlab="y",cex=1,pch=20,
          surf= list(x = x1,y = x2,z = y,facets=NA,fit = z3))

## ----figure5, echo=TRUE, fig.cap="Figure 5. Total variation solution for noisy sombrero." , fig.width=5, fig.height = 7----
set.seed(117)
mym <- matrix(rep(sqrt(length(ynoisy)),2),ncol=1)
data <- cbind(z1,z2)
tvmod <- mvtv(data,ynoisy,mym,verbose=FALSE)
plot(tvmod,adddata = TRUE)

## ----figure6,echo=TRUE, fig.cap="Figure 6. Thin plate spline solution for noisy sombrero." , fig.width=5, fig.height = 7----
fit <- Tps(data,ynoisy)
out.p <- predictSurface(fit, xy = c(1,2))
plot.surface(out.p, type="p")

## ----figure7, echo=TRUE, fig.cap="Figure 7. Total variation solution for noisier sombrero.", fig.width=5, fig.height = 7----
set.seed(117)
ynoisy2 <- z3 + rnorm(length(z3),0,0.5)

set.seed(117)
tvmod2 <- mvtv(data,ynoisy2,mym,verbose=FALSE)
plot(tvmod,adddata = TRUE)

## ----figure8,echo=TRUE, fig.cap="Figure 8. Thin plate spline solution for noisier sombrero.", fig.width=5, fig.height = 7----
fit2 <- Tps(data,ynoisy2)
out.p2 <- predictSurface(fit2, xy = c(1,2))
plot.surface(out.p2, type="p")

