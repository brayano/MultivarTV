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

## ----eval=FALSE----------------------------------------------------------
#  # Note that we use gen_mesh() in MultivarTV package to generate function.
#  dat <- seq(0+0.0001,1-0.0001,length.out = 5)
#  mat <- cbind(dat,dat)
#  m <- matrix(rep(30,2),ncol=1)
#  dataMesh <- gen_mesh(mat,m,NULL )
#  y <- towers(dataMesh[,1],dataMesh[,2])
#  set.seed(117)
#  ynoisy <- y + rnorm(length(y),0,0.1)

## ----fig1, echo=FALSE, fig.cap = "Figure 1. Plot of N=100 points drawn from Towers function (drawn as mesh) with noise.", fig.width=4, fig.height = 4----
x1 <- seq(0,1,length.out=40); x2 <- x1
x1x2 <- expand.grid(x1,x2)
z <- matrix(towers(x1x2[,1],x1x2[,2]),nrow=40,ncol = 40)
#y <- outer(x1,x2,towers)
set.seed(117)
z1 <- runif(100)
z2 <- runif(100)
z3 <- towers(z1,z2)
ynoisy <- z3 + rnorm(length(z3),0,0.5)

scatter3D(z1,z2,ynoisy,theta=30,phi=30,xlab="x1",ylab="x2",zlab="y",cex=1,pch=20,
          surf= list(x = x1,y = x2,z = z,facets=NA,fit = z3))


## ----figure2, echo=TRUE, fig.cap="Figure 2. Total variation solution for noisy sombrero." , fig.width=5, fig.height = 7----
set.seed(117)
mym <- matrix(rep(sqrt(length(ynoisy)),2),ncol=1)
data <- cbind(z1,z2)
tvmod <- mvtv(data,ynoisy,mym,verbose=FALSE)
plot(tvmod,adddata = TRUE)

## ----figure3,echo=TRUE, fig.cap="Figure 3. Thin plate spline solution for noisy sombrero." , fig.width=5, fig.height = 7----
fit <- Tps(data,ynoisy)
out.p <- predictSurface(fit, xy = c(1,2))
plot.surface(out.p, type="p")

## ----figure4, echo=TRUE, fig.cap="Figure 4. Total variation solution for noisier sombrero.", fig.width=5, fig.height = 7----
set.seed(117)
ynoisy2 <- z3 + rnorm(length(z3),0,0.5)

set.seed(117)
tvmod2 <- mvtv(data,ynoisy2,mym,verbose=FALSE)
plot(tvmod,adddata = TRUE)

## ----figure5,echo=TRUE, fig.cap="Figure 5. Thin plate spline solution for noisier sombrero.", fig.width=5, fig.height = 7----
fit2 <- Tps(data,ynoisy2)
out.p2 <- predictSurface(fit2, xy = c(1,2))
plot.surface(out.p2, type="p")

