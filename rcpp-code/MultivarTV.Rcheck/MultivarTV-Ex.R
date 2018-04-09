pkgname <- "MultivarTV"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
base::assign(".ExTimings", "MultivarTV-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('MultivarTV')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
cleanEx()
nameEx("MultivarTV-package")
### * MultivarTV-package

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: MultivarTV-package
### Title: Mesh Based Solutions to Multivariate Total Variation Problems
### Aliases: MultivarTV-package MultivarTV
### Keywords: package

### ** Examples

  ## Optional simple examples of the most important functions
  ## Use \dontrun{} around code to be shown but not executed



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("MultivarTV-package", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("mvtv.default")
### * mvtv.default

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: mvtv.default
### Title: Default Multivariate Total Variation Denoising Solver
### Aliases: mvtv.default

### ** Examples

# Approximating Bivariate Fused Lasso for Uniform Data 
## Generate Data
set.seed(117)
x <- matrix(runif(100),ncol = 2)
y <- matrix(runif(50),ncol=1)
m <- matrix(c(3,3))

## Find Total Variation Solution over range of lambdas and whole data set
mvtv_fold1 <- mvtv(x,y,m,folds=1, verbose = FALSE)

## Find 5-fold validated MVTV Model over range of lambdas
mvtv_fold5 <- mvtv(x,y,m,folds=5, verbose = FALSE)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("mvtv.default", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("predict.mvtv")
### * predict.mvtv

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: predict.mvtv
### Title: MVTV Predict Function
### Aliases: predict.mvtv

### ** Examples

# Approximating Bivariate Fused Lasso for Uniform Data 
## Generate Data
set.seed(117)
x <- matrix(runif(100),ncol = 2)
y <- matrix(runif(50),ncol=1)
m <- matrix(c(3,3))

## Find 5-fold validated MBS Model over range of lambdas
mbs_fold5 <- mvtv(x,y,m,folds=5,verbose=FALSE)

# Access fitted values of training data; equivalent to mbs_fold5$fitted
fitted.values <- predict(mbs_fold5) 
newdata <- matrix( runif(50), ncol = 2) # Generate new data
newfits <- predict(mbs_fold5, newdata) # Fit new data



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("predict.mvtv", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
