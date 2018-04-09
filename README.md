MultivarTV
======

In this repository, we build mesh-based solutions (MBS) to multivariate total variation (TV) denoising problems. These efficient procedures written in C++ fit approximate solutions to multivariate total variation denoising problems. The algorithm uses the alternating direction method of multipliers (ADMM), as described by Boyd et al. (2011).

In cpp-code/, we have written our C++ procedures for solving total variation denoising. Our main goal is to port this C++ code to R and Python.

Python implementation for MultivarTV can be found in code/. Note that soon we will port C++ implementations to Python. 

In rcpp-code/, we develop the R package "MultivarTV." 
