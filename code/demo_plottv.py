import numpy as np
import plottv

# Testing 	
# Create bivariate exponential function

def myexp2(args):
	x1 = args[0]
	x2 = args[1]
	z = 2*np.maximum(0,x1+x2)
	y= np.exp(z) - (z+z**2/2+z**3/6)
	return y

np.random.seed([117])
n = 10000
x1 = np.random.uniform(-1,1,n)
x2 = np.random.uniform(-1,1,n)
ytrue = myexp2([x1,x2])
y = ytrue + np.random.normal(0,1,n)
m = np.array([20,20])
plottv.plot2_truesurface(x1,x2,y,m,ftrue=myexp2)
plottv.plot2_fitdata(x1,x2,y,m,ntune=200)
