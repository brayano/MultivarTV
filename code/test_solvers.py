import numpy as np
import time 
import solvers

# Fit noisy bivariate exponential function
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
data = np.concatenate((x1.reshape((n,1)),x2.reshape((n,1))),1)
ytrue = myexp2([x1,x2])
y = ytrue + np.random.normal(0,1,n)
m = np.array([10,10])

# Unit testing mbs_one():

def test_mbs_one():
	object1 = solvers.mbs_one(data,y,m) # This defaults to use lambda=lambda_max, so we expect theta.hat=np.mean(y)
	a= np.round(np.mean(object1['theta.hat']),3)
	b=np.round(np.mean(object1['fitted']),3)
	c=np.round(np.mean(y ),3)
	assert a==b==c
 
# Unit testing mbs():

#m = np.array([50,50])
#start = time.time()
#object2 = solvers.mbs(data,y,m,ftrue = ytrue, ntune=50)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (m[0],end-start,object2['minmse'],object2['minmse.lam'])
#start = time.time()

#t1 = meshy2(x1=x1,x2=x2,y=y,m=30,k1=[0],k2=[0],ftrue=ytrue,interp=0,ntune=50)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (30,end-start,t1['minmse'],t1['minmse.lam'])
#print crack

#start = time.time()
#t1 = meshy2(x1=x1,x2=x2,y=y,m=60,k1=[0],k2=[0],ftrue=ytrue,interp=0,ntune=50)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (60,end-start,t1['minmse'],t1['minmse.lam'])
