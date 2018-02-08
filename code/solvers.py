import numpy as np
import cvxpy as cvx
import cvxopt as cvxopt
import scipy as scipy
import utils
import math
import time # USeful for clocking
from scipy.sparse.linalg import spsolve, inv, splu
from scipy.sparse import coo_matrix, vstack, csr_matrix, csc_matrix

# IN THIS FILE, WE CREATE SOLVERS FOR THE MESH BASED SOLUTION TO P-DIMENSIONAL TOTAL VARIATION PROBLEMS

# SOFT-THRESHOLDING FUNCTION
def softthresh(z,lam):
	sign  = np.sign(z)
	pmax = np.maximum(abs(z)-lam,0)
	#print type(sign), sign.shape, type(pmax), pmax.shape
	return np.multiply(sign,pmax)

# PSEUDO-INVERSE FUNCTION
def mypinv(spm, spmt, Oty):
	# spm: sparse matrix; spmt: transpose of sparse matrix
	#return (spmt.dot(inv(spm.dot(spmt)))).dot(spm)
	rows = Oty.shape[0]
	aprime = spmt.dot(spm)
	#print spm
	#print aprime.transpose()
	a = splu(aprime.tocsc())
	#print a.shape, Oty.shape
	b = a.solve(Oty.reshape((rows,)))
	#return inv(spmt.dot(spm)).dot(spmt)
	return spm.dot(b)

def mbs_one(data,y, m, theta_init=None,mesh = None, tune=1.0, eps=0.01, tol=0.0001, cache=None):
	# An ADMM solver for the mesh-based approximation to a total variation problem at a fixed tuning parameter.
	# data: (n,p)-array; y: (n ,1)-array; m:(p,)-array 
	# Assume a regular mesh, i.e. 

	n = y.size
	y = np.array(y).reshape(n,1)
	
	if cache == None:
		# Create O-matrix: specify x and number of desired cuts
		# Create D-matrix: specify n and k
		if mesh is None:
			mesh = utils.mesh_coords(data,mesh_dims=m)
		O = utils.nearest_interp_matrix(data,mesh)
		D = utils.create_D(dims = m)
		Dt = D.transpose()
		Ot = O.transpose()
		rowsD = D.shape[0]
		cache1 = splu(Ot.dot(O)+tuners[i]*Dt.dot(D))
		cache2 = Ot.dot(y)
		ntheta = np.prod(m)
		
	else:
		cache1 = cache[0]
		cache2 = cache[1]
		D = cache[2]
		Dt = cache[3]
		rowsD = cache[4]
		O = cache[5]
		Ot= cache[6]
		mesh = cache[7]
		ntheta = cache[8]

	if tune is None:
		A = mypinv(D, Dt, cache2)
		tune = np.max(abs(A))
	
	# Solve convex problem.
	lamda = tune
	rho = tune
	## Initialize
	#print theta_init
	if theta_init is None:
		theta = np.repeat(np.mean(np.array(y)),ntheta).reshape(ntheta,1)
	else:
		theta = theta_init
	#print D.shape, theta.shape
	alpha = D.dot(theta) #.reshape(m-k-1,1)
	u = np.repeat(1/float(lamda),rowsD).reshape(rowsD,1)
	thetaold = np.repeat(np.mean(np.array(y))-1,ntheta).reshape(ntheta,1)

	counter = 1
	maxc = 10000
	#times = []

	while any(abs(theta-thetaold)>tol):
	#while (counter<=400):
		thetaold = theta
		alphaold = alpha
		uold = u
		#start = time.time()
		#theta = cache1.dot(O.T.dot(y)+rho*D.T.dot(alpha+u))
		b = (cache2+rho*Dt.dot(alpha+u)).reshape((ntheta,))
		theta = (cache1.solve(b)).reshape((ntheta,1))
		#theta = spsolve(cache1,O.T.dot(y)+rho*D.T.dot(alpha+u)).reshape((m**2,1))
		#print counter, theta.shape, thetaold.shape	
		#print any(abs(theta-thetaold)>tol)	
		alpha = softthresh(z=D.dot(theta)-u,lam=lamda/float(rho))
		u = u+alpha-D.dot(theta)
		#end = time.time()
		#times.append(end-start)
		counter = counter+1
		if counter>maxc:
			raise Exception("Solver did not converge with %s iterations! (N=%s,m=%s)" % (counter,n,m) )
	#print "For lambda= %s, average cycle time was %s s" % (tune, np.mean(times))
	output = {'mesh': mesh, 'theta.hat': theta,'fitted':O.dot(theta),'data':data,'y':y, 'eps':eps,'m':m,'counter': counter} 
	#print counter 
	return output

def mbs_predict(mbs_one_object,data):
	O = utils.nearest_interp_matrix(data,mbs_one_object['mesh'])
	return O.dot(mbs_one_object['theta.hat'])

def mbs_mse(mbs_one_object,y):
	yhat = mbs_one_object['fitted']
	yhat = yhat.reshape((yhat.size,))
	ytrue = y.reshape((y.size,))
	return np.sum((yhat-ytrue)**2)/y.size

def mbs(data,y,m,ftrue=None,mesh=None,ntune=100,tuners=None,eps=0.01):
	# mbs is an iterator on mbs_one, where the user supplies only m=(m_1,m_2,...,m_p) 
	# The tuning parameter is not supplied. Instead, mbs()
	# finds the lambda which minimizes MSE, based on either user supplied lambda
	# or number of lambda. Lambda_max is determined analytically. 
	# ntune must be supplied!
	# Create D-matrix: specify n and k
	n = data.shape[0]
	y = y.reshape((n,1))
	ntheta = np.prod(m)

	# Create D-matrix: specify n and k	
	if mesh is None:
		meshob = utils.mesh_coords(data,mesh_dims=m)
		mesh = meshob['mesh']
		#print mesh
		deltas = meshob['deltas']

	# Create O-matrix: specify x and number of desired cuts
	#start = time.time()
	O = utils.nearest_interp_matrix(data, mesh )
	#print data[5], mesh[0]
	#print O # csc_matrix organizes the non-zero values by column
	#end = time.time()
	#print "O: %s" % (end-start)
	Ot = O.transpose()
	#start = time.time()
	D = utils.create_D(dims=m,deltas=deltas) # NEED TO INCORPORATE DELTAS!!
	#end = time.time()
	#print "D: %s" % (end-start)
	#start = time.time()
	Dt = D.transpose()
	#end = time.time()
	#print "Dt: %s" % (end-start)
	#D2 = csr_matrix(D)
	#D_coo = D.tocoo()
	#D = cvxopt.spmatrix(D_coo.data,D_coo.row.tolist(), D_coo.col.tolist())
	rowsD  = D.shape[0]
	#start = time.time()
	cache2 = Ot.dot(y)
	#print cache2
	#end = time.time()
	#print "cache2: %s" % (end-start)
	#start = time.time()
	crossD = Dt.dot(D)
	#end = time.time()
	#print "crossD: %s" % (end-start)
	#start = time.time()
	crossO = Ot.dot(O)
	#end = time.time()
	#print "crossO: %s" % (end-start)
	#print crack
	if tuners is None:
		#start = time.time()
		A = mypinv(D, Dt, cache2)
		lam_max = np.max(abs(A))		
		#end = time.time()
		#print "Lam_max= %s, Time = %s, s" % (lam_max, end-start)
		#print crack
		tuners = np.exp(np.linspace(np.log(lam_max*10**(-6)),np.log(lam_max),ntune))[::-1] # First we order them in increasing order, then we remove the first element
		#print tuners
		#tuners=np.exp(np.linspace(-4,5,ntune))[::-1]
		#tuners=np.exp(np.linspace(np.log(30*10**(-6)),np.log(2),ntune))[::-1]
	
	if ftrue is None:
		ftrue = y
	
	fits = []
	MSEs = []
	counts = []
	thetainit = np.repeat(np.mean(np.array(y)),ntheta).reshape(ntheta,1)

	for i in range(len(tuners)):
		#start = time.time()
		#cache1 = np.linalg.inv(O.T.dot(O)+tuners[i]*D.T.dot(D))
		#cache1 = splu(O.T.dot(O)+tuners[i]*D.T.dot(D))
		#print i
		crosses = crossO+tuners[i]*crossD
		cache1 = splu(crosses.tocsc())
		#print i+1
		#print type(O.T.tocsc()), type(D), type(cache1)
		#print crack
		#end = time.time()
		#print "lambda: %s, cache: %s s" % (tuners[i], end-start)
		#cache1 = csr_matrix(O.T.dot(O)+tuners[i]*D.T.dot(D))
		#start = time.time()
		fits.append(mbs_one(data=data,y=y, m=m,mesh=mesh,tune=tuners[i],eps=eps,theta_init=thetainit,cache=[cache1,cache2,D,Dt,rowsD, O, Ot,mesh, ntheta]))
		#end = time.time()
		#print "Iter %s : %s s; lambda = %s" % (i,end-start, tuners[i])
		#start = time.time()
		MSEs.append(mbs_mse(mbs_one_object=fits[i],y=ftrue))
		#end = time.time()
		#print end-start
		counts.append(fits[i]['counter'])
		thetainit = fits[i]['theta.hat']

	lowestMSE = np.argmin(MSEs)
	#print "ADMM Counts: Min %s, Max %s, Mean %s" % (np.min(counts), np.max(counts), np.mean(counts))
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

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
#print x1
#print x2
data = np.concatenate((x1.reshape((n,1)),x2.reshape((n,1))),1)
#print data
ytrue = myexp2([x1,x2])
y = ytrue + np.random.normal(0,1,n)
#print np.mean(y)
#print y.reshape((100,1))
m = np.array([20,20])
#start = time.time()
#test = mbs(data,y,m,ftrue = ytrue, ntune=100)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (m[0],end-start,test['minmse'],test['minmse.lam'])
#start = time.time()
#t1 = meshy2(x1=x1,x2=x2,y=y,m=30,k1=[0],k2=[0],ftrue=ytrue,interp=0,ntune=50)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (30,end-start,t1['minmse'],t1['minmse.lam'])
#print crack

#start = time.time()
#t1 = meshy2(x1=x1,x2=x2,y=y,m=60,k1=[0],k2=[0],ftrue=ytrue,interp=0,ntune=50)
#end = time.time()
#print "m=%s, time = %s s, MSE: %s, minlam = %s" % (60,end-start,t1['minmse'],t1['minmse.lam'])
