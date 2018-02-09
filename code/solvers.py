import numpy as np
import utils
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csc_matrix

# IN THIS FILE, WE CREATE SOLVERS FOR THE MESH BASED SOLUTION TO P-DIMENSIONAL TOTAL VARIATION PROBLEMS

# SOFT-THRESHOLDING FUNCTION
def softthresh(z,lam):
	sign  = np.sign(z)
	pmax = np.maximum(abs(z)-lam,0)
	return np.multiply(sign,pmax)


def mbs_one(data,y, m, theta_init=None,mesh = None, tune=1.0, eps=0.01, tol=0.001, cache=None):
	# An ADMM solver for the mesh-based approximation to a total variation problem at a fixed tuning parameter.
	# data: (n,p)-array; y: (n ,1)-array; m:(p,)-array 
	# Assume a regular mesh, i.e. 
	# cache=None: some matrix operations are stored to avoid excess computation
	n = y.size
	y = np.array(y).reshape(n,1)
	
	if cache is None:
		if mesh is None:
			meshob = utils.mesh_coords(data,mesh_dims=m)
			mesh = meshob['mesh'] # output as column vector, i.e. for p-covariates, mesh is a tensor, so we flatten into column. 
			deltas = meshob['deltas'] # for x_i, deltas[i] = (max(x_i)-min(x_i))/m_i (for regular mesh)
		# Create O-matrix, i.e. nearest neighbor interpolation matrix: for x_i nearest to theta_j, O_ij = 1, else O_il=0, l != j    
		O = utils.nearest_interp_matrix(data,mesh); Ot = O.transpose()
		# Create D-matrix, i.e. stack of difference matrices providing linear operator on theta for estimating penalty, see utils.py
		D = utils.create_D(dims = m,deltas=deltas); Dt = D.transpose()
		# Store common matrix operations
		crossD = Dt.dot(D); crossO = Ot.dot(O)
		rowsD = D.shape[0]
		cache2 = Ot.dot(y)
		tune = utils.lam_max_pinv(D,Dt,cache2) # if cache is None, tune is None
		crosses = crossO+tune*crossD
		cache1 = splu(crosses.tocsc())
		ntheta = np.prod(m)
		
	else:
		# cache=[cache1,cache2,D,Dt,rowsD, O, Ot,mesh, ntheta]
		cache1 = cache[0]
		cache2 = cache[1]
		D = cache[2]
		Dt = cache[3]
		rowsD = cache[4]
		O = cache[5]
		Ot= cache[6]
		mesh = cache[7]
		ntheta = cache[8]
	
	# Solve convex problem.
	lamda = tune
	rho = tune # Set step size to tuning parameter
	## Initialize
	if theta_init is None:
		theta = np.repeat(np.mean(np.array(y)),ntheta).reshape(ntheta,1)
	else:
		theta = theta_init
	alpha = D.dot(theta) 
	u = np.repeat(1/float(lamda),rowsD).reshape(rowsD,1)
	thetaold = np.repeat(np.mean(np.array(y))-1,ntheta).reshape(ntheta,1)

	counter = 1
	## ADMM requires many iterations, be generous
	maxc = 5000

	while any(abs(theta-thetaold)>tol):
		thetaold = theta; alphaold = alpha; uold = u
		b = (cache2+rho*Dt.dot(alpha+u)).reshape((ntheta,))
		theta = (cache1.solve(b)).reshape((ntheta,1))	
		alpha = softthresh(z=D.dot(theta)-u,lam=lamda/float(rho))
		u = u+alpha-D.dot(theta)
		if counter>maxc:
			raise Exception("Solver did not converge with %s iterations! (N=%s,m=%s)" % (counter,n,m) )
	output = {'mesh': mesh, 'theta.hat': theta,'fitted':O.dot(theta),'data':data,'y':y, 'eps':eps,'m':m,'counter': counter} 
	return output

def mbs_predict(mbs_one_object,data):
	# Given data and estimated theta.hats, calculate fitted values of data. 
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
		deltas = meshob['deltas']

	# Create O-matrix: specify x and number of desired cuts
	O = utils.nearest_interp_matrix(data, mesh ); Ot = O.transpose()
	D = utils.create_D(dims=m,deltas=deltas) ; Dt = D.transpose()
	rowsD  = D.shape[0]
	cache2 = Ot.dot(y)
	crossD = Dt.dot(D)
	crossO = Ot.dot(O)
	if tuners is None:
		lam_max = utils.lam_max_pinv(D,Dt,cache2)*np.prod(deltas)
		tuners = np.exp(np.linspace(np.log(lam_max*10**(-4)),np.log(lam_max),ntune))[::-1] # First we order them in increasing order, then we remove the first element
	
	if ftrue is None:
		ftrue = y
	
	fits = []
	MSEs = []
	counts = []
	thetainit = np.repeat(np.mean(np.array(y)),ntheta).reshape(ntheta,1)
	rho = lam_max

	for i in range(len(tuners)):
		#crosses = crossO+tuners[i]*crossD
		crosses = crossO+rho*crossD
		if i > 0:
			rho = tuners[i-1]
		cache1 = splu(crosses.tocsc())
		fits.append(mbs_one(data=data,y=y, m=m,mesh=mesh,tune=tuners[i],eps=eps,theta_init=thetainit,cache=[cache1,cache2,D,Dt,rowsD, O, Ot,mesh, ntheta]))
		MSEs.append(mbs_mse(mbs_one_object=fits[i],y=ftrue))
		counts.append(fits[i]['counter'])
		thetainit = fits[i]['theta.hat']

	lowestMSE = np.argmin(MSEs)
	output = {'minmse.fits':fits[lowestMSE],'minmse':MSEs[lowestMSE],'minmse.lam': tuners[lowestMSE]}
	return output

