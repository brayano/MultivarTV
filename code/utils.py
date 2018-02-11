import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, csc_matrix, vstack, linalg as sla
from scipy.sparse.linalg import spsolve, inv, splu
import cvxopt as cvxopt
import math
import itertools
from operator import add, sub

# Tensor to Vector Function

def t2v_unit(ind,dims): # Strictly for unit testing
	# Examples: For 3x3 mesh (bivariate case, 9 thetas), t2v_ind([0,0]) = 0, t2v_ind([1,0])=1, ..., t2v_ind([2,2]) = 8
	val = ind[0]
	for i in range(1,dims.shape[0]):
		val = val + ind[i]*np.prod(dims[:i])
	return val

def t2v(dims):
	# Closure: outputs function handling tensor to vector mappings given multi-index
	def t2v_ind(ind):
		# Examples: For 3x3 mesh (bivariate case, 9 thetas), t2v_ind([0,0]) = 0, t2v_ind([1,0])=1, ..., t2v_ind([2,2]) = 8
		val = ind[0]
		for i in range(1,dims.shape[0]):
			val = val + ind[i]*np.prod(dims[:i])
		return val
	
	return t2v_ind

# Vector to tensor function

def v2t_unit(ind,dims): # Version for unit testing
	vals = range(dims.shape[0])
	i = dims.shape[0]
	ind = float(ind+1)
	while i > 0:
		rel_prod = np.prod(dims[:i-1]); 
		vals[i-1] = int(math.ceil(ind/rel_prod))-1
		ind = ind-(vals[i-1])*rel_prod
		i = i -1
	return np.asarray(vals)

def v2t(dims):
	# Takes the dimensions of the tensor, outputs a function that takes vector position and outputs tensor position
	def v2t_ind(ind):
		vals = range(dims.shape[0])
		i = dims.shape[0]
		ind = float(ind+1)
		while i > 0:
			rel_prod = np.prod(dims[:i-1]); #print rel_prod
			vals[i-1] = int(math.ceil(ind/rel_prod))-1
			ind = ind-(vals[i-1])*rel_prod
			i = i -1
		return np.asarray(vals)
	return v2t_ind

# Getting all first order difference binaries for p-dimension into a list without using python list comprehension

#def fd_binaries(p):
#	lst = [list(i) for i in itertools.product([0,1],repeat=p)][1:] # the 0th element is [0,0,0] for p=3, [0,0] p=2, etc. 
#	return np.array(lst)

def fd_binaries(p):
	bins = []
	for i in range(1<<p):
		s=bin(i)[2:] # bin(i) grabs the ith binary as a string, bin(0)='0', bin(1)='1' (01), bin(2)='10', bin(3)='11', bin(4)='100', etc.
		s='0'*(p-len(s))+s # push the hidden zeros back into the string
		bins.append( map(int,list(s)))
	return np.array(bins[1:])

def calc_alpha(bins,dims):
	# Calculates all differences needed to be summed to evaluate an approximate penalty, not clear if useful yet. 
	nbins = len(bins)
	alpha = 0
	for i in range(nbins):
		alpha = alpha + np.prod([i-j for i, j in zip(dims,bins[i])])
	return alpha

# For any particular set of partial derivatives we want, we aim to approximate them using differences. 
# We need to create a function that returns a difference matrix for differences along a particular dimension of the array
# i.e. if direction = 0, then first differences along the rows of the matrix
# if direciton = 1, then along the columns
# Inputes given as arrays/vectors

def build_diffmat(dim,direction):
	col_ind = []; row_ind = []; val = [] # create empty lists for non-zero positions in csc_matrix
	tensor_pointer = t2v(dim) ; vector_pointer = v2t(dim) 
	
	ind = np.zeros(dim.shape)
	row = 0
	
	for i in range(int(np.prod(dim))): 
		if not ind[direction] + 1 >= dim[direction]:
			col_ind.append(tensor_pointer(ind)); row_ind.append(row); val.append(1.0)
			ind[direction] += 1; #print ind # assigned too early before
			col_ind.append(tensor_pointer(ind)); row_ind.append(row); val.append(-1.0)
			row += 1
		ind = vector_pointer( i + 1)
	D = csc_matrix((val,(row_ind,col_ind)))
	return D

def mixedpartial(dims, binary):
	diffmats_list = []
	# I want to know which partial derivatives we are approximating given the binary
	indices = []
	for i in range(dims.shape[0]):
		if binary[i] == 1:
			indices.append(i)
	# For p-covariates and q<p many partials, we will have D_1,...,D_q many difference matrices.
	# For D_1, build_diffmat() requires the full dimensions
	# However, for each subsequent difference matrix, dims in build_diffmat() reduces in the index where we last took a partial/difference approximation. 
	# myidents will be the way we store those reductions in dimension. We update dims at each pass. 
	myidents = np.identity(dims.shape[0])[indices]
	for j in range(len(indices)):
		if j == 0:
			diffmats_list.append(build_diffmat(dims,direction = 0))
		else:
			dims = dims - myidents[j-1]
			diffmats_list.append(build_diffmat(dims,direction = indices[j]))
	n_mats = len(diffmats_list)
	if n_mats == 1:
		return diffmats_list[0]
	else:
		D = diffmats_list[n_mats-1]
		j = n_mats - 2
		while j >= 0 :
			D = D.dot(diffmats_list[j])
			j -= 1
		return D

def binary2diffmat(dims,binary):
	if np.sum(binary)==1: # i.e. single partial
		D = build_diffmat(dims, direction= np.sum( np.arange(dims.shape[0])*binary) ) # because I don't know C's which functions
	else: # i.e. multiple partials
		D = mixedpartial(dims,binary)
	return D

def create_D(dims,deltas= None): # None so that we can debug easier with matrix of 0s and 1s
	binaries = fd_binaries(dims.shape[0])
	diffmats = []
	if deltas is None:
		for i in range(binaries.shape[0]):
			diffmats.append(binary2diffmat(dims,binary=binaries[i]))
	else:	
		for i in range(binaries.shape[0]-1):
			delts = np.prod(deltas**(1-binaries[i])) # If asking partials for x_j (j=1,2,...), we do not multiply delta_j. 
			diffmats.append(binary2diffmat(dims,binary=binaries[i])*delts)
	D = vstack(diffmats)
	return D

# Nearest point in p-dimensions

def nearest1_unit(target,choices):
	# choices, i.e. the mesh, should always input as column by mesh_coords
	dists = np.sum((choices-target)**2,axis=1)
	# We want to return vector-index of mesh
	return np.argmin(dists)

def nearest1(data, mesh):
	# We find the nearest mesh point to the data.
	# mesh and data input as vectors or lists
	n = len(data)
	fit_inds = [0]*n
	for i in range(n):
		fit_inds[i] = nearest1_unit(target = data[i,],choices = mesh)
	return fit_inds

def nearest_interp_matrix(data,mesh):
	# Create the n x (m_1 x m_2 x ... m_p) interpolation matrix
	n = len(data)
	cols = len(mesh)
	# Find the inds
	col_ind = nearest1(data,mesh); row_ind = range(n); val = [1.0]*n
	# Instantiate the sparse matrix
	O = csc_matrix((val,(row_ind,col_ind)),(n,cols))
	return O

# Need to create mesh covariate values
def mesh_coords(data,mesh_dims):
	# Given data, output knots that are evenly space in the data domain. 
	ntheta = np.prod(mesh_dims)
	eps = 0.01
	unilat_mesh = []; deltas = []
	for i in range(data.shape[1]):
		meshi = np.linspace(min(data[:,i])-eps,max(data[:,i])+eps,mesh_dims[i])
		unilat_mesh.append(meshi)
		deltas.append(np.diff(meshi)[0])
	Theta = np.meshgrid(*unilat_mesh)
	flats = []
	for j in range(data.shape[1]):
		flats.append(Theta[j].reshape((ntheta,1)))
	mesh = np.concatenate(flats,1)
	return {'mesh':mesh, 'deltas':deltas}

# PSEUDO-INVERSE FUNCTION
# can only unit test within optimization problem

def mypinv(spm, spmt, Oty):
	# spm: sparse matrix; spmt: transpose of sparse matrix
	rows = Oty.shape[0]
	aprime = spmt.dot(spm)
	a = splu(aprime.tocsc())
	b = a.solve(Oty.reshape((rows,)))
	return spm.dot(b)

def lam_max_pinv(spm, spmt, Oty):
	A = mypinv(spm, spmt, Oty)
	tune = np.max(abs(A))
	return tune

