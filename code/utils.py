import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, csc_matrix, vstack, linalg as sla
import cvxopt as cvxopt
import math
import itertools
from operator import add, sub

# Tensor to Vector Function

def t2v(dims):
	# Takes the dimensions of the tensor outputs index in long vector
	def t2v_ind(index):
		val = index[0]
		for i in range(1,dims.shape[0]):
			val = val + index[i]*np.prod(dims[:i])
		return val
	
	return t2v_ind

#test = t2v(np.array([5,5]))
#print test(np.array([0,0]))

# Vector to tensor function
def v2t(dims):
	# Takes the dimensions of the tensor, outputs a function that takes vector position and outputs tensor position
	def v2t_ind(index):
		vals = range(dims.shape[0])
		i = dims.shape[0]
		index = float(index+1)
		while i > 0:
			rel_prod = np.prod(dims[:i-1]); #print rel_prod
			vals[i-1] = int(math.ceil(index/rel_prod))-1
			index = index-(vals[i-1])*rel_prod
			i = i -1
		return np.asarray(vals)
	return v2t_ind

# Getting all first order difference binaries for p-dimension into a list without using python map function, list comprehensions are just for loops so should be able to port into C++
def fd_binaries(p):
	lst = [list(i) for i in itertools.product([0,1],repeat=p)][1:] # the 0th element is [0,0,0] for p=3, [0,0] p=2, etc. 
	return np.array(lst)

def calc_alpha(bins,dims):
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


#D3 = build_diffmat(dim = np.array([3,3]),direction=0) # For a 3x3 grid, first differences in row directions
#print D3, "break"
#print build_diffmat(dim = np.array([3,3]),direction=1)
#D2 = build_diffmat(dim=np.array([2,3]), direction = 1)
#print D2.dot(D3)
#print build_diffmat(dims = np.array([3,3]),direction=0)
#print build_diffmat(dims = np.array([3,3]),direction=1)

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
	#print diffmats_list
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

#print binary2diffmat(dims=np.array([3,3]), binary = np.array([0,1]))
#print "BREAK"
#print binary2diffmat(dims=np.array([3,3]), binary = np.array([1,0]))
#print binary2diffmat(dims=np.array([3,3]), binary = np.array([1,1]))
# We need to come up with a test to verify that p=3 case works

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
	#print diffmats
	D = vstack(diffmats)
	return D

#deltas = np.array([.2,.3])
#D =  create_D(dims=np.array([3,3]),deltas=deltas)
#print D
#print D*.03
#print D.shape
#print D.transpose().dot(D)

# Nearest point in p-dimensions
# Written pythonically first, then using only map functions
def nearest1_unit(target,choices):
	choices = np.asarray(choices) # should be input as array, but if not we coerce it
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
	#O = np.zeros((n,cols))
	
	# Find the inds
	# We cannot do it this way unless one data point is nearest the min and another near the max, i.e. we lose columns. 
	col_ind = nearest1(data,mesh); row_ind = range(n); val = [1.0]*n
	O = csc_matrix((val,(row_ind,col_ind)),(n,cols))
	#for i in range(n):
	#	O[i,col_ind[i]] = 1.0
	return O

# Need to create mesh covariate values
def mesh_coords(data,mesh_dims):
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

#a = np.random.uniform(-1,1,20).reshape((20,1))
#b = np.random.uniform(0,2,20).reshape((20,1))
#c = np.random.uniform(3,4,20).reshape((20,1))
#data = np.concatenate((a,b,c),1)
#print data.shape
#mesh = mesh_coords(data,mesh_dims=np.array([3,3,3]))
#O = nearest_interp_matrix(data,mesh)
#print O

# Basic tests
#print create_D([2,2,2])
#print fd_binaries(p=4)

#mydim = np.asarray([3,3])
#test = t2v(mydim)
#test2 = v2t(mydim)
#print test2(9)
# If the number 1 through 27 come out sequentially, working.
#for i in range(3):
#	for j in range(3):
#		for k in range(3):
#			print test([k,j,i])

#for i in range(1,4):
#	for j in range(1,4):
#		print test([j,i])
#print test2(0)
#print test2(26)
#for i in range(3**2):
#	print i, test2(i)

#for i in range(3):
#	for j in range(3):
#		print test(np.array([j,i]))
