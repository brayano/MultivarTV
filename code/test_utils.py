import utils
import numpy as np
import time

# Unit tests for tensor-to-vector and vector-to-tensor mapping functions

mydim = np.array([3,3,3])
#ind = np.array([0,0,0])

def test_t2v_0():
	# If the number 0 comes out, working.
	result = utils.t2v_unit(dims=mydim,ind=np.array([0,0,0]))
	assert result == 0

def test_v2t_000():
	# If the index [0,0,0] comes out, working.
	result = utils.v2t_unit(dims =mydim,ind=0)
	assert all(result == np.array([0,0,0]))

#ind = np.array([2,2,2])
def test_t2v_26():
	# If the number 26 comes out, working.
	result = utils.t2v_unit(dims=mydim,ind=np.array([2,2,2]))
	assert result == 26

def test_v2t_222():
	# If the index [2,2,2] comes out, working.
	result = utils.v2t_unit(dims =mydim,ind=26)
	assert all(result == np.array([2,2,2]))

# Unit test for create_D():

def test_create_D():
	D =  utils.create_D(dims=np.array([3,3]),deltas=None)
	theta = np.tile([1,-1,1],3)
	assert np.sum(D.dot(theta))== 0.0 # Should be 0

# Unit test for nearest1():

def test_nearest1_unit():
	target = np.array(0.1)
	choices = np.array([[0],[0.5],[1.0]])
	result = utils.nearest1_unit(target,choices)
	assert result == 0

def test_nearest1():
	data = np.array([0.1,0.9])
	mesh = np.array([[0],[0.5],[1.0]])
	result = utils.nearest1(data, mesh)
	assert all(result==np.array([0,2]))

def test_nearest_interp_matrix():
	data = np.array([0.1,0.9])
	mesh = np.array([[0],[0.5],[1.0]])
	O = utils.nearest_interp_matrix(data, mesh)
	theta = mesh*np.array([[1],[2],[3]])
	assert all(O.dot(theta)==np.array([[0],[3]]))

# Unit test for mesh_coords

def test_mesh_coords():
	data = np.linspace(0.01,0.99,10)
	result = utils.mesh_coords(data.reshape((10,1)),mesh_dims=np.array([6])) # Since eps=0.01, the mesh should be evenly spaced between 0, 1 at four interior points, two exterior.
	assert np.round(result['deltas'],2) == 0.20 # Round 0.20000000001 to 0.2


