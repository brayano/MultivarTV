import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy as scipy
import solvers as solvers

# Univariate Plot Functions

# Bivariate Plot Functions
def plot2_truesurface(x1,x2,y, m,eps=0.01,ftrue=None,mres=50,ntune=50,model=None):
	xx, yy = np.meshgrid(np.linspace(min(x1)-eps,max(x1)+eps,mres),np.linspace(min(x2)-eps,max(x2)+eps,mres))
	xx_a = xx.reshape((mres**2,1))
	yy_a = yy.reshape((mres**2,1))
	args1 = [xx_a,yy_a]
	args2 = [x1,x2]
	zz = ftrue(args1)
	zz = zz.reshape((mres,mres))
	ytrue = ftrue(args2)
	
	if model is None:
		data = np.concatenate((x1.reshape((y.shape[0],1)),x2.reshape((y.shape[0],1))),1)
		model = solvers.mbs(data = data,y=y,m=m,ftrue=ytrue,ntune=ntune)
	fit = model['minmse.fits']
	tune = model['minmse.lam']
	modelmse = model['minmse']
	#fit2 = t1['all_fits'][80]

	xx2, yy2 = np.meshgrid(np.linspace(min(x1),max(x1),mres),np.linspace(min(x2),max(x2),mres))
	newdata = np.concatenate( (xx2.reshape((mres**2,1)),yy2.reshape((mres**2,1))),1)
	zzhat = solvers.mbs_predict(fit,newdata)
	zzhat_grid = zzhat.reshape((mres,mres))
	
	print "Pars: n=%s, lambda = %s, MSE = %s" % (len(y), tune,modelmse)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
	ax.plot_surface(xx2,yy2,zzhat_grid,rstride=1,cstride=1)
	plt.show()	

def plot2_fitdata(x1,x2,y, m,eps=0.01, mres=50,ntune=50,model=None):
	# Data Plotting Function, i.e. no true function is known
	if model is None:
		data = np.concatenate((x1.reshape((y.shape[0],1)),x2.reshape((y.shape[0],1))),1)
		model = solvers.mbs(data=data,y=y,m=m,ftrue=None,ntune=ntune)
	fit = model['minmse.fits']
	tune = model['minmse.lam']
	modelmse = model['minmse']
	#fit2 = t1['all_fits'][80]

	xx2, yy2 = np.meshgrid(np.linspace(min(x1),max(x1),mres),np.linspace(min(x2),max(x2),mres))
	newdata = np.concatenate( (xx2.reshape((mres**2,1)),yy2.reshape((mres**2,1))),1)
	zzhat = solvers.mbs_predict(fit,newdata)
	zzhat_grid = zzhat.reshape((mres,mres))
	
	print "Pars: n=%s, lambda = %s, MSE = %s" % (len(y), tune,modelmse)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(xx2,yy2,zzhat_grid,rstride=1,cstride=1)
	plt.show()	

