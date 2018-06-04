import numpy as np
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf, log
from matplotlib import pyplot as plt
import xarray as xr
import inspect
import math

from SolitonScattering import SG
from SolitonScattering.SG import boundStateRegion

dx = 0.0025
dt = 0.002

xLim = [-40,0]
x0 = -20
M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

v0 = 0.875
k = np.linspace(0,0.2,201)

state = SG.kink(x,0,v0,x0,-1)
field = SG.SineGordon(state)
for t in [1000]:
	field.time_evolve('euler_robin', t+abs(x0)/v0, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
	field.save('v875Kinematics_t%s_k%i_dx0025_dt002_field.nc'%(str(t), len(k))) # save field to disk

with xr.open_dataset('v875Kinematics_t1000_k201_dx0025_dt002_field.nc', engine='h5netcdf') as state:
	print(state)
	field = SG.SineGordon(state) # load field state from disk
	eigenvalues = field.boundStateEigenvalues(vRange=[-0.925, 0.1], maxFreq=0.999, 
		verbose=2, saveFile='v875Kinematics_t1000_k201_dx0025_dt002_eigenvalues.nc',
		rootFindingKwargs={},
		)

eigenvalues = SG.ScatteringData('v875Kinematics_t1000_k201_dx0025_dt002_eigenvalues.nc')
print(eigenvalues)
eigenvalues.show_2Dkinematics(axis='k')
