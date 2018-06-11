from matplotlib import pyplot as plt
import xarray as xr
import numpy as np
from numpy import pi

from SolitonScattering import SG

dx = 0.025
dt = 0.02

xLim = [-40,0] # x range

v0 = 0.875	# Antikink's initial velocity
x0 = -20 	# Antikink's initial position
k = np.linspace(0,0.2,2001) 	# Values for the Robin boundary parameter

### Setup x grid
M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

# ### Run the time evolution for each value of k and save the resulting field
# state = SG.kink(x,0,v0,x0,-1)
# field = SG.SineGordon(state)
# for t in [1000]:
# 	field.time_evolve('euler_robin', t+abs(x0)/v0, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
# 	field.save('v875Kinematics_t%s_k%i_dx025_dt02_field.nc'%(str(t), len(k))) # save field to disk

### Find the bound state eigenvalues associated with the solitons produced in the antikink/boundary collision
### Ignore any breathers with frequency > 0.999
with xr.open_dataset('v875Kinematics_t1000_k2001_dx025_dt02_field.nc', engine='h5netcdf') as state:
	print(state)
	field = SG.SineGordon(state) # load field state from disk
	eigenvalues = field.boundStateEigenvalues(vRange=[-0.925, 0.1], maxFreq=0.999, 
		verbose=2, saveFile='v875Kinematics_t1000_k2001_dx025_dt02_eigenvalues.nc',
		rootFindingKwargs={},
		)

### Print the bound state eigenvalues
eigenvalues = SG.ScatteringData('v875Kinematics_t1000_k2001_dx025_dt02_eigenvalues.nc')
print(eigenvalues)

### Plot the kinematics of the solitons produced in the antikink/boundary collision 
eigenvalues.plot_2Dkinematics(axis='k')
plt.title('$v_0=0.875$')
plt.savefig('Kinematics_v875_k2001.pdf', bbox_inches='tight')
plt.close()
