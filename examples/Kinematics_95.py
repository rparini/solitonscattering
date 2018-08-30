import xarray as xr
import numpy as np
from numpy import pi

from SolitonScattering import SG

dx = 0.0025
dt = 0.002

xLim = [-40,0] # x range

v0 = 0.95	# Antikink's initial velocity
x0 = -20 	# Antikink's initial position
k = np.linspace(0.06,0.075,7501) 	# Values for the Robin boundary parameter
print('dk', k[1]-k[0])

### Setup spatial grid
M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

### Run the time evolution and save the result ###
state = SG.kink(x,0,v0,x0,-1)
field = SG.SineGordon(state)
tList = [100] # t=100 -> ~104 hours compute time!  Hope to get this down in the future.
for t in tList:
	field.time_evolve('euler_robin', t+abs(x0)/v0, dt=dt, k=k, 
		dirichletValue=2*pi, asymptoticBoundary={'L':2*pi})
	field.save('Kinematics_v95_t%s_k%i_dx0025_dt002_field.nc'%(str(t), len(k))) # save field to disk

### Compute the bound state eigenvalues
fieldFileName = 'Kinematics_v95_t100_k%i_dx0025_dt002_field.nc'%len(k)
eigenFileName = 'Kinematics_v95_t100_k%i_dx0025_dt002_eigenvalues.nc'%len(k)
with xr.open_dataset(fieldFileName, engine='h5netcdf') as state:
	print(state)
	field = SG.SineGordon(state)

	eigenvalues = field.boundStateEigenvalues(
		vRange=[-0.955, 0.1], 
		maxFreq=0.999,
		verbose=2, 
		saveFile=eigenFileName,
		)

print(eigenvalues)

### Plot the kinematics of the solitons produced in the antikink/boundary collision
from matplotlib import pyplot as plt
eigenvalues.plot_2Dkinematics(axis='k')
plt.title('$v_0=0.95$')
plt.xlim(0.06, 0.075)
plt.savefig('Kinematics_v95_k%i.pdf'%len(k), bbox_inches='tight')
plt.close()

