import numpy as np
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf, log
from matplotlib import pyplot as plt
import xarray as xr
import inspect
import math

from SolitonScattering import SG

dx = 0.25
dt = 0.2

xLim = [-20,0]

M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

v0 = np.array([0.4, 0.6, 0.8])
# v0 = 0.6
k = np.array([0, 0.2, 0.4, 0.6])
# k = 0.2
# v0, k = 0.6, 0.9
x0 = -10

state = SG.kink(x,0,v0,x0,-1)
field = SG.SineGordon(timeStepFunc='eulerRobin', state=state)

### XXX: What if tFin could be a function of the paramters?
### so tFin = lambda v0, x0: x0/v0 + 200
field.time_evolve(tFin=150, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)

vRange = [-0.95, 0]
eigenvalues = field.boundStateEigenvalues(vRange, selection={'v':1, 'k':1}, verbose=1)
print(eigenvalues)

# field.show(selection={'v':0, 'k':1})


### XXX: Save/Load states and eigenvalues

### XXX: Show eigenvalues