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

v0 = np.array([0.6, 0.7, 0.8])
k = np.array([0, 0.9])
x0 = -10

state = SG.kink(x,0,v0,x0,-1)

field = SG.SineGordon(timeStepFunc='eulerRobin', state=state)

field.time_evolve(tFin=200, dt=dt, k=k, dynamicRange=True)

print(field.state)

vRange = [-0.9, 0]
print(field.boundStateEigenvalues(vRange))
