import pytest
import numpy as np
from numpy import pi

from SolitonScattering import SG
from SolitonScattering.PDE import getval

def test_progressBar():
	from SolitonScattering import SG
	dx = 0.025
	dt = 0.02

	xLim = [-60,0]

	M  = int((xLim[1] - xLim[0])/dx) + 1
	x = np.linspace(xLim[0],xLim[1],M)

	v0 = .95
	k = .145
	x0 = -20

	state = SG.kink(x,0,v0,x0,epsilon=-1)
	field = SG.SineGordon(timeStepFunc='eulerRobin', state=state)
	field.time_evolve(tFin=50, progressBar=False, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
	t1 = getval(field.state, 't')

	field.reset_state()
	field.time_evolve(tFin=50, progressBar=True, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
	t2 = getval(field.state, 't')

	assert t1 == t2