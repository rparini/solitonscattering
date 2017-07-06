import numpy as np
from scipy import pi, sqrt

from SolitonScattering import SG

dx = 0.025
dt = 0.02

xLim = [-40,0]

M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

v0 = 0.8
k = 0.145
x0 = -20

state = SG.kink(x,0,v0,x0,epsilon=-1)
field = SG.SineGordon(timeStepFunc='eulerRobin', **state)
# print('Plotting initial state of the field')
# field.show_state()
# print('Time evolving field')
# field.time_evolve(tFin=150, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
# print('Potting new state of the field')
# print(field.indexLims)
# field.show_state(showLims='xR')

field.show_animation(skipFrames=5, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
# field.reset_state()
field.show_state(showLims='xR')
# field.save_animation('test.mov', skipFrames=10, tFin=10, fps=60, dt=dt, k=k, dirichletValue=0)

vRange = [-0.9, 0]
field.show_eigenvalues(vRange, ODEIntMethod='CRungeKuttaArray')
