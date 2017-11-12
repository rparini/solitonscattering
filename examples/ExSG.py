import numpy as np
from scipy import pi, sqrt
from matplotlib import pyplot as plt

from SolitonScattering import SG

dx = 0.25
dt = 0.2

xLim = [-60,0]

M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

v0, k = .95, .145
x0 = -10

### start with an antikink with initial velocity v0 and position x0
state = SG.kink(x,0,v0,x0,epsilon=-1)
field = SG.SineGordon(timeStepFunc='eulerRobin', state=state)

### plot the initial state of this field
field.show(saveFile='SGRobinEx_initialField.pdf', useTex=True)

### time evolve the field to t=200 and plot
field.time_evolve(tFin=150, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
field.show(saveFile='SGRobinEx_finalField.pdf', useTex=True)

### revert the field to its original state and save an animation of the time evolution
field.reset_state()
field.save_animation('SGRobinEx.mov', skipFrames=10, tFin=100, fps=60, dt=dt, k=k, dirichletValue=2*pi)

### find the bound state eigenvalues
vRange = [-0.96, 0]
eigenvalues = field.boundStateEigenvalues(vRange)
print(eigenvalues)






# from time import time
# t0 = time()
# eigenvalues_romb = field.boundStateEigenvalues(vRange, contourIntMethod='romb')
# t1 = time()
# eigenvalues_quad = field.boundStateEigenvalues(vRange, contourIntMethod='quad')
# t2 = time()

# print('romb: t =', t1-t0)
# print(eigenvalues_romb)
# print('quad: t =', t2-t1)
# print(eigenvalues_quad)


# print('Q', field.charge)
# field.show()

# # field.show_animation(skipFrames=5, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
# # field.reset_state()
# # field.show_state(showLims='xR')


# # print(np.vectorize(SG.solitonFrequency)(eigenvalues))


# field.show_eigenvalues(eigenvalues)
