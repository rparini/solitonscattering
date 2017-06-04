import numpy as np
from scipy import pi, sqrt

from SolitonScattering import SG

dx = 0.025
dt = 0.02

xLim = [-40,0]

M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

v0 = -0.8
k = 0.145
x0 = -20

state = SG.kink(x,0,v0,x0,epsilon=-1)
field = SG.SineGordon(timeStepFunc='eulerRobin', **state)
# print('Plotting initial state of the field')
# field.show_state()
# print('Time evolving field')
# field.time_evolve(tFin=150, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
# # print('Potting new state of the field')
# field.show_state()

# print(field.eigenfunction_wronskian(1j,ODEIntMethod='RungeKuttaArray'))
# print(field.eigenfunction_wronskian(1j,ODEIntMethod='CRungeKuttaArray'))


# field.show_animation(skipFrames=10, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
# field.reset_state()
# field.show_state()
# field.save_animation('magnetic_v0=0_04_k=0_03.mov', skipFrames=10, tFin=600, fps=60, dt=dt, k=k, dirichletValue=0)


# assuming -v0<v<0 implies that the eigenvalue magnitude for kinks/antikinks lies within this range
radiusBuffer = 0.05
radiusRange = [0.25-radiusBuffer, 0.25*sqrt((1+v0)/(1-v0))+radiusBuffer]
radiusRange = [0.25, 1]

mu = 0.25j*sqrt((1-v0)/(1+v0))
print('mu', mu)
print('W', field.eigenfunction_wronskian(mu, ODEIntMethod='CRungeKuttaArray'))
# field.show_eigenfunction(mu)

# # import matplotlib.pyplot as plt
# # W = lambda z: field.eigenfunction_wronskian(z, ODEIntMethod='CRungeKuttaArray')
# # W = np.vectorize(W)
# # mu = -1j*np.linspace(0.01,1,1000)
# # plt.ylim(-10,10)
# # plt.plot(np.imag(mu), np.abs(W(mu)))
# # plt.show()

# eigenvalues = field.boundStateEigenvalues(radiusRange, ODEIntMethod='CRungeKuttaArray')
# print(eigenvalues)

# field.show_wronskianMag()

# field.show_eigenfunction(mu)

# field.show_eigenvalues(radiusRange, ODEIntMethod='CRungeKuttaArray')

# print(self.eigenfunction_wronskian(1e-6, ODEIntMethod))

