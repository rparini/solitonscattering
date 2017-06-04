import numpy as np
from scipy import pi, sqrt

from SolitonScattering import SG

dx = 0.025
dt = 0.02

xLim = [-40,0]

M  = int((xLim[1] - xLim[0])/dx) + 1
x = np.linspace(xLim[0],xLim[1],M)

v0 = 0.95
k = 0.145
x0 = -20

state = SG.kink(x,0,v0,x0,epsilon=-1)

"""
Looping over field.eigenfunction_wronskian(m, ODEIntMethod='CRungeKuttaArray') seems to be the quickest by far
"""
import time
def times(N):
	mu = np.linspace(0.2,1.5,N)
	t0 = time.clock()

	for m in mu:
		field.eigenfunction_wronskian(m, ODEIntMethod='RungeKuttaArray')
	t1 = time.clock()

	for m in mu:
		field.eigenfunction_wronskian(m, ODEIntMethod='CRungeKuttaArray')
	t2 = time.clock()

	field.eigenfunction_wronskian(mu, ODEIntMethod='RungeKuttaArray')
	t3 = time.clock()

	field.eigenfunction_wronskian(mu, ODEIntMethod='CRungeKuttaArray')
	t4 = time.clock()

	print('N', N, 'loop', t1-t0, 'cloop', t2-t1, 'array', t3-t2, 'carray', t4-t3)
	return t1-t0, t2-t1, t3-t2, t4-t3


import matplotlib.pyplot as plt
N = np.linspace(1,200,50)
loopt, cloopt, arrayt, carrayt = np.vectorize(times)(N)
plt.xlabel('Number of $\mu$ values')
plt.ylabel('Time')
plt.plot(N, loopt, color='b', label='loop')
plt.plot(N, cloopt, color='k', label='cloop')
plt.plot(N, arrayt, color='r', label='vectorized')
plt.plot(N, carrayt, color='g', label='cvectorized')
plt.legend()
plt.savefig('RungeKuttaTest.pdf')
plt.show()

