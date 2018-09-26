import numpy as np
from scipy import pi, sqrt, cos, sign, sin, log, tan, arctan, exp

from scipy.optimize import brentq
from scipy.integrate import simps

def metastable_u0s(k, includeNegativeu0=True):
	# compute all the values of u0 corresponding to metastable boundaries
	u0List = [0]

	if k < 0:
		raise NotImplemented('Metastable boundaries for negative k not implemented')

	# compute metastable bc for n positive
	n = 0
	err = lambda u0: k*u0 - abs(sin(u0/2))
	while sign(err(2*pi*(n+.5))) != sign(err(2*pi*(n+1))):
		u0 = brentq(err, 2*pi*(n+.5), 2*pi*(n+1))
		u0List.append(u0)

		if includeNegativeu0:
			# add negative metastable u0
			u0List.insert(0, -u0)

		n += 1

	return np.array(u0List)

def metastable_u0(n, k):
	# the value of the field at the boundary corresponding to the nth metastable vacuum
	if k < 0:
		raise NotImplemented('Metastable boundaries for negative k not implemented')

	errFunc = lambda u0: k*u0 - abs(sin(u0/2))
	return brentq(errFunc, 2*pi*n-pi, 2*pi*n)

def antikink_x0(u0):
	# u = 4*arctan(exp(x0-x)) + 2*pi*m
	m = u0//(2*pi)
	return log(tan(0.25*(u0-2*m*pi)))

def kink_x0(u0):
	# u = 4*arctan(exp(x-x0)) + 2*pi*m
	m = u0//(2*pi)
	return -log(tan(0.25*(u0-2*m*pi)))

def metastable_antikink(n, k):
	u0 = metastable_u0(n, k)
	x0 = antikink_x0(u0)
	return lambda x: 4*arctan(exp(x0-x)) + 2*pi*(u0//(2*pi))


def metastable_energy(n, k):
	# energy of the nth metastable state
	u0 = metastable_u0s(k, includeNegativeu0=False)[abs(n)]
	if n < 0:
		u0 = -u0

	eps = (-1)**int(n)
	return 4 - 4*eps*cos(u0/2) + k*u0**2

def closest_metastable(u0, k):
	# find n where the value of the field for the nth metastable boundary is closest to u0
	metau0 = metastable_u0s(k)
	n = np.argmin(abs(metau0-u0)) - (len(metau0)-1)//2
	return n
