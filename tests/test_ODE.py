import pytest
import numpy as np
from scipy import sin

AFunc = lambda t: sin(t)*np.array([[1]], dtype=complex)
BFunc = lambda t: t**2*np.array([2], dtype=complex) + 1.3

expected = [
	1,
	1.92279705192254876,
	4.1573104772917742,
	9.4526896259717216,
	20.0767566883386656,
	36.3153195145809955,
	52.7044841428913778,
	60.6854221242737289,
	58.4281882049469648,
	52.6489908564037885,
	50.6181017125020636,
]

args = [(AFunc, BFunc, np.array([1], dtype=complex), 0, 5, 0.5, expected)]

@pytest.mark.parametrize("A,B,u0,t0,tf,h,expected", args)
def test_zvode(A, B, u0, t0, tf, h, expected):
	### Not actually a test, just a note on how to use scipy's integrator
	### for this kind of problem
	from scipy import interpolate, integrate
	M = int((tf-t0)/h)+1
	tList = np.linspace(t0,tf,M)

	solver = integrate.ode(lambda t, u: AFunc(t).dot(u) + BFunc(t))
	solver.set_integrator('zvode')
	solver.set_initial_value(u0, t0)
	# for i in range(M-1):
	# 	print('t', solver.t + h, 'u', solver.integrate(solver.t+h))


@pytest.mark.parametrize("A,B,u0,t0,tf,h,expected", args)
def test_RungeKuttaSolver(A, B, u0, t0, tf, h, expected):
	from SolitonScattering.ODE import RungeKuttaSolver

	solver = RungeKuttaSolver(u0, AFunc, BFunc)
	solver.set_StepSize(h,t0,tf)
	u, tList = solver.solve()
	assert u == pytest.approx(expected)


@pytest.mark.parametrize("A,B,u0,t0,tf,h,expected", args)
def test_RungeKuttaArray(A, B, u0, t0, tf, h, expected):
	from SolitonScattering.ODE import RungeKuttaArray

	M = 2*int((tf-t0)/h)+1
	tList = np.linspace(t0,tf,M)
	A, B = A(tList).reshape(M,1,1), B(tList).reshape(M,1)

	tList, u = RungeKuttaArray(h, u0, A, B, returnT=True)
	assert u == pytest.approx(expected)


@pytest.mark.parametrize("A,B,u0,t0,tf,h,expected", args)
def test_CRungeKuttaArray(A, B, u0, t0, tf, h, expected):
	from SolitonScattering.ODE import CRungeKuttaArray

	M = 2*int((tf-t0)/h)+1
	tList = np.linspace(t0,tf,M)
	A, B = A(tList).reshape(M,1,1), B(tList).reshape(M,1)
	
	u = CRungeKuttaArray(h, u0, A, B)
	assert u == pytest.approx(expected[-1])