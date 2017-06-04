"""
4th Order Runge-Kutta for a System of 1st Order Linear ODEs
"""
from __future__ import division
import numpy as np
from scipy import *

from .CUtilities_caller import CRungeKuttaArray


class RungeKuttaSolver:
    """
    solves initial value problems of the form u'(t) = A(t).u(t) + B(t) in the range a<t<b where u is a vector, A(t) a function which returns a matrix and B(t) a function which returns a vector, given initial conditions u(a) = u0.
    """
    def __init__(self,u0,A,B = None):
        if B == None or B == 0:
            def B(t):
                return np.zeros(len(u0))

        self.A = A
        self.B = B
        self.u0 = np.array(u0, dtype = 'complex')

    def set_MeshPoints(self,startPoint,endPoint,NoNodes):
        """defines mesh points through start and end points with a known number of nodes"""
        a,b,N = (startPoint,endPoint,NoNodes)
        self.N = N
        self.h = (b-a)/N
        self.a = a
        self.b = b

    def set_StepSize(self,stepSize,startPoint,endPoint = inf):
        """defines mesh points through startPoint and stepSize and an optional endPoint"""
        self.a = startPoint
        self.b = endPoint
        self.h = stepSize
        self.N = int((endPoint-startPoint)/stepSize) + 1

    def solve(self):
        #define the generating function which will step t forward
        def timestep(self):
            a, b, h, u0 = self.a, self.b, self.h, np.copy(self.u0)
            A, B = self.A, self.B

            u,t = u0,a
            while True:

                k1 = h*(np.dot(A(t),u) + B(t))
                k2 = h*(np.dot(A(t+h/2),u+k1/2) + B(t+h/2))
                k3 = h*(np.dot(A(t+h/2),u+k2/2) + B(t+h/2))
                k4 = h*(np.dot(A(t+h),u+k3) + B(t+h))

                u += (k1 + 2*k2 + 2*k3 + k4)/6

                t += h
                yield u, t

        step = timestep(self)

        if self.b != None:
            #if there's an endpoint give the solution on the specified domain of t
            U, T = [self.u0],[self.a]
            for i in range(self.N-1):
                ui, ti = next(step)
                U.append(np.copy(ui))
                T.append(ti)

            U = np.array(U)
            return U,T

        else:
            #if there's no endpoint given return the generator
            return step

def RungeKuttaArray(h, u0, A, B = None, printAll = False, returnT=False):
    """
    solves initial value problems of the form u'(t) = A(t).u(t) + B(t) in the range a<t<b where u is a vector.

    Or A, B and u0 are all given as arrays with A[0] being a matrix corresponding to A(t = a) and A[-1] being A(t = b) ect.
    h = t[1]-t[0] is the distance between each given data point of A

    h is the size of the step in t taken by Runge Kutta
    """
    try:
        # if A is a 3D array
        A0 = A.shape[0]
        AType = A.dtype
    except:
        # if A is a list of matricies
        A0 = len(A)
        AType = A[0].dtype

    if B is None:
        B = np.zeros((A0, u0.size), dtype = AType)

    # Runge Kutta uses a midpoint so the first step of size h requires A[0] = A(a), A[1]=A(h/2), A[2]=A(h)
    # The number of steps to be taken based on the size of the supplied A is therefore
    M = (A0-1)//2


    u = np.zeros((M+1, u0.size), dtype = u0.dtype)
    u[0] = u0
    t = np.zeros(M+1)

    for i in range(M):
        k1 = h * (A[2*i].dot(u[i]) + B[2*i])
        k2 = h * (A[2*i+1].dot(u[i] + k1/2) + B[2*i+1])
        k3 = h * (A[2*i+1].dot(u[i] + k2/2) + B[2*i+1])
        k4 = h * (A[2*i+2].dot(u[i] + k3) + B[2*i+2])

        # print np.dot(A[2*i], u[i]), B[2*i], k1
        # print np.dot(A[2*i+1], u[i]+k1/2), B[2*i+1], k2

        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        t[i+1] = t[i] + h

    if returnT:
        return t, u
    return u


if __name__ == "__main__":
    import time
    from scipy import interpolate, integrate
    # (a,b,h) = (0,500,1)
    (a,b,h) = (0,5,1)
    M = 2*int((b-a)/h)+1
    print('M', M)


    u0 = np.array([1], dtype=complex)
    # u0 = 0.2

    # problem = RungeKutta(u0,A,B)
    # problem.set_StepSize(h,a,b)
    # u,t = problem.solve()
    # print zip(u,t)

    # AFunc = lambda t: sin(5*t)*np.array([[-4,3],[-2.4,1.6]], dtype=complex)
    # BFunc = lambda t: cos(2*t)*np.array([1,2], dtype=complex)

    AFunc = lambda t: sin(t)*np.array([[1]], dtype=complex)
    BFunc = lambda t: t**2*np.array([2], dtype=complex) + 1.3

    tList = np.linspace(a,b,M)

    A = np.empty((M,1,1), dtype = 'complex')
    B = np.empty((M,1), dtype = 'complex')
    # A = np.empty(M, dtype = 'complex')
    # B = np.empty(M, dtype = 'complex')
    for i, t in enumerate(tList):
        A[i] = AFunc(t)
        B[i] = BFunc(t)

    print('Number of steps', (A.shape[0]-1)//2)

    np.set_printoptions(precision=10)
    print('A', A)
    print('B', B)


    # print '-- Scipy --'
    # t0 = time.time()
    # solver = integrate.complex_ode(lambda t, u: AFunc(t).dot(u) + BFunc(t))
    # # solver = integrate.complex_ode(lambda t, u: AFunc(t)*u + BFunc(t))

    # solver.set_initial_value(u0, a)
    # # solver.set_integrator('zvode')
    # # print 't', solver.t, 'u', u0
    # # for i in xrange(M-1):
    # #     print 't', solver.t + h, 'u', solver.integrate(solver.t+h)
    # for i in range(M-1)[::2]:
    #     print 't', solver.t + 2*h, 'u', solver.integrate(solver.t+2*h)
    #     # solver.integrate(solver.t+2*h)
    # print 't', solver.t, 'u', solver.y


    print('-- Python --')
    solver = RungeKuttaSolver(u0, AFunc, BFunc)
    solver.set_StepSize(h,a,b)
    u, T = solver.solve()
    for i, t in enumerate(T):
        print('t', t, 'u', u[i])

    print('-- Python Array --')
    t0 = time.time()
    tList, u = RungeKuttaArray(h, u0, A, B, returnT=True)
    tf = time.time()
    tRK = tf - t0
    for i, t in enumerate(tList):
        print('t', t, 'u', u[i])

    print('-- C Array --')
    t0 = time.time()
    u = CRungeKuttaArray(h, u0, A, B)
    tf = time.time()
    print(u)


