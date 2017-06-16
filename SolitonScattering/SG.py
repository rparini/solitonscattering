from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf, log
from warnings import warn

from .PDE import PDE

#### Exact solutions to the sine-Gordon Eq. ####
gamma = lambda v: 1 / sqrt(1 - v ** 2)

def kink(x, t, v, x0, epsilon=1):
	# epsilon = \pm 1
	g = gamma(v)
	u  = 4*arctan(exp(epsilon*g*(x-x0-v*t)))
	ut = -2*epsilon*g*v / cosh(epsilon*g*(x-x0-v*t))
	return {'t':t, 'x':x, 'u':u, 'ut':ut}


# def init_magnetic(x, v0, k, x0):
# 	if k != 0:
# 		eta2 = - log((2/k) * (1 - sqrt(1 - k**2 / 4)))
# 		u = 4*arctan(exp(gamma(v0) * (x - x0))) + 4*arctan(exp(x - eta2))
# 	else:
# 		eta2 = np.inf
# 		u = 4*arctan(exp(gamma(v0) * (x - x0)))
# 	ut = - 2 * gamma(v0) * v0 / cosh(gamma(v0) * (x - x0))
# 	return {'t':0, 'x':x, 'u':u, 'ut':ut}


#### Time Stepping Methods ####
def euler_robin(t, x, u, ut, dt, k, dirichletValue=2*pi, dynamicRange=True):
	dx = x[1] - x[0]

	# save the value of the left and right boundaries for later use
	uRightOld = u[-1]
	uLeftOld  = u[0]

	# u_tt = u_xx - sin(u)
	# Get u_tt by using a second order central difference formula to calcuate u_xx
	utt = (np.roll(u,-1) - 2*u + np.roll(u,1))/dx**2 - sin(u)

	# Use utt in a simple (Euler) integration routine:
	ut += dt * utt
	u  += dt * ut

	# Impose Robin boundary condition at the right hand end
	if k == inf:
		# impose Dirichlet boundary condition
		u[-1] = 0
	else:
		u[-1] = u[-2]/(1 + 2*k*dx)

	# Impose Dirichlet boundary condition at left:
	u[0] = dirichletValue

	# Rolling messes ut up at the boundaries so fix here:
	ut[-1] = (u[-1] - uRightOld) / dt
	ut[0]  = (u[0]  - uLeftOld ) / dt

	if dynamicRange:
		checkRange = 10
		# check if there is anything within checkRange spatial points of the left boundary
		if np.any(abs(u[:checkRange]-dirichletValue) > 1e-4):
			# add another checkRange points on to the end
			newPoints = np.linspace(x[0] - checkRange*dx, x[0] - dx, checkRange)

			x = np.insert(x, 0, newPoints)
			u = np.insert(u, 0, dirichletValue*np.ones_like(newPoints))
			ut = np.insert(ut, 0, np.zeros_like(newPoints))

	t += dt
	return {'t':t, 'x':x, 'u':u, 'ut':ut}

def euler_magnetic(t, x, u, ut, dt, k, dirichletValue=2*pi):
	dx = x[1] - x[0]

	# save the value of the left and right boundaries for later use
	uRightOld = u[-1]
	uLeftOld  = u[0]

	# u_tt = u_xx - sin(u)
	# Get u_tt by using a second order central difference formula to calcuate u_xx
	utt = (np.roll(u,-1) - 2 * u + np.roll(u,1))/dx**2 - sin(u)

	# Use utt in a simple (Euler) integration routine:
	ut += dt * utt
	u  += dt * ut

	# Impose magnetic boundary condition at the right hand end
	u[-1] = k*dx + u[-2]

	# Impose Dirichlet boundary condition at left:
	u[0] = dirichletValue

	# Rolling messes ut up at the boundaries so fix here:
	ut[-1]  = (u[-1] - uRightOld)/dt
	ut[0]   = (u[0] - uLeftOld)/dt

	t += dt
	return {'t':t, 'x':x, 'u':u, 'ut':ut}

def euler_integrable(t, x, u, ut, dt, k, dirichletValue=2*pi):
	dx = x[1] - x[0]

	# save the value of the left and right boundaries for later use
	uRightOld = u[-1]
	uLeftOld  = u[0]

	# u_tt = u_xx - sin(u)
	# Get u_tt by using a second order central difference formula to calcuate u_xx
	utt = (np.roll(u,-1) - 2 * u + np.roll(u,1))/dx**2 - sin(u)

	# Use utt in a simple (Euler) integration routine:
	ut += dt * utt
	u  += dt * ut

	# Impose one parameter integrable boundary condition at the right hand end
	# ux + 4 k sin(u/2) = 0
	# u[-1] + 4hk sin(u[-1]/2) = u[-2]
	# solve boundary condition with newton method
	u0 = u[-1]
	error = abs(u0 + 4*dx*k * sin(u0/2) - u[-2])
	tol = 10 ** -20

	i = 0
	while error > tol:
		# print u0, error
		N = 2*dx*k * (u0*cos(u0/2) - 2*sin(u0/2)) + u[-2]
		D = 1 + 2*dx*k * cos(u0/2)
		u0 = N/D

		error = abs(u0 + 4*dx*k * sin(u0/2) - u[-2])
		i += 1

		if i > 500 and error < 10 ** -10:
			u[-1] = u0

	# Impose Dirichlet boundary condition at left:
	u[0] = dirichletValue

	# Rolling messes ut up at the boundaries so fix here:
	ut[-1]  = (u[-1]  - uRightOld ) / dt
	ut[0]   = (u[0]  - uLeftOld ) / dt


	t += dt
	return {'t':t, 'x':x, 'u':u, 'ut':ut}



class SineGordon(PDE):
	def __init__(self, timeStepFunc = 'eulerRobin',  **state):
		"""x, u, ut should be 1D numpy arrays which define the initial conditions
		timeStep should either be an explicit time step function which takes (t, x, u, ut, dt, *args) 
			and returns the state of the field at time t+dt or a string which is the key in named_timeStepFunc
		"""
		self.requiredStateKeys = ['t', 'x', 'u', 'ut'] # everything needed to evolve sine-Gordon one step in time
		self.named_timeStepFuncs = {'eulerRobin': euler_robin,
								   'eulerMagnetic': euler_magnetic,
								   'eulerIntegrable': euler_integrable}
		self.named_solutions = {'kink' : kink}

		super(SineGordon, self).__init__(timeStepFunc, **state)

	@property
	def ux(self):
		# XXX: use a higher order approx to ux
		# XXX: should cache ux and reuse if t is the same

		# get the x derivative of u with a 1st order forward difference formula
		x, u = [self.state[key] for key in ('x','u')]
		ux = np.diff(u)/np.diff(x)

		# add the derivative of the point ux[-1] with a backwards difference formula
		ux = np.append(ux, (u[-1] - u[-2])/(x[1]-x[0]))
		return ux

	def xLax(self, mu):
		# Return the V in the linear eigenvalue problem Y'(x) = V(x,mu).Y(x)
		# as a (1,2,2) matrix where the x is along the first axis
		u, ut = [self.state[key] for key in ('u','ut')]
		ux = self.ux

		# Note that lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		def Vfunc(mu):
			w = ut + ux

			# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
			# v11 = - 0.25j*w
			# v12 = (mu + exp(-1j*u)/(16*mu)) * 1j
			# v21 = - (mu + exp(1j*u)/(16*mu)) * 1j
			# v22 = - v11

			# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
			# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
			v11 = - 0.25j*w
			v12 = mu - exp(-1j*u)/(16*mu)
			v21 = exp(1j*u)/(16*mu) - mu
			v22 = - v11

			V = np.array([[v11, v12], [v21, v22]])

			# Need the axis corresponding to the x coordinate to be the 0th axis
			V = np.rollaxis(V, 2, 0)
			V = np.ascontiguousarray(V)
			return V

		if hasattr(mu, '__iter__'):
			V = np.array([Vfunc(m) for m in mu])
		else:
			V = Vfunc(mu)

		return V

	def left_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> -inf

		# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		# if hasattr(mu, '__iter__'):
		# 	return np.outer(exp(x * (mu + 1/(16*mu))), np.array([1, -1j]))
		# else:
		# 	return exp(x * (mu + 1/(16*mu))) * np.array([1, -1j])

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		if hasattr(mu, '__iter__'):
			return np.outer(exp(-1j*(mu-1/(16*mu))*x), np.array([1, -1j]))
		else:
			return exp(-1j*(mu-1/(16*mu))*x) * np.array([1, -1j])


	def right_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> +inf
		
		# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		# if hasattr(mu, '__iter__'):
		# 	return np.outer(exp(-x * (mu + 1/(16*mu))), np.array([1, 1j]))
		# else:
		# 	return exp(-x * (mu + 1/(16*mu))) * np.array([1, 1j])

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		if hasattr(mu, '__iter__'):
			return np.outer(exp(1j*(mu-1/(16*mu))*x), np.array([1, 1j]))
		else:
			return exp(1j*(mu-1/(16*mu))*x) * np.array([1, 1j])

	def boundStateRegion(self, vRange):
		from cxroots import PolarRect
		# get the region in which to look for bound state eigenvalues
		radiusBuffer = 0.05
		mu = lambda v: 0.25j*sqrt((1-v)/(1+v))
		radiusRange = np.sort(np.abs(mu(np.array(vRange))))
		radiusRange = [radiusRange[0]-radiusBuffer,
					   radiusRange[1]+radiusBuffer]

		center = 0
		phiRange = [0,pi]
		return PolarRect(center, radiusRange, phiRange)

	def boundStateEigenvalues(self, vRange, ODEIntMethod='CRungeKuttaArray'):
		# find the bound state eigenvalues of the 'x' part of the Lax pair
		from cxroots import findRoots
		C = self.boundStateRegion(vRange)

		W = lambda z: self.eigenfunction_wronskian(z,ODEIntMethod)

		# roots occour either on the imaginary axis or in pairs with opposite real parts
		rootSymmetry = lambda z: [-z.conjugate()]

		roots, multiplicities = C.findRoots(W, guessRootSymmetry=rootSymmetry,
			absTol=1e-4, relTol=1e-4)
		return np.array(roots)

	def show_eigenvalues(self, radiusRange, ODEIntMethod='CRungeKuttaArray'):
	@property
	def indexLims(self):
		# values of the field index at which the field and its derivatives are suitably small
		x, u, ut = self.state['x'], self.state['u'], self.state['ut']
		uerr = np.abs(u-2*pi*np.round(u/(2*pi)))
		uterr = np.abs(ut)
		uxerr = np.abs(self.ux)

		# get the regions to the right and left of anything 'interesting'
		uerrOk = np.where(uerr<1e-1, np.ones_like(x), np.zeros_like(x))
		uerrOkIndicies = np.where(uerr<1e-1)[0]

		if uerrOk[0] == 1:
			lRegionIndicies = (int(uerrOkIndicies[0]), int(np.where(np.diff(uerrOk)<0)[0][0]))
		else:
			lRegionIndicies = (int(np.where(np.diff(uerrOk)<0)[0][0]+1), int(np.where(np.diff(uerrOk)<0)[0][0]))

		if uerrOk[-1] == 1:
			rRegionIndicies = (int(np.where(np.diff(uerrOk)>0)[0][-1]+1), uerrOkIndicies[-1])
		else:
			rRegionIndicies = (int(np.where(np.diff(uerrOk)>0)[0][-1]+1), int(np.where(np.diff(uerrOk)<0)[0][-1]+1))

		# within these regions find the minimum of the error function
		# XXX: make a more considered choice for the error function, perhaps involving x
		errfunc = uerr+uterr+uxerr

		lBndry = np.argmin(errfunc[lRegionIndicies[0]:lRegionIndicies[1]])
		rBndry = rRegionIndicies[0]+np.argmin(errfunc[rRegionIndicies[0]:rRegionIndicies[1]])

		# import matplotlib.pyplot as plt
		# ax = plt.gca()
		# plt.plot(x,u)
		# plt.plot(x,uerrOk)
		# ax.axvline(x[lRegionIndicies[0]])
		# ax.axvline(x[lRegionIndicies[1]])
		# ax.axvline(x[rRegionIndicies[0]])
		# ax.axvline(x[rRegionIndicies[1]])
		# plt.plot(x,errfunc)
		# ax.axvline(x[lBndry], color='k')
		# ax.axvline(x[rBndry], color='k')
		# plt.show()

		return lBndry, rBndry

	@property
	def charge(self):
		# The topological charge of the field
		x, u = self.state['x'], self.state['u']
		Q = np.round(u/(2*pi))
		Qerr = np.abs(u-2*pi*Q)

		QerrIndicies = np.where(Qerr<1e-2)[0]
		lBndry = QerrIndicies[0]
		rBndry = QerrIndicies[-1]

		import matplotlib.pyplot as plt
		eigenvalues = field.boundStateEigenvalues(radiusRange, ODEIntMethod)
		return int(Q[rBndry]-Q[lBndry])

	def plot_state(self, *args, **kwargs):
		PDE.plot_state(self, *args, **kwargs)

		# make the y axis in multiples of 2pi
		from matplotlib import pyplot as plt
		ylim = plt.gca().get_ylim()

		# XXX: should generalize to arbitary limits
		plt.yticks([-pi,0,pi,2*pi,3*pi,4*pi],['$-\pi$','$0$','$\pi$','$2\pi$','$3\pi$','$4\pi$'])
		plt.ylim(ylim)


		plt.scatter(eigenvalues.real, eigenvalues.imag)

		plotRad = ceil(radiusRange[1])
		plt.xlim(-plotRad, plotRad)
		plt.ylim(0, plotRad)
		plt.xlabel('Re[$\lambda$]')
		plt.ylabel('Im[$\lambda$]')
