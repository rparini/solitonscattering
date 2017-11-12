from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf, log
from warnings import warn
import xarray as xr
import math
import inspect

import cxroots
from .PDE import PDE, stateFunc, timeStepFunc

#### Some useful equations
solitonVelocity = lambda l: (1-16*np.abs(l)**2)/(1+16*np.abs(l)**2)
solitonFrequency = lambda l: np.real(l)/np.abs(l)

#### Exact solutions to the sine-Gordon Eq. ####
gamma = lambda v: 1 / sqrt(1 - v ** 2)

@stateFunc
def kink(x, t, v, x0, epsilon=1):
	# epsilon = \pm 1
	g = gamma(v)
	u  = 4*arctan(exp(epsilon*g*(x-x0-v*t)))
	ut = -2*epsilon*g*v / cosh(epsilon*g*(x-x0-v*t))
	return {'u':u, 'ut':ut}


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
@timeStepFunc
def euler_robin(t, x, u, ut, dt, k, dirichletValue=2*pi, dynamicRange=True):
	dx = float(x[1] - x[0])

	# save the value of the left and right boundaries for later use
	uRightOld = u[{'x':-1}].copy(deep=True)
	uLeftOld  = u[{'x':0}].copy(deep=True)

	# u_tt = u_xx - sin(u)
	# Get u_tt by using a second order central difference formula to calcuate u_xx
	utt = (u.shift(x=-1) - 2*u + u.shift(x=1))/dx**2 - sin(u)

	# Use utt in a simple (Euler) integration routine:
	ut += dt * utt
	u  += dt * ut

	# Impose Robin boundary condition at the right hand end
	u[{'x':-1}] = u[{'x':-2}]/(1 + 2*k*dx)

	# Impose Dirichlet boundary condition at left:
	u[{'x':0}] = dirichletValue

	# Rolling messes ut up at the boundaries so fix here:
	ut[{'x':-1}] = (u[{'x':-1}] - uRightOld) / dt
	ut[{'x':0}]  = (u[{'x':0}]  - uLeftOld ) / dt

	if dynamicRange:
		checkRange = 10
		# check if there is anything within checkRange spatial points of the left boundary
		if np.any(abs(u[{'x':slice(0,checkRange)}]-dirichletValue) > 1e-4):
			# add another checkRange points on to the end
			newPoints = np.linspace(float(x[0]-checkRange*dx), float(x[0]-dx), checkRange)

			# create new data points for the new region
			newSize = dict([(key, u.sizes[key]) for key in u.sizes]) # copy dictionary
			newSize['x'] = len(newPoints)

			newCoords = dict([(key, u.coords[key]) for key in u.coords]) # copy dictionary
			newCoords['x'] = newPoints

			newData = xr.Dataset(data_vars = {'u':(u.dims, dirichletValue*np.ones([newSize[key] for key in u.dims])),
								  			  'ut':(ut.dims, np.zeros([newSize[key] for key in ut.dims]))}, 
								 coords = newCoords)

			x = np.insert(x, 0, newPoints)
			u = xr.concat([newData['u'], u], dim='x')
			ut = xr.concat([newData['ut'], ut], dim='x')

	# increment time forward a step
	t += dt
	
	# return anything which might have changed
	return {'t':t, 'x':x, 'u':u, 'ut':ut}

# def euler_magnetic(t, x, u, ut, dt, k, dirichletValue=2*pi):
# 	dx = x[1] - x[0]

# 	# save the value of the left and right boundaries for later use
# 	uRightOld = u[-1]
# 	uLeftOld  = u[0]

# 	# u_tt = u_xx - sin(u)
# 	# Get u_tt by using a second order central difference formula to calcuate u_xx
# 	utt = (np.roll(u,-1) - 2 * u + np.roll(u,1))/dx**2 - sin(u)

# 	# Use utt in a simple (Euler) integration routine:
# 	ut += dt * utt
# 	u  += dt * ut

# 	# Impose magnetic boundary condition at the right hand end
# 	u[-1] = k*dx + u[-2]

# 	# Impose Dirichlet boundary condition at left:
# 	u[0] = dirichletValue

# 	# Rolling messes ut up at the boundaries so fix here:
# 	ut[-1]  = (u[-1] - uRightOld)/dt
# 	ut[0]   = (u[0] - uLeftOld)/dt

# 	t += dt
# 	return {'t':t, 'x':x, 'u':u, 'ut':ut}

# def euler_integrable(t, x, u, ut, dt, k, dirichletValue=2*pi):
# 	dx = x[1] - x[0]

# 	# save the value of the left and right boundaries for later use
# 	uRightOld = u[-1]
# 	uLeftOld  = u[0]

# 	# u_tt = u_xx - sin(u)
# 	# Get u_tt by using a second order central difference formula to calcuate u_xx
# 	utt = (np.roll(u,-1) - 2 * u + np.roll(u,1))/dx**2 - sin(u)

# 	# Use utt in a simple (Euler) integration routine:
# 	ut += dt * utt
# 	u  += dt * ut

# 	# Impose one parameter integrable boundary condition at the right hand end
# 	# ux + 4 k sin(u/2) = 0
# 	# u[-1] + 4hk sin(u[-1]/2) = u[-2]
# 	# solve boundary condition with newton method
# 	u0 = u[-1]
# 	error = abs(u0 + 4*dx*k * sin(u0/2) - u[-2])
# 	tol = 10 ** -20

# 	i = 0
# 	while error > tol:
# 		# print u0, error
# 		N = 2*dx*k * (u0*cos(u0/2) - 2*sin(u0/2)) + u[-2]
# 		D = 1 + 2*dx*k * cos(u0/2)
# 		u0 = N/D

# 		error = abs(u0 + 4*dx*k * sin(u0/2) - u[-2])
# 		i += 1

# 		if i > 500 and error < 10 ** -10:
# 			u[-1] = u0

# 	# Impose Dirichlet boundary condition at left:
# 	u[0] = dirichletValue

# 	# Rolling messes ut up at the boundaries so fix here:
# 	ut[-1]  = (u[-1]  - uRightOld ) / dt
# 	ut[0]   = (u[0]  - uLeftOld ) / dt


# 	t += dt
# 	return {'t':t, 'x':x, 'u':u, 'ut':ut}



class SineGordon(PDE):
	def __init__(self, timeStepFunc = 'eulerRobin',  **state):
		"""x, u, ut should be 1D numpy arrays which define the initial conditions
		timeStep should either be an explicit time step function which takes (t, x, u, ut, dt, *args) 
			and returns the state of the field at time t+dt or a string which is the key in named_timeStepFunc
		"""
		self.requiredStateKeys = ['t', 'x', 'u', 'ut'] # everything needed to evolve sine-Gordon one step in time
		self.named_timeStepFuncs = {'eulerRobin': euler_robin}
		# self.named_timeStepFuncs = {'eulerRobin': euler_robin,
		# 						   'eulerMagnetic': euler_magnetic,
		# 						   'eulerIntegrable': euler_integrable}
		self.named_solutions = {'kink' : kink}

		super(SineGordon, self).__init__(timeStepFunc, **state)

	def setticks(self):
		# mark yticks in multiples of pi
		from matplotlib import pyplot as plt
		ax = plt.gca()
		yticks = np.arange(math.floor(ax.get_ylim()[0]/pi)*pi, math.ceil(ax.get_ylim()[1]/pi)*pi, pi)

		def nameticks(tick):
			multiple = int(round(tick/pi))
			if multiple == -1:
				return '$-\pi$'
			elif multiple == 0:
				return '0'
			elif multiple == 1:
				return '$\pi$'
			return '$'+str(multiple)+r'\pi$'

		plt.yticks(yticks, list(map(nameticks, yticks)))


	@property
	def ux(self):
		# get the x derivative of u with a 2nd order central difference formula
		# with 1st order differences at the boundaries
		x = self.state['x'].data
		dx = x[1]-x[0]

		u = self.state['u']
		xAxis = list(u.indexes).index('x')
		ux = np.gradient(u, dx, edge_order=1, axis=xAxis)

		# put ux in an xarray
		return xr.DataArray(ux, u.coords, u.dims)

	def xLax(self, mu, selection={}):
		from xarray.ufuncs import exp
		# Return the V in the linear eigenvalue problem Y'(x) = V(x,mu).Y(x)
		# as a (1,2,2) matrix where the x is along the first axis
		u, ut, ux = self.state['u'][selection], self.state['ut'][selection], self.ux[selection]

		# make mu a DataArray
		mu = xr.DataArray(np.array(mu), coords={'mu':np.array(mu)}, dims=('mu'))

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

		# extend v11 and v22 in the mu dimension
		v11 = v11 * mu/mu
		v22 = v22 * mu/mu

		V = xr.concat([xr.concat([v11, v12], dim='Vj'), xr.concat([v21, v22], dim='Vj')], dim='Vi')

		return V

	def left_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> -inf
		from xarray.ufuncs import exp
		mu = xr.DataArray(mu, [('mu', mu)])

		# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		# E = exp((mu+1/(16*mu))*x)

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		E = exp(-1j*(mu-1/(16*mu))*x)

		return xr.concat([E, -1j*E], dim='Phii')

	def right_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> +inf
		from xarray.ufuncs import exp
		mu = xr.DataArray(mu, [('mu', mu)])

		# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		# E = exp(-(mu+1/(16*mu))*x)

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		E = exp(1j*(mu-1/(16*mu))*x)

		return xr.concat([E, 1j*E], dim='Phii')

	def boundStateRegion(self, vRange):
		from cxroots import AnnulusSector
		# get the region in which to look for bound state eigenvalues
		radiusBuffer = 0.05

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		mu = lambda v: 0.25j*sqrt((1-v)/(1+v))
		phiRange = [0,pi]

		radiusRange = np.sort(np.abs(mu(np.array(vRange))))
		radiusRange = [radiusRange[0]-radiusBuffer,
					   radiusRange[1]+radiusBuffer]
		center = 0
		return AnnulusSector(center, radiusRange, phiRange)

	def boundStateEigenvalues(self, vRange, ODEIntMethod='CRungeKuttaArray', rootFindingKwargs={}, selection={}, verbose=1):
		# find the bound state eigenvalues of the 'x' part of the Lax pair
		u = self.state['u'][selection]

		# set custom defaults
		rootFindingKwargs.setdefault('guessRootSymmetry', lambda z: [-z.conjugate()])	# roots occour either on the imaginary axis or in pairs with opposite real parts
		rootFindingKwargs.setdefault('absTol', 1e-2)
		rootFindingKwargs.setdefault('relTol', 1e-2)
		rootFindingKwargs.setdefault('M', 3)
		rootFindingKwargs.setdefault('intMethod', 'romb')

		if verbose == 2:
			rootFindingKwargs['verbose'] = True
		else:
			rootFindingKwargs['verbose'] = False

		# add any remaining defaults to rootFindingKwargs
		rootArgs, rootVarargs, rootKeywords, rootDefaults = inspect.getargspec(cxroots.RootFinder.findRootsGen)
		rootDefaultDicts = dict(zip(rootArgs[-len(rootDefaults):], rootDefaults))
		for key, val in rootDefaultDicts.items():
			rootFindingKwargs.setdefault(key, val)

		# create array to store roots in
		rootsCoords = dict((key, u.coords[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		rootsDims = list(rootsCoords.keys())
		rootsShape = list(len(u[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)

		rootsAttrs = {'vRange':vRange,
					  'ODEIntMethod':ODEIntMethod}
		rootsAttrs.update(rootFindingKwargs)
		del rootsAttrs['verbose']	# no need to record verbose

		roots = xr.DataArray(np.zeros(rootsShape, dtype=object), coords=rootsCoords, dims=rootsDims, attrs=rootsAttrs)

		# create progressbar
		if verbose:
			from tqdm import trange
			progressBar = trange(sum(rootsShape))

		# compute the roots
		for index, dummy in np.ndenumerate(np.empty(rootsShape)):
			indexDict = dict([(key, index[i]) for i, key in enumerate(rootsCoords)])

			if verbose:
				coordDict = dict([(key, float(rootsCoords[key][index[i]])) for i, key in enumerate(rootsCoords)])
				progressBar.set_description(desc='Computing eigenvalues for '+str(coordDict))

			wronskian_selection = selection.copy()
			wronskian_selection.update(indexDict)

			W = lambda z: np.array(self.eigenfunction_wronskian(z, ODEIntMethod, selection=wronskian_selection), dtype=np.complex128)

			C = self.boundStateRegion(vRange)
			rootResult = C.roots(W, **rootFindingKwargs)
			r, m = rootResult.roots, rootResult.multiplicities
			roots[indexDict] = np.array(r)

			if verbose:
				progressBar.update()

		return roots

	@property
	def indexLims(self):
		# get the values of the x index to the left and right of any soliton content
		# at which the field and its derivatives are suitably small
		x, u, ut, ux = self.state['x'], self.state['u'], self.state['ut'], self.ux

		# create array to put the index lims into
		indexLimsNames = [name for name in u[{'x':0}].coords if name!='x']
		indexLimsNames.append('side')
		indexLimsCoords = [u[{'x':0}].coords[name].data for name in u[{'x':0}].coords if name!='x']
		indexLimsCoords.append(['L','R'])
		indexLimsShape = tuple(map(len, indexLimsCoords))
		indexLims = np.zeros(indexLimsShape, dtype=int)
		indexLims = xr.DataArray(indexLims, indexLimsCoords, indexLimsNames)

		for index, dummy in np.ndenumerate(u[{'x':0}]):
			indexDict = dict([(key, index[i]) for i, key in enumerate(u[{'x':0}].indexes) if key!='x'])

			uerr = np.abs(u[indexDict]-2*pi*np.round(u[indexDict]/(2*pi)))
			uterr = np.abs(ut[indexDict])
			uxerr = np.abs(ux[indexDict])

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
				if np.where(np.diff(uerrOk)>0)[0]:
					rRegionIndicies = (int(np.where(np.diff(uerrOk)>0)[0][-1]+1), int(np.where(np.diff(uerrOk)<0)[0][-1]+1))
				else:
					# just take the right most point
					rRegionIndicies = (int(np.where(np.diff(uerrOk)<0)[0][-1]), int(np.where(np.diff(uerrOk)<0)[0][-1]+1))

			# within these regions find the minimum of the error function
			# XXX: make a more considered choice for the error function, perhaps involving x
			errfunc = uerr+uterr+uxerr

			lBndry = np.argmin(errfunc[lRegionIndicies[0]:lRegionIndicies[1]])
			rBndry = rRegionIndicies[0]+np.argmin(errfunc[rRegionIndicies[0]:rRegionIndicies[1]])

			# import matplotlib.pyplot as plt
			# ax = plt.gca()
			# plt.plot(x,u[{'k':1}])
			# plt.plot(x,uerrOk)
			# ax.axvline(x[lRegionIndicies[0]])
			# ax.axvline(x[lRegionIndicies[1]])
			# ax.axvline(x[rRegionIndicies[0]])
			# ax.axvline(x[rRegionIndicies[1]])
			# plt.plot(x,errfunc)
			# ax.axvline(x[lBndry], color='k')
			# ax.axvline(x[rBndry], color='k')
			# plt.show()

			indexLims[indexDict] = lBndry, rBndry

		return indexLims

	@property
	def charge(self):
		# The topological charge of the field
		x, u = self.state['x'], self.state['u']
		Q = np.round(u/(2*pi))
		Qerr = np.abs(u-2*pi*Q)

		QerrIndicies = np.where(Qerr<1e-2)[0]
		lBndry = QerrIndicies[0]
		rBndry = QerrIndicies[-1]

		return int(Q[rBndry]-Q[lBndry])

	def plot_state(self, *args, **kwargs):
		PDE.plot_state(self, *args, **kwargs)

		# make the y axis in multiples of 2pi
		from matplotlib import pyplot as plt
		ylim = plt.gca().get_ylim()

		# XXX: should generalize to arbitary limits
		plt.yticks([-pi,0,pi,2*pi,3*pi,4*pi],['$-\pi$','$0$','$\pi$','$2\pi$','$3\pi$','$4\pi$'])
		plt.ylim(ylim)


	def typeEigenvalues(self, eigenvalues):
		# get the total field topological charge
		Q = self.charge

		# first filter out all the breathers
		breatherIndicies = np.where(abs(solitonFrequency(eigenvalues)) > 1e-5)[0]
		typedEigenvalues = [(l, 'Breather') for l in eigenvalues[breatherIndicies]]
		eigenvalues = np.delete(eigenvalues, breatherIndicies)

		if len(eigenvalues) == 1:
			# if there's only one kink/antikink left look at the field's total topological charge
			if Q == 1:
				typedEigenvalues.append((eigenvalues[0], 'Kink'))
			elif Q == -1:
				typedEigenvalues.append((eigenvalues[0], 'Antikink'))

			eigenvalues = np.delete(eigenvalues, 0)


		# elif len(untypedEigenvalues) == 2 and Q == 0 and not self.quickSoltype:
		# 	# if there's two then it's a kink + antikink and we need to work out which is which
		# 	# the only distingusing factor between the eigenvalues is the speed
		# 	# so we'll estimate the speed the old fashioned way by running the time evolution a little more

		# 	if not hasattr(self, 'ut') or not hasattr(self, 'u'):
		# 		# if we haven't saved the field then just run the time evolution until we have it
		# 		self.get_rebounded(self.t)

		# 	def get_typedPositions(u, x):
		# 		# get the positions of kinks and antikinks

		# 		# find where kinks and antikinks have their midpoint
		# 		if u[np.abs(u).argmax()] > 0:
		# 			midpoint = eq.roundNearest(u[0]) + pi
		# 		elif u[np.abs(u).argmax()] < 0:
		# 			midpoint = eq.roundNearest(u[0]) - pi

		# 		# get the two points just above the kink/antikink midpoint
		# 		try:
		# 			pointsAboveMidpoint = (u[u > midpoint][0], u[u > midpoint][-1])
		# 		except IndexError:
		# 			# sometimes the kink and antikink are too close together
		# 			return None

		# 		# Interpolate to find the place where the field = midpoint
		# 		typedPositions = {}
		# 		for u1 in pointsAboveMidpoint:
		# 			index1 = np.abs(u - u1).argmin()
		# 			index0 = index1 - 1
		# 			x0, x1 = x[index0], x[index1]
		# 			u0 = u[index0]

		# 			ux = (u1 - u0) / (x1 - x0)
		# 			if ux > 0:
		# 				soltype = 'Kink'
		# 			else:
		# 				soltype = 'Antikink'

		# 			solpos = x0 + (x1 - x0) * (midpoint - u0) / (u1 - u0)
		# 			typedPositions[soltype] = solpos

		# 		try:
		# 			typedPositions['Kink']
		# 			typedPositions['Antikink']
		# 		except KeyError:
		# 			return None

		# 		return typedPositions

		# 	# get the initial position of the kink and antikink
		# 	typedPos0 = get_typedPositions(self.u, self.x)
		# 	if typedPos0 == None:
		# 		# can't figure out which eigenvalue is which
		# 		for eigenvalue in untypedEigenvalues:
		# 			typedEigenvalues.append((eigenvalue, 'Unknown'))
		# 		return typedEigenvalues


		# 	# run the time evolution
		# 	k = self.parameters[1]
		# 	timeStep = SG.timeStepGen(self.x[0], self.x[-1], self.Options['pointDensity'], self.t,
		# 							  inf, self.Options['dt'], self.u, self.ut, self.Options['boundaryType'], k, xi = self.xi)

		# 	t, t0 = self.t, self.t
		# 	u, x = self.u, self.x
		# 	while t < t0 + 10 or get_typedPositions(u, x) == None:
		# 		(u,ut,t,x) = timeStep.next()
		# 	t1 = t

		# 	# from matplotlib import pyplot as plt
		# 	# print t1
		# 	# plt.plot(x,u)
		# 	# plt.show()

		# 	# get the position of the kink and antikink at t1
		# 	typedPos1 = get_typedPositions(u, x)

		# 	# work out the speed
		# 	kinkSpeed = (typedPos1['Kink'] - typedPos0['Kink']) / (t1 - t0)
		# 	antikinkSpeed = (typedPos1['Antikink'] - typedPos0['Antikink']) / (t1 - t0)

		# 	# now match the manual speed with the eigenvalue speed
		# 	eigenvalueSpeed    = np.vectorize(solitonVelocity)(untypedEigenvalues)
		# 	kinkEigenvalue     = untypedEigenvalues.pop(np.abs(eigenvalueSpeed - kinkSpeed).argmin())
		# 	typedEigenvalues.append((kinkEigenvalue, 'Kink'))

		# 	eigenvalueSpeed    = np.vectorize(solitonVelocity)(untypedEigenvalues)
		# 	antikinkEigenvalue = untypedEigenvalues.pop(np.abs(eigenvalueSpeed - antikinkSpeed).argmin())
		# 	typedEigenvalues.append((antikinkEigenvalue, 'Antikink'))

		# elif len(untypedEigenvalues) == abs(q):
		# 	# we've probably got a series of kinks/antikinks
		# 	for i in xrange(abs(q)):
		# 		eigenvalue = untypedEigenvalues.pop(0)
		# 		if Q > 0:
		# 			typedEigenvalues.append((eigenvalue, 'Kink'))
		# 		elif Q < 0:
		# 			typedEigenvalues.append((eigenvalue, 'Antikink'))

		# if there are any eigenvalues left then they are unknown
		for eigenvalue in eigenvalues:
			typedEigenvalues.append((eigenvalue, 'Unknown'))

		return typedEigenvalues

	def show_eigenvalues(self, vRange, ODEIntMethod='CRungeKuttaArray', saveFile=None):
		import matplotlib.pyplot as plt
		eigenvalues = self.boundStateEigenvalues(vRange, ODEIntMethod)
		eigenvalues = self.typeEigenvalues(eigenvalues)
		C = self.boundStateRegion(vRange)

		colorDict = {'Kink':'b', 
					 'Antikink':'r', 
					 'Breather':'g', 
					 'Unknown':'k'}

		path = C(np.linspace(0,1,1e3))
		plt.plot(path.real, path.imag, linestyle='--', color='k')
		for e, t in eigenvalues:
			plt.scatter(e.real, e.imag, color=colorDict[t])

		ax = plt.gca()
		ax.set_aspect(1)

		plotRad = 0.1+np.ceil(C.rRange[1]*10)/10
		plt.xlim(-plotRad, plotRad)
		plt.ylim(-0.02, plotRad)
		plt.xlabel('Re[$\lambda$]')
		plt.ylabel('Im[$\lambda$]')

		if saveFile is not None:
			plt.savefig(saveFile, bbox_inches='tight')
			plt.close()
		else:
			plt.show()
