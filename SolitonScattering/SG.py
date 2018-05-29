from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf, log, arctan2, sinh, tanh
import warnings
import xarray as xr
import math
import inspect
import os.path

import cxroots
from .PDE import PDE, stateFunc, timeStepFunc, getval
from .SGRobin import metastable_u0s, metastable_energy, closest_metastable

#### Some useful equations
gamma = lambda v: 1/sqrt(1 - v**2)

solitonVelocity = lambda l: (1-16*np.abs(l)**2)/(1+16*np.abs(l)**2)
solitonFrequency = lambda l: np.real(l)/np.abs(l)
solitonEnergy = lambda v: 8*gamma(v)
breatherEnergy = lambda v, w: 16*gamma(v)/gamma(w)

#### Exact solutions to the sine-Gordon Eq. ####
@stateFunc
def kink(x, t, v, x0, epsilon=1):
	# epsilon = \pm 1
	g = gamma(v)
	u  = 4*arctan(exp(epsilon*g*(x-x0-v*t)))
	ut = -2*epsilon*g*v / cosh(epsilon*g*(x-x0-v*t))
	return {'u':u, 'ut':ut}

@stateFunc
def breather(x, t, v, w, x0, xi = -pi/2):
	W = sqrt(1-w**2)
	g = 1/sqrt(1-v**2)
	Sin = sin(w*g*(t-v*(x-x0)) + xi)
	Cosh = cosh(W*g*(x-x0-v*t))

	u = 4*arctan2(W*Sin/w, Cosh)

	Cos = cos(w*g*(t-v*(x-x0)) + xi)
	Tanh = tanh(W*g*(x-x0-v*t))
	N = v*W**2*Sin*Tanh + w*W*Cos
	D = (W*Sin/Cosh)**2 + w**2

	ut = 4*w*g/Cosh * N/D
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
	# XXX: any way to avoid this copying?
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
		newPoints = 10**4
		# check if there is anything within checkRange spatial points of the left boundary
		if np.any(abs(u[{'x':slice(0,checkRange)}]-dirichletValue) > 1e-4):
			# add another newPoints points on to the end
			newPoints = np.linspace(float(x[0]-newPoints*dx), float(x[0]-dx), newPoints)

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
	# how to store types on disk (for sine-Gordon topological charge is used)
	type_encoding = {'Kink':1, 'Antikink':-1, 'Breather':0, 'Unknown':9}

	def __init__(self, state):
		"""x, u, ut should be 1D numpy arrays which define the initial conditions
		timeStep should either be an explicit time step function which takes (t, x, u, ut, dt, *args) 
			and returns the state of the field at time t+dt or a string which is the key in named_timeStepFunc
		"""
		self.requiredStateKeys = ['t', 'x', 'u', 'ut'] # everything needed to evolve sine-Gordon one step in time
		self.named_timeStepFuncs = {'eulerRobin': euler_robin}
		# self.named_timeStepFuncs = {'eulerRobin': euler_robin,
		# 						   'eulerMagnetic': euler_magnetic,
		# 						   'eulerIntegrable': euler_integrable}
		self.named_solutions = {'kink':kink, 'breather':breather}

		super(SineGordon, self).__init__(state)

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
				return '$0$'
			elif multiple == 1:
				return '$\pi$'
			return '$'+str(multiple)+r'\pi$'

		plt.yticks(yticks, list(map(nameticks, yticks)))

	@property
	def ux(self):
		if hasattr(self, '_ux'):
			saved_t, ux = self._ux
			if saved_t == getval(self.state, 't'):
				return ux

		# get the x derivative of u with a 2nd order central difference formula
		# with 1st order differences at the boundaries
		x = self.state['x'].data
		dx = x[1]-x[0]

		u = self.state['u']
		xAxis = list(u.indexes).index('x')
		ux = np.gradient(u, dx, edge_order=1, axis=xAxis)

		# put ux in an xarray
		ux = xr.DataArray(ux, u.coords, u.dims)

		self._ux = getval(self.state, 't'), ux
		return ux

	def xLax(self, mu, selection={}):
		from xarray.ufuncs import exp
		# Return the V in the linear eigenvalue problem Y'(x) = V(x,mu).Y(x)
		# as a (1,2,2) matrix where the x is along the first axis
		u, ut, ux = self.state['u'][selection], self.state['ut'][selection], self.ux[selection]

		if hasattr(mu, '__iter__'):
			# make mu a DataArray
			mu = xr.DataArray(np.array(mu), coords={'mu':np.array(mu)}, dims=('mu'))
			identityMu = xr.DataArray(np.ones_like(mu), coords={'mu':np.array(mu)}, dims=('mu'))
		else:
			identityMu = 1

		w = ut + ux

		# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		# v11 = - 0.25j*w
		# v12 = (mu + exp(-1j*u)/(16*mu)) * 1j
		# v21 = - (mu + exp(1j*u)/(16*mu)) * 1j
		# v22 = - v11

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		v11 = - 0.25j*w * identityMu
		v12 = mu - exp(-1j*u)/(16*mu)
		v21 = exp(1j*u)/(16*mu) - mu
		v22 = - v11

		V = xr.concat([xr.concat([v11, v12], dim='Vj'), xr.concat([v21, v22], dim='Vj')], dim='Vi')
		return V

	def left_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> -inf
		from xarray.ufuncs import exp
		if hasattr(mu, '__iter__'):
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
		if hasattr(mu, '__iter__'):
			mu = xr.DataArray(mu, [('mu', mu)])

		# With lambda=i*mu in Eq.9 of "Breaking integrability at the boundary"
		# E = exp(-(mu+1/(16*mu))*x)

		# With lambda=mu in Eq.9 of "Breaking integrability at the boundary"
		# As in Eq. II.2 in "Spectral theory for the periodic sine-Gordon equation: A concrete viewpoint" with lambda=Sqrt[E]
		E = exp(1j*(mu-1/(16*mu))*x)

		return xr.concat([E, 1j*E], dim='Phii')

	def boundStateEigenvalues(self, vRange, ODEIntMethod='CRungeKuttaArray', rootFindingKwargs={}, selection={}, verbose=1, saveFile=None):
		# if saveFile is given then the eigenvalues will be saved to disk as they are computed


		# find the bound state eigenvalues of the 'x' part of the Lax pair
		u = self.state['u'][selection]

		# Set custom defaults.  These can be overridden using rootFindingKwargs
		rootFindingKwargs.setdefault('guessRootSymmetry', lambda z: [-z.conjugate()])	# roots occour either on the imaginary axis or in pairs with opposite real parts
		rootFindingKwargs.setdefault('absTol', 1e-2)
		rootFindingKwargs.setdefault('relTol', 1e-2)
		rootFindingKwargs.setdefault('M', 3)
		rootFindingKwargs.setdefault('divMin', 4)
		rootFindingKwargs.setdefault('divMax', 15)
		rootFindingKwargs.setdefault('m', 2) 	# 2*m+1 stencil size for numerical differentiation during contour integration
		rootFindingKwargs.setdefault('NintAbsTol', .02)
		rootFindingKwargs.setdefault('integerTol', .1)
		rootFindingKwargs.setdefault('intMethod', 'romb')

		if verbose >= 2:
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
		del rootsAttrs['verbose']			# no need to record verbose
		del rootsAttrs['guessRootSymmetry']	# unable to save functions to disk
		del rootsAttrs['df']				# df (the derivative of the Wronskian) is not available

		# Unable to store bools so convert to 1 or 0
		rootsAttrs['attemptIterBest'] = int(rootsAttrs['attemptIterBest'])

		if verbose >= 2:
			print('rootFindingKwargs:', rootFindingKwargs)

		# create arrays to store eigenvalues and types of eigenvalues
		eigenvalue_array = np.full(rootsShape, np.nan, dtype=object)
		type_array = np.full(rootsShape, np.nan, dtype=object)

		# we will store the topolgical charge rather than the string returned by 'typeEigenvalues'
		type_dtype = np.dtype('i4')   # 32-bit signed integer

		arrayDims = rootsDims + ['eigenvalue_index']
		arrayShape = rootsShape + [1]
		if saveFile and not os.path.isfile(saveFile):
			# initialise spectral DataSet if it doesn't already exist on disk
			xr_eigenvalue_array = np.full(arrayShape, np.nan, dtype=np.complex128)
			xr_type_array = np.full(arrayShape, np.nan, dtype=type_dtype)
			spectralData = xr.Dataset({'eigenvalues':(arrayDims, xr_eigenvalue_array),
								   	   'types':(arrayDims, xr_type_array)},
								   	   coords=rootsCoords, attrs=rootsAttrs)

			if saveFile[-3:] != '.nc':
				saveFile += '.nc'
			# need to use engine='h5netcdf' to save complex files
			spectralData.to_netcdf(saveFile, engine='h5netcdf')	

		# create progressbar
		makeProgressbar = verbose and rootsShape
		if makeProgressbar:
			from tqdm import trange
			progressBar = trange(np.prod(rootsShape))

		skipped = []

		# compute the roots
		maxNumberOfEigenvalues = 0
		maxTypeStringSize = 0
		for index, dummy in np.ndenumerate(np.empty(rootsShape)):
			indexDict = dict([(key, index[i]) for i, key in enumerate(rootsCoords)])

			if makeProgressbar:
				coordDict = dict([(key, float(rootsCoords[key][index[i]])) for i, key in enumerate(rootsCoords)])
				progressBar.set_description(desc='Computing eigenvalues for '+str(coordDict))

			# check to see if roots were already computed
			alreadyComputed = False
			if saveFile:
				with xr.open_dataset(saveFile, engine='h5netcdf') as spectralData:
					eigenvalueData = spectralData['eigenvalues']
					if not xr.ufuncs.isnan(eigenvalueData[indexDict][{'eigenvalue_index':0}]):
						alreadyComputed = True
						maxNumberOfEigenvalues = eigenvalueData.sizes['eigenvalue_index']

			if not alreadyComputed:
				wronskian_selection = selection.copy()
				wronskian_selection.update(indexDict)

				W = lambda z: np.array(self.eigenfunction_wronskian(z, ODEIntMethod, selection=wronskian_selection), dtype=np.complex128)
				C = boundStateRegion(vRange)

				try:
					if verbose >= 3:
						rootFindingKwargs.update({'automaticAnim':True})
						rootResult = C.demo_roots(W, **rootFindingKwargs)
					else:
						rootResult = C.roots(W, **rootFindingKwargs)
					r, m = rootResult.roots, rootResult.multiplicities

					if r and np.any(np.array(m) != 1):
						print(rootResult)
						raise RuntimeError('Multiplicities are not all 1!')
				except Exception as e:
					raise e
					print('Skipping ', indexDict)
					print(e)
					skipped.append(indexDict)
					if makeProgressbar:
						progressBar.update()
					continue


				# store computed eigenvalues and types
				typedEigenvalues = self.typeEigenvalues(r, u[indexDict])
				t = np.array([self.type_encoding[ty] for ty in typedEigenvalues], dtype=type_dtype)
				eigenvalue_array[index] = r
				type_array[index] = t

				if verbose >= 2:
					print(print_eigenvalues(np.array(r), np.array(typedEigenvalues)))

				# update maxNumberOfEigenvalues
				if len(r) > maxNumberOfEigenvalues:
					maxNumberOfEigenvalues = len(r)

				# store computed eigenvalues and types on disk
				if saveFile:
					# XXX: is there a way to write incrementally without loading and overwriting the whole array?
					spectralData = xr.open_dataset(saveFile, engine='h5netcdf')
					spectralData.load()		# load whole file into memory (by default it is only lazily read)
					spectralData.close()

					if len(spectralData['eigenvalues'][indexDict]) < len(r):
						# need to extend the array
						padShape = rootsShape + [len(r) - len(spectralData['eigenvalues'][indexDict])]
						padded_eigenvalue_array = np.full(padShape, np.nan, dtype=np.complex128)
						padded_type_array = np.full(padShape, np.nan, dtype=type_dtype)
						padArray = xr.Dataset({'eigenvalues':(arrayDims, padded_eigenvalue_array),
											   'types':(arrayDims, padded_type_array)},
											   coords=rootsCoords, attrs=rootsAttrs)

						spectralData = xr.concat([spectralData, padArray], dim='eigenvalue_index')

					padded_eigenvalues = np.full(len(spectralData['eigenvalues'][indexDict]), np.nan, dtype=np.complex128)
					padded_types	   = np.full(len(spectralData['eigenvalues'][indexDict]), np.nan, dtype=type_dtype)

					padded_eigenvalues[:len(r)] = r[:]
					padded_types[:len(r)] 	    = t[:]

					spectralData['eigenvalues'][indexDict] = padded_eigenvalues[:]
					spectralData['types'][indexDict] = padded_types[:]

					with warnings.catch_warnings():
						# FutureWarning: complex dtypes are supported by h5py, but not part of the NetCDF API. 
						# You are writing an HDF5 file that is not a valid NetCDF file! In the future, this will 
						# be an error, unless you set invalid_netcdf=True.
						warnings.simplefilter(action='ignore', category=FutureWarning)
						spectralData.to_netcdf(saveFile, engine='h5netcdf')

			if makeProgressbar:
				progressBar.update()

		if makeProgressbar:
			progressBar.close()

		if skipped:
			print('Skipped ', skipped)

		# pad out eigenvalue array so that it can be stored as a single complex-valued array
		arrayShape = rootsShape + [maxNumberOfEigenvalues]
		padded_eigenvalue_array = np.full(arrayShape, np.nan, dtype=np.complex128)
		padded_type_array = np.full(arrayShape, np.nan, dtype=type_dtype)
		for index, dummy in np.ndenumerate(np.empty(rootsShape)):
			if not np.all(np.isnan(eigenvalue_array[index])):
				pad = np.full(maxNumberOfEigenvalues - len(eigenvalue_array[index]), np.nan)
				padded_eigenvalue_array[index] = np.append(eigenvalue_array[index], pad)
				padded_type_array[index] = np.append(type_array[index], pad)

		# put eigenvalue and type arrays into an xarray Dataset
		spectralData = xr.Dataset({'eigenvalues':(arrayDims, padded_eigenvalue_array),
								   'types':(arrayDims, padded_type_array)},
								   coords=rootsCoords, attrs=rootsAttrs)

		if saveFile:
			with xr.open_dataset(saveFile, engine='h5netcdf') as oldSpectralData:
				spectralData = spectralData.combine_first(oldSpectralData)

		return ScatteringData(spectralData)

	def lims_index(self, selection={}):
		if hasattr(self, '_indexLims') and self._indexLims[1] == selection:
			# use cached value of _indexLims for this time and selection
			return self._indexLims[0]

		# get the values of the x index to the left and right of any soliton content
		# at which the field and its derivatives are suitably small
		x, u, ut, ux = self.state['x'], self.state['u'], self.state['ut'], self.ux
		dx = float(x[1]-x[0])

		# Create array to put the index lims into		
		indexLimsNames = [name for name in u[{'x':0}].coords if name!='x']
		indexLimsCoords = [u[{'x':0}].coords[name].data for name in indexLimsNames]

		indexLimsNames.append('side')
		indexLimsCoords.append(np.array(['L','R']))

		indexLimsShape = tuple(map(len, indexLimsCoords))

		indexLims = np.zeros(indexLimsShape, dtype=int)
		indexLims = xr.DataArray(indexLims, coords=indexLimsCoords, dims=indexLimsNames)
		
		# Restrict to selection
		u, ut, ux = u[selection], ut[selection], ux[selection]
		indexLims = indexLims[selection]

		for index, dummy in np.ndenumerate(u[{'x':0}]):
			indexDict = dict([(key, index[i]) for i, key in enumerate(u[{'x':0}].indexes) if key!='x'])

			uerr = np.abs(u[indexDict]-2*pi*np.round(u[indexDict]/(2*pi)))
			uterr = np.abs(ut[indexDict])
			uxerr = np.abs(ux[indexDict])

			# XXX: make a more considered choice for the error function
			errfunc = uerr+uterr+uxerr

			### The allowed region where xR or xL might be placed should contain energy < 1
			energyDensity = .5*ut[indexDict]**2 + .5*ux[indexDict]**2 + 1-cos(u[indexDict])

			# Get allowed region to the left
			energyToLeft = energyDensity.cumsum(dim='x', dtype=float)*dx
			leftAllowedBoundaryIndex = np.where(energyToLeft < 1)[0][-1]

			# Get allowed region to the right
			# XXX: This is Robin boundary specific!  Try to generalise.
			uR = float(u[{'x':-1}][indexDict])
			# if 'k' in self.state.attrs.keys():
			# 	k = self.state.attrs['k']
			# elif 'k' in self.state.coords.keys():
			# 	k = self.state['k'][selection][indexDict]
			k = getval(self.state[selection][indexDict], 'k')
			n = closest_metastable(uR, k) 			# we are closest to the nth metastable boundary
			metaEnergy = metastable_energy(n, k)	# the energy of the nth metastable boundary
			energyToRight = k*uR**2 - metaEnergy + dx*(energyDensity.sum(dim='x', dtype=float) - energyDensity.cumsum(dim='x', dtype=float))
			rightAllowedBoundaryIndex = np.where(energyToRight < 1)[0][0]

			# within these regions find the minimum of the error function, biased
			# slightly towards reducing the distance between xL and xR
			xLIndex = np.argmin(errfunc[:leftAllowedBoundaryIndex] - 1e-5*x[:leftAllowedBoundaryIndex])
			xRIndex = rightAllowedBoundaryIndex+np.argmin(errfunc[rightAllowedBoundaryIndex:] + 1e-5*x[rightAllowedBoundaryIndex:])

			# sometimes (when there is a boundary breather) there is too much energy at the boundary
			# so xR ends up being on the boundary even though the field is nowhere near the full line vacuum
			# therefore enforce a minimum of -20 for xR
			maxError = .1
			if x[xRIndex] > -20 and errfunc[xRIndex] > 10*maxError:
				warnings.warn('xR was at %.3f due to the energy at the boundary but will instead be set to -20.  Consider allowing time evolution to run for longer.'%x[xRIndex], RuntimeWarning)
				xRIndex = np.argmin(abs(x+20))

			if errfunc[xLIndex] > maxError:
				warnings.warn('At xL uerr=%.3f, uterr=%.3f, uxerr=%.3f.  Consider allowing time evolution to run for longer.'%(uerr, uterr, uxerr), RuntimeWarning)
			if errfunc[xRIndex] > maxError:
				warnings.warn('At xR uerr=%.3f, uterr=%.3f, uxerr=%.3f.  Consider allowing time evolution to run for longer.'%(uerr, uterr, uxerr), RuntimeWarning)

			indexLims[indexDict][:] = np.array([xLIndex, xRIndex], dtype=int)

		# cache for later
		self._indexLims = indexLims, selection
		return indexLims

	def typeEigenvalues(self, eigenvalues, selection={}):
		types = np.zeros_like(eigenvalues, dtype=object)
		if len(types) == 0:
			return types
		else:
			types[:] = ''

		u = self.state['u'][selection]

		# first filter out all the breathers
		for i in np.where(abs(np.vectorize(solitonFrequency)(eigenvalues)) > 1e-5)[0]:
			types[i] = 'Breather'

		# get the total field topological charge
		Q = charge(u)

		if len(np.where(types=='')[0]) == 1:
			# if there's only one kink/antikink left look at the field's total topological charge
			if Q == 1:
				types[np.where(types=='')[0]] = 'Kink'
			elif Q == -1:
				types[np.where(types=='')[0]] = 'Antikink'

		elif Q == 0 and len(np.where(types=='')[0]) > 0 and getval(self.state, 'k') is not None:
			# if there's two then it's probably a kink + antikink and we need to work out which is which
			# the only distingusing factor between the eigenvalues is the speed
			# so we'll estimate the speed the old fashioned way by running the time evolution a little more

			roundNearest = lambda u: 2*pi * np.round(u/(2*pi))

			def get_typedPositions(u, x):
				# get the positions of kinks and antikinks

				# find where kinks and antikinks have their midpoint
				if u[np.abs(u).argmax()] > 0:
					midpoint = roundNearest(u[0]) + pi
				elif u[np.abs(u).argmax()] < 0:
					midpoint = roundNearest(u[0]) - pi

				# get the two points just above the kink/antikink midpoint
				try:
					pointsAboveMidpoint = (u[u > midpoint][0], u[u > midpoint][-1])
				except IndexError:
					# sometimes the kink and antikink are too close together
					return None

				# Interpolate to find the place where the field = midpoint
				typedPositions = {}
				for u1 in pointsAboveMidpoint:
					index1 = np.abs(u - u1).argmin()
					index0 = index1 - 1
					x0, x1 = x[index0], x[index1]
					u0 = u[index0]

					ux = (u1 - u0) / (x1 - x0)
					if ux > 0:
						soltype = 'Kink'
					else:
						soltype = 'Antikink'

					solpos = x0 + (x1 - x0) * (midpoint - u0) / (u1 - u0)
					typedPositions[soltype] = float(solpos)

				try:
					typedPositions['Kink']
					typedPositions['Antikink']
				except KeyError:
					return None

				return typedPositions

			# get the initial position of the kink and antikink
			typedPos0 = get_typedPositions(u, self.state['x'])
			if typedPos0 == None:
				for i in np.where(types=='')[0]:
					types[i] = 'Unknown'
				return types

			# run the time evolution a bit more to manually see the speed
			if getval(self.state, 'k') is not None:
				timeStepFunc = 'euler_robin' 	### XXX: Need a better way to decide that we are using the Robin boundary

			t0 = getval(self.state, 't')

			tempField = SineGordon(self.state[selection])

			t = getval(tempField.state, 't')
			while t < t0 + 5 or get_typedPositions(tempField.state['u'], tempField.state['x']) == None:
				dt = self.state.attrs['dt']
				t = getval(tempField.state, 't')
				if timeStepFunc == 'euler_robin':
					k = getval(self.state, 'k')
					tempField.time_evolve(timeStepFunc, t+dt, progressBar=False, dt=dt, k=k, dirichletValue=2*pi, dynamicRange=True)
			t1 = t

			# get the position of the kink and antikink at t1
			typedPos1 = get_typedPositions(tempField.state['u'], tempField.state['x'])

			# work out the speed
			kinkSpeed = (typedPos1['Kink'] - typedPos0['Kink']) / (t1 - t0)
			antikinkSpeed = (typedPos1['Antikink'] - typedPos0['Antikink']) / (t1 - t0)

			# now match the manual speed with the eigenvalue speed
			eigenvalueSpeed = np.vectorize(solitonVelocity)(eigenvalues)
			types[np.abs(eigenvalueSpeed - kinkSpeed).argmin()] = 'Kink'
			types[np.abs(eigenvalueSpeed - antikinkSpeed).argmin()] = 'Antikink'

		# else:
		# 	# Extract charge from spectral data
		# 	# [TK] "Essentially Nonlinear One-dimensional model of classical field theory"
			
		# 	# shift x axis so that 0 is at the center of the [xL, xR] interval
		# 	xRIndex, xLIndex = self.lims_index()
		# 	centerIndex = int((xRIndex + xLIndex)/2)
		# 	centerX = self.state['x'][centerIndex]
		# 	xL = float(self.state['x'][xLIndex])

		# 	self.state['x'] -= centerX

		# 	xR = float(self.state['x'][xRIndex])

		# 	W = self.eigenfunction_wronskian
		# 	# dW = cxroots.CxDeriv(W)
		# 	dz = 1e-8
		# 	dW = lambda z: (complex(W(z+dz))-complex(W(z)))/dz

		# 	for i in np.where(abs(np.vectorize(solitonFrequency)(eigenvalues)) < 1e-5)[0]:
		# 		mu = eigenvalues[i]

		# 		yR_asymp = self.right_asyptotic_eigenfunction(mu, xR)
		# 		yR = self.eigenfunction_right(mu)

		# 		f = yR.values.reshape((2,1))
		# 		g = yR_asymp.values

		# 		ratio = f/g
		# 		m = 2*ratio[0]/dW(mu)

		# 		# print('--- Should be same ---')
		# 		# print(2*ratio[0]/dW(mu))
		# 		# print(2*ratio[1]/dW(mu))
		# 		# print('----------------------')

		# 		b = -1j*m[0]

		# 		# [TK] says sign[b] instead of -sign[b]
		# 		epsilon = -int(b.real/abs(b.real)) # =-sign[b]

		# 		# print('mu', mu, )
		# 		# print('W(mu)', complex(W(mu)))
		# 		# print('dW(mu)', complex(dW(mu)))
		# 		# print('ratio', f/g)
		# 		# print('m', m)
		# 		# print('b', b)
		# 		# print('epsilon', epsilon)

		# 		if epsilon == 1:
		# 			types[i] = 'Kink'
		# 		elif epsilon == -1:
		# 			types[i] = 'Antikink'

		# 	# shift x axis back to original position
		# 	self.state['x'] += centerX

		# if there are any eigenvalues left then they are unknown
		for i in np.where(types=='')[0]:
			types[i] = 'Unknown'

		return types

	def plot_state(self, *args, **kwargs):
		PDE.plot_state(self, *args, **kwargs)

		# make the y axis in multiples of 2pi
		from matplotlib import pyplot as plt
		ylim = plt.gca().get_ylim()

		# XXX: should generalize to arbitary limits
		plt.yticks([-pi,0,pi,2*pi,3*pi,4*pi],['$-\pi$','$0$','$\pi$','$2\pi$','$3\pi$','$4\pi$'])
		plt.ylim(ylim)


def charge(u):
	# The topological charge of the field
	Q = np.round(u/(2*pi))
	Qerr = np.abs(u-2*pi*Q)

	QerrIndicies = np.where(Qerr<1e-2)[0]
	lBndry = QerrIndicies[0]
	rBndry = QerrIndicies[-1]

	return int(Q[rBndry]-Q[lBndry])


def boundStateRegion(vRange):
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

def print_eigenvalues(roots, types):
	if len(roots) == 0:
		return 'No bound state eigenvalues'

	velocity = solitonVelocity(roots)
	frequency = solitonFrequency(roots)
	energy = np.array([breatherEnergy(velocity[i], frequency[i]) if types[i]=='Breather' else solitonEnergy(velocity[i]) for i in range(len(roots))])

	# sort eigenvalues by energy in decending order
	sortargs = np.array(np.argsort(energy)[::-1])
	roots 		= roots[sortargs]
	types 		= types[sortargs]
	velocity 	= velocity[sortargs]
	frequency 	= frequency[sortargs]
	energy 		= energy[sortargs]

	s =  '     Type     |           Eigenvalues           |  Velocity  | Frequency |  Energy  '
	s+='\n------------------------------------------------------------------------------------'

	skipNextBreather = False
	for i, root in enumerate(roots):
		if np.isnan(root):
			continue

		if types[i] == 'Breather':
			if skipNextBreather:
				skipNextBreather = False
				continue

			if abs(root.real + roots[i+1].real) < 1e-8 and abs(root.imag - roots[i+1].imag) < 1e-8:
				skipNextBreather = True

		if types[i] == 'Breather':
			s += '\n{: ^14s}| ±{:.12f} {:+.12f}i|{: ^12f}|{: ^11f}|{: ^10f}'.format(types[i], abs(root.real), root.imag, velocity[i].real, abs(frequency[i]), energy[i])
		elif root.real < 0:
			s += '\n{: ^14s}| {:.12f} {:+.12f}i|{: ^12f}|{: ^11f}|{: ^10f}'.format(types[i], root.real, root.imag, velocity[i].real, abs(frequency[i]), energy[i])
		else:
			s += '\n{: ^14s}|  {:.12f} {:+.12f}i|{: ^12f}|{: ^11f}|{: ^10f}'.format(types[i], root.real, root.imag, velocity[i].real, abs(frequency[i]), energy[i])

	return s


class ScatteringData(object):
	def __init__(self, data):
		if type(data) == str:
			data = xr.open_dataset(data, engine='h5netcdf')
			data.load()
			data.close()

		self.data = data

		self.colorDict = {'Kink':'C0', 'Antikink':'C3', 'Breather':'C2', 'Unknown':'C7'}
		self.type_decoding = {v: k for k, v in SineGordon.type_encoding.items()}

	def decode_types(self, encodedTypes):
		return np.array([self.type_decoding[e] if e in self.type_decoding.keys() else e for e in encodedTypes])

	def __str__(self):
		eigenvalues = self.data['eigenvalues']
		types = self.data['types']

		if len(self.data['eigenvalues'].dims) == 1:
			return print_eigenvalues(eigenvalues.values, self.decode_types(types.data))

		else:
			s = ''
			for index, dummy in np.ndenumerate(np.empty(eigenvalues.shape[:-1])):
				indexDict = dict([(key, index[i]) for i, key in enumerate(self.data.coords)])
				coordDict = dict([(key, float(self.data.coords[key][index[i]])) for i, key in enumerate(self.data.coords)])
				s += '\n' + str(coordDict) + '\n'
				s += print_eigenvalues(eigenvalues[index].data, self.decode_types(types[index].data)) + '\n'
			return s

	def save(self, saveFile):
		if saveFile[-3:] != '.nc':
			saveFile += '.nc'
		self.data.to_netcdf(saveFile, engine='h5netcdf')

	def show(self, *args, **kwargs):
		arrayShape = self.data['eigenvalues'][{'eigenvalue_index':0}].shape
		if len(arrayShape) == 0:
			self.show_eigenvalues(*args, **kwargs)
		elif len(arrayShape) == 1:
			self.show_2Dkinematics(*args, **kwargs)

	def show_eigenvalues(self, saveFile=None):
		import matplotlib.pyplot as plt
		types = self.decode_types(self.data['types'].data)

		C = boundStateRegion(self.data.attrs['vRange'])

		path = C(np.linspace(0,1,1e3))
		plt.plot(path.real, path.imag, linestyle='--', color='k')
		for i, e in enumerate(self.data['eigenvalues'].values):
			plt.scatter(e.real, e.imag, color=self.colorDict[types[i]])

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

	def show_2Dkinematics(self, axis):
		import matplotlib.pyplot as plt
		for t in ['Kink', 'Antikink', 'Breather']:
			data = self.data.where(self.data['types'] == SineGordon.type_encoding[t])

			# plot frequency
			freq = abs(solitonFrequency(data['eigenvalues']))
			freq = freq.where(freq < .999)
			if t == 'Breather':
				plt.plot(data[axis], np.sort(freq)[:,0], 'k', linestyle='--', label='Frequency')

			# plot speed
			speed = abs(solitonVelocity(data['eigenvalues']))
			speed = speed.where(freq < .999)
			if t == 'Breather':
				# plot lowest frequency breather
				speed = speed[:, np.argsort(freq)[:,0]]
			else:
				speed = np.sort(speed)
			plt.plot(data[axis], speed, self.colorDict[t], label=t)

		plt.ylim(0,1)
		plt.xlim(data[axis][0], data[axis][-1])
		plt.ylabel('Speed/Frequency')
		plt.xlabel(axis)
		plt.show()

