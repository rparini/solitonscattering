from __future__ import division
import numpy as np
import scipy
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf
from copy import deepcopy
from warnings import warn
import inspect
import math
import xarray as xr
import itertools

from . import ODE

try:
	import matplotlib
except ImportError:
	warn('Unable to import matplotlib')

isnparray = lambda x: isinstance(x, np.ndarray)

def getval(state, key):
	# get a value whether it is a coordinate, data or attribute
	# XXX: it would be nice of xarray did this automatically
	if key in state:
		return state[key]
	elif key in state.attrs:
		return state.attrs[key]

def stateFunc(fieldFunc):
	def dataset_wrap(*args, **kwargs):
		# include defaults explicitly in the kwargs
		# argnames, varargs, kwargs, defaults = inspect.getargspec(fieldFunc)
		defaults = inspect.getargspec(fieldFunc)[3]
		tempkwargs = dict((key, defaults[i]) for i, key in enumerate(inspect.getargspec(fieldFunc)[0][-len(defaults):]))
		tempkwargs.update(kwargs)
		kwargs = tempkwargs

		# put args in the kwargs
		# argnames, varargs, kwargs, defaults = inspect.getargspec(fieldFunc)
		kwargs.update(zip(inspect.getargspec(fieldFunc)[0], args))

		# figure out what variables we're vectorizing over (only numpy arrays)
		vectorize = [key for key in kwargs.keys() if isnparray(kwargs[key])]

		# make vectorized arguments into coordinate arrays
		coords = dict((key, xr.DataArray(kwargs[key], coords=[kwargs[key]], dims=[key])) for key in vectorize)
		kwargs.update(coords)

		# get u, ut ect. from the given function
		dataDict = fieldFunc(**kwargs)

		state = xr.Dataset(dataDict)
		state.attrs = dict([(key, kwargs[key]) for key in kwargs if key not in vectorize])
		return state

	return dataset_wrap


def timeStepFunc(stepFunc):
	def timestep_wrap(state, **timestepKwargs):
		# include defaults explicitly
		# argnames, varargs, kwargs, defaults = inspect.getargspec(fieldFunc)
		defaults = inspect.getargspec(stepFunc)[3]
		tempkwargs = dict((key, defaults[i]) for i, key in enumerate(inspect.getargspec(stepFunc)[0][-len(defaults):]))
		tempkwargs.update(timestepKwargs)
		timestepKwargs = tempkwargs

		# figure out what variables we're vectorizing over (only numpy arrays)
		vectorize = [key for key in timestepKwargs.keys() if isnparray(timestepKwargs[key])]

		for key in timestepKwargs.keys():
			if key not in state.keys():
				if key in vectorize:
					# introduce new dimensions to vectorize over
					state = xr.concat([state]*len(timestepKwargs[key]), dim=key)
					state[key] = timestepKwargs[key]

				else:
					# anything we are not vectorizing over should be an attribute
					state.attrs[key] = timestepKwargs[key]

		# pass state to the timeStepFunc
		funcArgs = dict((key, getval(state, key)) for key in inspect.getargspec(stepFunc)[0] if getval(state, key) is not None)

		# take time step
		newVals = stepFunc(**funcArgs)

		# Update attributes
		for key in newVals:
			if key in state.attrs:
				state.attrs[key] = newVals[key]

		oldSize = [state[key].size for key in newVals if key not in state.attrs]
		newSize = [newVals[key].size for key in newVals if key not in state.attrs]
		if np.any(oldSize != newSize):
			# the size of the state has changed so we need to create a new one
			# doesn't seem possible to update state 'in place' with the new coordinates
			state = xr.Dataset(data_vars = dict((key, newVals[key]) for key in state.data_vars.keys()), 
							   attrs = state.attrs)

		return state
	return timestep_wrap


class PDE(object):
	def __init__(self, state, saveInitialState=False):
		self.state = state
		if saveInitialState:
			self._initialState = deepcopy(state)

	@property
	def state(self):
		return self._state
	
	@state.setter
	def state(self, stateVal):
		"""
		stateVal should be an xarray Dataset with data {'u', 'ut'}
		or {'solName', ...} where 'solName' is the name of a known solution given in get_solutions add the other elements of the 
			dictionary should be the solArgs required there
		"""
		if 'solName' in stateVal:
			# if a name of a solution is given then use that function name to create the state
			self._state = self.named_solutions[stateVal.pop('solName')](**stateVal)
		elif type(stateVal) == str:
			# load from disk
			self.state = xr.open_dataset(stateVal, engine='h5netcdf')

		stateValKeys = set(stateVal.data_vars.keys()).union(set(stateVal.attrs)).union(set(stateVal.coords))
		if not self.requiredStateKeys or set(self.requiredStateKeys).issubset(stateValKeys):
			# set the time step funciton as the given dictionary
			self._state = stateVal
		else:
			raise TypeError("The given state should be an xarray Dataset with data_vars or attributes:", self.requiredStateKeys)

	def reset_state(self):
		# reset the state of the field to the state it was in when the instance was first initilized
		self.state = self._initialState

	def save(self, saveFile):
		if saveFile[-3:] != '.nc':
			saveFile += '.nc'
		self.state.to_netcdf(saveFile, engine='h5netcdf')

	def time_evolve(self, timeStepFunc, tFin, progressBar=True, callbackFunc=None, **timeStepArgs):
		# tFin should be a real number or a function which returns a real number
		# if a function then the inputs of that function should be found in the
		# coordinates or attributes of self.state

		# force lims_index to be recalculated
		if hasattr(self, '_indexLims'):
			del self._indexLims

		if type(timeStepFunc) == str:
			# if a name of a time step function is given then return that
			timeStepFunc = self.named_timeStepFuncs[timeStepFunc]

		if callable(tFin):
			### create an array storing the values of tFin for each parameter

			# get arguments to pass to tFin function
			tFinArgNames = inspect.getargspec(tFin)[0]
			tFinArgs = []	# should be a list of DataArrays
			tFinKwargs = {} # for arguments which are not arrays
			for tFinArgName in tFinArgNames:
				if tFinArgName in timeStepArgs.keys():
					tFinArg = timeStepArgs[tFinArgName]
				else:
					val = getval(self.state, tFinArgName)
					if getval(self.state, tFinArgName) is None:
						raise ValueError("Names of tFin arguments should be coordinates or attributes of the field state\n\
							or the names of arguments of time_evolve which have been passed as numpy arrays.")
					tFinArg = getval(self.state, tFinArgName)

				if isnparray(tFinArg) or isinstance(tFinArg, xr.core.dataarray.DataArray):
					tFinArgs.append(xr.DataArray(tFinArg, dims=tFinArgName))
				else:
					tFinKwargs.update({tFinArgName:tFinArg})

			# create array of tFin
			tFin = xr.apply_ufunc(tFin, *tFinArgs, kwargs=tFinKwargs)


			
		### store values of t as an array if necessary
		### XXX: doesn't seem to be an isxarray()
		t = getval(self.state, 't')
		if isinstance(tFin, xr.core.dataarray.DataArray) and not isinstance(t, xr.core.dataarray.DataArray):
			# copy the parameters from self.state
			# (the parameters in timeStepArgs will be added in the timeStepFunc wrapper)
			timeDims = [dim for dim in self.state.dims if dim != 'x']
			timeCoords = [self.state.coords[dim] for dim in timeDims]

			# populate time array with current time
			t = self.state.attrs['t']
			timeArrayShape = list(map(len, timeCoords))
			tArray = xr.DataArray(np.ones(timeArrayShape)*t, coords=timeCoords, dims=timeDims)
			
			# Add time array to the self.state and remove time attribute
			del self.state.attrs['t']
			self.state['t'] = tArray
			t = getval(self.state, 't')

		### Set up the progress bar
		if progressBar and 'dt' in timeStepArgs:
			from tqdm import tqdm
			dt = timeStepArgs['dt']
			N = math.ceil(np.max(tFin-t)/dt)
			pBar = tqdm(total=N, desc='Time Evolution')
		else:
			progressBar = False
			
		while np.any(t < tFin):
			# pass the time step function the current state and any additional given arguments
			if isinstance(t, xr.core.dataarray.DataArray):
				newstate = timeStepFunc(self.state.where(t < tFin), **timeStepArgs)

				# replace nan with old state
				self.state = newstate.fillna(self.state)
			
			else:
				self.state = timeStepFunc(self.state, **timeStepArgs)

			t = getval(self.state, 't')

			if progressBar:
				pBar.update()

			if callbackFunc is not None:
				exit = callbackFunc(self)
				if exit:
					return

		if progressBar:
			pBar.close()

	def setticks(self):
		pass

	def plot(self, selection={}, showLims=False, useTex=False, ylim=None, fontSize=16):
		from matplotlib import pyplot as plt
		if useTex:
			plt.rc('text', usetex=True)
			# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
		plt.rcParams.update({'font.size': fontSize})

		x, u = [self.state[k] for k in ['x','u']]

		self.state['u'][selection].plot()
		plt.xlim(x[0],x[-1])
		ax = plt.gca()

		if showLims is not False:
			iLims = self.lims_index(selection)
			xL, xR = x[iLims[0]], x[iLims[1]]
			ax.set_ylim()

			# get the points x=xL and x=xR in axis coordinates
			xLAxis = ax.transAxes.inverted().transform(ax.transData.transform((xL,0)))[0]
			xRAxis = ax.transAxes.inverted().transform(ax.transData.transform((xR,0)))[0]

			if showLims != 'xR':
				plt.text(xLAxis-.01, 1.01, '$x_L$', transform=ax.transAxes)
				ax.axvline(xL, color='k', linestyle='--', linewidth=1)

			if showLims != 'xL':
				plt.text(xRAxis-.01, 1.01, '$x_R$', transform=ax.transAxes)
				ax.axvline(xR, color='k', linestyle='--', linewidth=1)

		self.setticks()

		if ylim is not None:
			plt.ylim(ylim[0], ylim[1])

		plt.ylabel('$u$', rotation=0)
		plt.xlabel('$x$', labelpad=-1)

	def show(self, saveFile=None, **kwargs):
		from matplotlib import pyplot as plt
		self.plot(**kwargs)
		if saveFile:
			plt.savefig(saveFile, bbox_inches='tight')
			plt.close()
		else:
			plt.show()

	def save_animation(self, saveFile, tFin, ylim=None, fps = 60, writer = None, dpi = 200, codec = None, **timeStepArgs):
		frames = int(tFin / timeStepArgs['dt'])
		saveAnimationDict = {'filename':saveFile, 'fps':fps, 'frames':frames, 'writer':writer, 'dpi':dpi, 'codec':codec}
		self.show_animation(ylim=ylim, saveAnimationDict = saveAnimationDict, **timeStepArgs)

	def show_animation(self, timeStepFunc, skipFrames = 0, ylim=None, saveAnimationDict = {}, 
			useTex=True, fontSize=16,
			saveInitFile=False, **timeStepArgs):
		from matplotlib import pyplot as plt
		from matplotlib import animation

		if useTex:
			plt.rc('text', usetex=True)
			# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
		plt.rcParams.update({'font.size': fontSize})

		if type(timeStepFunc) == str:
			# if a name of a time step function is given then return that
			timeStepFunc = self.named_timeStepFuncs[timeStepFunc]

		fig = plt.gcf()
		fig.tight_layout()
		ax = plt.gca()

		plt.xlim(self.state['x'].values[0], self.state['x'].values[-1])
		line, = ax.plot(self.state['x'], self.state['u'])
		if ylim is not None:
			ax.set_ylim(ylim[0],ylim[-1])
		self.setticks()

		plt.ylabel('$u$', rotation=0)
		plt.xlabel('$x$', labelpad=-1)

		timeLabel = ax.text(0.05, 0.91, '$t=%.1f$' % self.state.attrs['t'], transform=ax.transAxes)

		if saveInitFile:
			plt.savefig(saveInitFile)

		# initialization function sets the background of each frame
		# it should hide anything that will change in the animation
		def init():
			line.set_data([], [])
			timeLabel.set_text('')
			return line, timeLabel

		# animation function which is called every frame
		def update_animation(i):
			for k in range(skipFrames+1):
				# step the time evolution forward
				self.state = timeStepFunc(self.state, **timeStepArgs)

			line.set_data(self.state['x'], self.state['u'])
			timeLabel.set_text('$t = %.1f$' % self.state.attrs['t'])
			return line, timeLabel

		# call the animator
		frames = saveAnimationDict.pop('frames', None)
		if frames is not None:
			frames = 1+frames//(skipFrames+1)
		anim = animation.FuncAnimation(fig, update_animation, init_func=init, interval=20, blit=True, frames=frames)

		if saveAnimationDict:
			# save the animation as a file
			anim.save(**saveAnimationDict)
			plt.close()
		else:
			plt.show()

	def left_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> -inf
		raise NotImplementedError('Implement left_asyptotic_eigenfunction in a child class')

	def right_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> +inf
		raise NotImplementedError('Implement right_asyptotic_eigenfunction in a child class')		

	def eigenfunction_right(self, muList, ODEIntMethod='CRungeKuttaArray', selection={}):
		x, u = self.state['x'], self.state['u'][selection]
		indexLims = self.lims_index(selection)

		# muList should be a list
		if not hasattr(muList, '__iter__'):
			muList = [muList]

		axisShape = list(len(u[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)

		# create yR as an empty DataArray
		yRCoords = dict((key, u.coords[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		yRCoords['mu'] = muList
		yRDims = list(yRCoords.keys())
		yRDims.append('yRi')
		yRShape = list(len(u[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		yRShape.append(len(muList))
		yRShape.append(2)

		yR = xr.DataArray(np.zeros(yRShape, dtype=np.complex128), coords=yRCoords, dims=yRDims)

		### XXX: Calling xLax all at once is much faster than calling individually for each mu but eats memory.
		### Perhaps judge how much memory is going to be used and choose the appropriate method?
		# VFull = self.xLax(muList, selection=selection).transpose('mu','x','Vi','Vj')

		for index, dummy in np.ndenumerate(np.empty(axisShape)):
			indexDict = dict([(key, index[i]) for i, key in enumerate(self.state.coords) if key!='x' and len(u[key].shape)>0])

			fullSelection = {}
			fullSelection.update(selection)
			fullSelection.update(indexDict)

			xLIndex, xRIndex = map(int, indexLims[indexDict])
			x = self.state['x'][xLIndex:xRIndex+1]
			h = float(x[1] - x[0])
			xL, xR = x[0], x[-1]

			# get yBoundL and reorder
			yBoundL = self.left_asyptotic_eigenfunction(muList, xL)
			yBoundL = yBoundL.transpose('mu', 'Phii')

			for muindex, mu in enumerate(muList):
				# reorder V and cast as a numpy array 
				V = np.zeros((len(x),2,2), dtype=np.complex128)
				V[:] = self.xLax(mu, selection=fullSelection)[{'x':slice(xLIndex, xRIndex+1)}].transpose('x','Vi','Vj').data
				# V[:] = VFull[indexDict][{'x':slice(xLIndex, xRIndex+1), 'mu':muindex}].data

				yBoundL_mu = np.zeros(2, dtype=np.complex128)
				yBoundL_mu[:] = yBoundL[{'mu':muindex}].values

				# solve for the Jost eigenfunction which at xL matches the left asymptotic eigenfunction
				# note that stepsize for Runge Kutta 4th order is 2h since it requires midpoint values
				if ODEIntMethod == 'CRungeKuttaArray':
					yR[indexDict][{'mu':muindex}] = ODE.CRungeKuttaArray(2*h, yBoundL_mu, V)
				elif ODEIntMethod == 'RungeKuttaArray':
					yR[indexDict][{'mu':muindex}] = ODE.RungeKuttaArray(2*h, yBoundL_mu, V)[-1]
				else:
					raise ValueError("ODEIntMethod must be either 'CRungeKuttaArray' or 'RungeKuttaArray'")

		return yR

	def full_eigenfunction_right(self, mu, selection={}):
		if hasattr(mu, '__iter__'):
			raise NotImplementedError('mu cannot yet be a list')

		indexLims = self.lims_index(selection)
		xLIndex, xRIndex = map(int, indexLims)
		x = self.state['x'][xLIndex:xRIndex+1]
		h = float(x[1] - x[0])
		xL, xR = x[0], x[-1]

		VFull = self.xLax(mu, selection=selection)
		V = np.zeros((len(x),2,2), dtype=np.complex128)
		V[:] = VFull[selection][{'x':slice(xLIndex, xRIndex+1)}].transpose('x','Vi','Vj').values

		yBoundL = self.left_asyptotic_eigenfunction(mu, xL)

		x, y = ODE.RungeKuttaArray(2*h, yBoundL, V, returnT=True)
		x = np.array(x) + float(xL) 	# ODE.RungeKuttaArray assumes that x starts at 0
		return xr.DataArray(y, coords={'x':x}, dims=['x','yRi'])

	def eigenfunction_wronskian(self, mu, ODEIntMethod='CRungeKuttaArray', selection={}):
		# solve for the eigenfunction across x as an intial value problem
		# at x[0] the eigenfunction is yBoundL = self.left_asyptotic_eigenfunction(mu)
		# solve for the value of the eigenfunction at x[-1]

		# mu is assumed to be a list
		if not hasattr(mu, '__iter__'):
			mu = [mu]

		u = self.state['u'][selection]

		# create wronskian, W, as an empty DataArray
		WCoords = dict((key, u.coords[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		WCoords['mu'] = mu
		WDims = list(WCoords.keys())
		WShape = list(len(u[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		WShape.append(len(mu))

		W = np.empty(WShape, dtype=np.complex128)
		W = xr.DataArray(W, WCoords, WDims)

		# iterate over all axis except mu
		axisShape = list(len(u[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		for index, dummy in np.ndenumerate(np.empty(axisShape)):
			indexDict = dict([(key, index[i]) for i, key in enumerate(self.state.coords) if key!='x' and len(u[key].shape)>0])
			
			full_selection = {}
			full_selection.update(selection)
			full_selection.update(indexDict)

			xLIndex, xRIndex = map(int, self.lims_index(full_selection))

			# shift x axis so that 0 is at the center of the [xL, xR] interval
			centerIndex = int((xRIndex + xLIndex)/2)
			centerX = self.state['x'][centerIndex]
			self.state['x'] -= centerX

			# solve for yL as an initial value problem from xL to xR 
			yR = self.eigenfunction_right(mu, ODEIntMethod, full_selection)

			x = self.state['x'][xLIndex: xRIndex+1]
			h = float(x[1] - x[0])
			xL, xR = x[0], x[-1]

			# M is the number of steps acually taken by Runge Kutta
			M = (len(x)-1)//2

			# so that the real xR is
			xR = xL + 2*h*M

			# calculate the wronskian of the eigenfunction we solve for and the bound sate eigenfunction
			yBoundR = self.right_asyptotic_eigenfunction(mu, xR)
			W[indexDict] = yR[{'yRi':0}]*yBoundR[{'Phii':1}] - yR[{'yRi':1}]*yBoundR[{'Phii':0}]

			# put self.state['x'] back to its original value
			self.state['x'] += centerX

		return W

	def show_eigenfunction(self, mu, selection={}):
		xLIndex, xRIndex = map(int, self.lims_index(selection))
		u = self.state['u'][selection]

		y = self.full_eigenfunction_right(mu, selection)
		x = y.coords['x']

		import matplotlib.pyplot as plt
		plt.plot(x, np.real(y[{'yRi':0}]), color='C0')
		plt.plot(x, np.imag(y[{'yRi':0}]), color='C0', linestyle='--')
		plt.plot(x, np.real(y[{'yRi':1}]), color='C1')
		plt.plot(x, np.imag(y[{'yRi':1}]), color='C1', linestyle='--')
		plt.plot(self.state['x'], u, color='k', linestyle='--')
		plt.ylim(-10,10)
		plt.show()


	def show_wronskianMag(self, xlim=[-1,1], ylim=[-0.5,1], N=101):
		import matplotlib.pyplot as plt

		ReMu = np.linspace(*xlim, N)
		ImMu = np.linspace(*ylim, N)
		ReMuMesh, ImMuMesh = np.meshgrid(ReMu, ImMu)
		Mu = ReMuMesh + 1j*ImMuMesh
		W = lambda mu: self.eigenfunction_wronskian(mu, ODEIntMethod='RungeKuttaArray')
		W = np.vectorize(W)

		plt.pcolormesh(ReMu, ImMu, np.abs(W(Mu)),vmin=0,vmax=20)
		plt.colorbar()
		print('Plotted')
		plt.show()
