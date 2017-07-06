from __future__ import division
import numpy as np
import scipy
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf
from copy import deepcopy
from warnings import warn
import inspect
import math
import xarray as xr

from . import ODE

try:
	import matplotlib
	matplotlib.use('TkAgg')
except ImportError:
	warn('Unable to import matplotlib')

isnparray = lambda x: isinstance(x, np.ndarray)

def getval(state, key):
	# get a value whether it is a coordinate, data or attribute
	if key in state:
		return state[key]
	elif key in state.attrs:
		return state.attrs[key]

def stateFunc(fieldFunc):
	def dataset_wrap(*args, **kwargs):
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

		# XXX: save defaults as attributes

		# pass state to the timeStepFunc
		funcArgs = dict((key, getval(state, key)) for key in inspect.getargspec(stepFunc)[0] if getval(state, key) is not None)

		# take time step
		newVals = stepFunc(**funcArgs)

		# Update attributes
		for key in newVals.copy():
			if key in state.attrs:
				state.attrs[key] = newVals.pop(key)

		oldSize = [state[key].size for key in newVals]
		newSize = [newVals[key].size for key in newVals]
		if np.any(oldSize != newSize):
			# the size of the state has changed so we need to build a new one
			state = xr.Dataset(data_vars = dict((key, newVals[key]) for key in state.data_vars.keys()), 
							   attrs = state.attrs)

		return state
	return timestep_wrap


class PDE(object):
	def __init__(self, timeStepFunc, state):
		self.state     = state
		self._initialState = deepcopy(state)
		self.time_step = timeStepFunc

	@property
	def time_step(self):
		return self._time_step

	@time_step.setter
	def time_step(self, timeStep):
		if type(timeStep) == str:
			# if a name of a time step function is given then return that
			self._time_step = self.named_timeStepFuncs[timeStep]
		else:
			# set the time step funciton as the given function
			self._time_step = timeStep

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

		stateValKeys = set(stateVal.data_vars.keys()).union(set(stateVal.attrs)).union(set(stateVal.coords))
		if not self.requiredStateKeys or set(self.requiredStateKeys).issubset(stateValKeys):
			# set the time step funciton as the given dictionary
			self._state = stateVal
		else:
			raise TypeError("The given state should be an xarray Dataset with data_vars or attributes:", self.requiredStateKeys)

	def reset_state(self):
		# reset the state of the field to the state it was in when the instance was first initilized
		self.state = self._initialState

	def time_evolve(self, tFin, **timeStepArgs):		
		# pass the time step function the current state and any additional given arguments
		while getval(self.state, 't') < tFin:
			self.state = self.time_step(self.state, **timeStepArgs)				

	def setticks(self):
		pass

	def plot(self, selection={}, showLims=False, useTex=False, ylim=None):
		from matplotlib import pyplot as plt
		if useTex:
			plt.rc('text', usetex=True)
			plt.rcParams.update({'font.size': 16})

		x, u = [self.state[k] for k in ['x','u']]

		self.state['u'][selection].plot()
		plt.xlim(x[0],x[-1])
		ax = plt.gca()

		if showLims is not False:
			iLims = self.indexLims[selection]
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

		if ylim is not None:
			plt.ylim(ylim[0], ylim[1])

		self.setticks()

		plt.ylabel('$u$')
		plt.xlabel('$x$')

	def show(self, saveFile=None, **kwargs):
		from matplotlib import pyplot as plt
		self.plot(**kwargs)
		if saveFile:
			plt.savefig(saveFile, bbox_inches='tight')
		else:
			plt.show()

	def save_animation(self, saveFile, tFin, fps = 60, writer = None, dpi = 200, codec = None, **timeStepArgs):
		frames = int(tFin / timeStepArgs['dt'])
		saveAnimationDict = {'filename':saveFile, 'fps':fps, 'frames':frames, 'writer':writer, 'dpi':dpi, 'codec':codec}
		self.show_animation(saveAnimationDict = saveAnimationDict, **timeStepArgs)

	def show_animation(self, skipFrames = 0, saveAnimationDict = {}, **timeStepArgs):
		from matplotlib import pyplot as plt
		from matplotlib import animation

		fig = plt.figure()
		ax = plt.axes(xlim=(self.state['x'][0], self.state['x'][-1]), ylim=(-2, 10))
		line, = ax.plot([], [])
		plt.yrange = [-8,8]

		timeLabel = ax.text(0.05, 0.9, '', transform=ax.transAxes)

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
				args = self.state.copy()
				args.update(timeStepArgs)
				self.state = self.time_step(**args)

			line.set_data(self.state['x'], self.state['u'])
			timeLabel.set_text('Time = %.1f' % self.state['t'])
			return line, timeLabel

		# call the animator
		frames = saveAnimationDict.pop('frames', None)
		if frames is not None:
			frames = saveAnimationDict.pop('frames', None)//(skipFrames+1)
		anim = animation.FuncAnimation(fig, update_animation, init_func=init, interval=20, blit=True, frames=frames)

		if saveAnimationDict:
			# save the animation as a file
			anim.save(**saveAnimationDict)
		else:
			plt.show()

	def left_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> -inf
		raise NotImplementedError('Implement left_asyptotic_eigenfunction in a child class')

	def right_asyptotic_eigenfunction(self, mu, x):
		# return the asymptotic value of the bound state eigenfunction as x -> +inf
		raise NotImplementedError('Implement right_asyptotic_eigenfunction in a child class')		

	def eigenfunction_right(self, muList, ODEIntMethod='CRungeKuttaArray', selection={}):
		import matplotlib.pyplot as plt
		x, u = self.state['x'], self.state['u'][selection]
		indexLims = self.indexLims[selection]

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

		yR = np.zeros(yRShape, dtype=np.complex128)
		yR = xr.DataArray(yR, coords=yRCoords, dims=yRDims)

		VFull = self.xLax(muList, selection=selection)

		for index, dummy in np.ndenumerate(np.empty(axisShape)):
			indexDict = dict([(key, index[i]) for i, key in enumerate(self.state.coords) if key!='x' and len(u[key].shape)>0])
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
				V[:] = VFull[indexDict][{'x':slice(xLIndex, xRIndex+1)}][{'mu':muindex}].transpose('x','Vi','Vj').values

				yBoundL_mu = np.zeros(2, dtype=np.complex128)
				yBoundL_mu[:] = yBoundL[{'mu':muindex}].values

				# solve for the Jost eigenfunction which at xL matches the left asymptotic eigenfunction
				# note that stepsize for Runge Kutta 4th order is 2h since it requires midpoint values
				if ODEIntMethod == 'CRungeKuttaArray':
					yR[indexDict][{'mu':muindex}] = ODE.CRungeKuttaArray(2*h, yBoundL_mu, V)
				elif ODEIntMethod == 'RungeKuttaArray':
					yR[indexDict][{'mu':muindex}] = ODE.RungeKuttaArray(2*h, yBoundL_mu, V)[-1]

		return yR

	def eigenfunction_wronskian(self, mu, ODEIntMethod='CRungeKuttaArray', selection={}):
		# solve for the eigenfunction across x as an intial value problem
		# at x[0] the eigenfunction is yBoundL = self.left_asyptotic_eigenfunction(mu)
		# solve for the value of the eigenfunction at x[-1], yR
		yR = self.eigenfunction_right(mu, ODEIntMethod, selection)

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

		W = np.zeros(WShape, dtype=np.complex128)
		W = xr.DataArray(W, WCoords, WDims)

		# iterate over all axis except mu
		axisShape = list(len(u[key]) for key in u.coords if key != 'x' and len(u[key].shape)>0)
		for index, dummy in np.ndenumerate(np.empty(axisShape)):
			indexDict = dict([(key, index[i]) for i, key in enumerate(self.state.coords) if key!='x' and len(u[key].shape)>0])
			
			xLIndex, xRIndex = map(int, self.indexLims[selection][indexDict])
			x = self.state['x'][xLIndex: xRIndex+1]
			h = float(x[1] - x[0])
			xL, xR = x[0], x[-1]

			# M is the number of steps acually taken by Runge Kutta
			M = (len(x)-1)//2

			# so that the real xR is
			xR = xL + 2*h*M

			# calculate the wronskian of the eigenfunction we solve for and the bound sate eigenfunction
			yBoundR = self.right_asyptotic_eigenfunction(mu, xR)
			W[indexDict] = yR[indexDict][{'yRi':0}]*yBoundR[{'Phii':1}] - yR[indexDict][{'yRi':1}]*yBoundR[{'Phii':0}]

		return W

	def show_eigenfunction(self, mu):
		import matplotlib.pyplot as plt
		x, u = self.state['x'], self.state['u']

		xLIndex = 0
		xRIndex = len(x)-1

		x = x[xLIndex:xRIndex+1]
		h = x[1] - x[0]
		xL, xR = x[0], x[-1]
		yBoundL = self.left_asyptotic_eigenfunction(mu, xL)


		V = self.xLax(mu)[xLIndex:xRIndex+1]
		y = ODE.RungeKuttaArray(2*h, yBoundL, V)

		plt.plot(x[::2], y[:,0])
		plt.plot(x[::2], y[:,1])
		plt.plot(x, u[xLIndex:xRIndex+1], color='k', linestyle='--')
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
