from __future__ import division
import numpy as np
import scipy
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf
from copy import deepcopy
import warnings
import inspect
import math
import xarray as xr
import itertools
import functools

isnparray = lambda x: isinstance(x, np.ndarray)

def getval(state, key):
	# get a value whether it is a coordinate, data or attribute
	# XXX: it would be nice of xarray did this automatically
	if key in state:
		return state[key]
	elif key in state.attrs:
		return state.attrs[key]

def timeStepFunc(stepFunc):
	from numba import jit
	stepFunc_numba = jit(nopython=True, cache=True)(stepFunc)

	def timestep_wrap(state, tFin, asymptoticBoundary={}, accelerator=None, progressBar=None, **timestepKwargs):
		### include defaults explicitly
		# argnames, varargs, kwargs, defaults = inspect.getargspec(fieldFunc)
		defaults = inspect.getargspec(stepFunc).defaults
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

		# order the array axes
		transposeOrder = [axis for axis in state['u'].dims if axis not in vectorize and axis not in 'x']
		transposeOrder += vectorize
		transposeOrder += 'x'

		if state['u'].dims != transposeOrder:
			state['u']  = state['u'].transpose(*transposeOrder)
			state['ut'] = state['ut'].transpose(*transposeOrder)

		# pass state to the timeStepFunc
		funcArgs = dict((key, getval(state, key)) for key in inspect.getargspec(stepFunc)[0] if getval(state, key) is not None)

		# pass numpy arrays instead of xarray DataArrays
		for arg in funcArgs.keys():
			if isinstance(funcArgs[arg], xr.core.dataarray.DataArray):
				funcArgs[arg] = funcArgs[arg].data

		while np.any(getval(state, 't') < tFin):
			if accelerator == 'numba':
				stepFunc_numba(**funcArgs)
			elif accelerator is None:
				stepFunc(**funcArgs)
			else:
				raise ValueError(f"The accelerator '{accelerator}' is not recognised.  Should be 'numba' or None.")

			# update the time
			if 't' in state.attrs:
				state.attrs['t'] += funcArgs['dt']
			else:
				state['t'] += funcArgs['dt']

			if progressBar:
				progressBar.update()

			# implement asymptotic boundary conditions
			checkRange = 5
			newPoints = 100
			if 'L' in asymptoticBoundary.keys():
				# check if there is anything within checkRange spatial points of the left boundary
				if (abs(state['u'][{'x':slice(0,checkRange)}]-asymptoticBoundary['L']) > 1e-4).any():
					# add another newPoints points on to the end
					x = state['x']
					dx = float(x[1] - x[0])
					newPoints = np.linspace(float(x[0]-newPoints*dx), float(x[0]-dx), newPoints)

					# create new data points for the new region
					sizes = state['u'].sizes
					newSize = dict([(key, sizes[key]) for key in sizes]) # copy dictionary
					newSize['x'] = len(newPoints)

					coords = state['u'].coords
					newCoords = dict([(key, coords[key]) for key in coords]) # copy dictionary
					newCoords['x'] = newPoints

					newData = xr.Dataset(data_vars = {'u':(state['u'].dims, asymptoticBoundary['L']*np.ones([newSize[key] for key in state['u'].dims])),
										  			  'ut':(state['ut'].dims, np.zeros([newSize[key] for key in state['ut'].dims]))}, 
										 coords = newCoords)

					# add t if needed
					if 't' in state.data_vars:
						newData['t'] = state['t']

					newState = xr.concat([newData, state], dim='x')
					newState.attrs = state.attrs
					state = newState

					# update funcArgs
					funcArgs['x'] = state['x'].data
					funcArgs['u'] = state['u'].data
					funcArgs['ut'] = state['ut'].data
			
		return state
	return timestep_wrap
