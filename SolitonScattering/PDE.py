from __future__ import division
import numpy as np
import scipy
from scipy import sqrt, cos, sin, arctan, exp, cosh, pi, inf
from copy import deepcopy
from warnings import warn

from . import ODE

try:
	import matplotlib
	matplotlib.use('TkAgg')
except ImportError:
	warn('Unable to import matplotlib')

class PDE(object):
	def __init__(self, timeStepFunc, **state):
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
		stateVal should be a dict
		either with {'t', 'x', 'u', 'ut'}
		or {'solName', ...} where 'solName' is the name of a known solution given in get_solutions add the other elements of the 
			dictionary should be the solArgs required there
		"""
		if 'solName' in stateVal:
			# if a name of a solution is given then use that function name to create the state
			self._state = self.named_solutions[stateVal.pop('solName')](**stateVal)
		elif not self.requiredStateKeys or set(stateVal.keys()) == set(self.requiredStateKeys):
			# set the time step funciton as the given dictionary
			self._state = stateVal
		else:
			# warn("The given state is not a dictionary with keys:", self.requiredStateKeys)
			pass

	def reset_state(self):
		# reset the state of the field to the state it was in when the instance was first initilized
		self.state = self._initialState

	def time_evolve(self, tFin, **timeStepArgs):
		while self.state['t'] < tFin:
			# pass the time step function the current state and any additional arguments it needs
			# by combining the self.state and timeStepArgs dictionaries
			args = self.state.copy()
			args.update(timeStepArgs)
			self.state = self.time_step(**args)

	def show_state(self):
		from matplotlib import pyplot as plt
		x, u = [self.state[k] for k in ['x','u']]
		plt.plot(x, u, label='u')
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

	def eigenfunction_right(self, mu, ODEIntMethod='CRungeKuttaArray'):
		import matplotlib.pyplot as plt
		x, u = self.state['x'], self.state['u']

		xLIndex = 0
		xRIndex = len(x)-1

		x = x[xLIndex:xRIndex+1]
		h = x[1] - x[0]
		xL, xR = x[0], x[-1]
		yBoundL = self.left_asyptotic_eigenfunction(mu, xL)

		V = self.xLax(mu)[xLIndex:xRIndex+1]
		if ODEIntMethod == 'CRungeKuttaArray':
			yR = ODE.CRungeKuttaArray(2*h, yBoundL, V)
		elif ODEIntMethod == 'RungeKuttaArray':
			yR = ODE.RungeKuttaArray(2*h, yBoundL, V)[-1]

		return yR

	def eigenfunction_wronskian(self, mu, ODEIntMethod='CRungeKuttaArray'):
		# solve for the eigenfunction across x as an intial value problem
		# at x[0] the eigenfunction is yBoundL = self.left_asyptotic_eigenfunction(mu)
		# solve for the value of the eigenfunction at x[-1], yR

		x = self.state['x']

		# XXX: need to fix xRIndex based on 'smoothness' of the field at xR
		xLIndex = 0
		xRIndex = len(x)-1

		xL, xR = x[xLIndex], x[xRIndex]

		fullx = self.state['x']
		h = fullx[1] - fullx[0]

		# XXX: At the moment the fastest way to take a vector of mu is to just loop over mu.
		# if mu is not given as an array make it an array of length 1
		if not hasattr(mu, '__iter__'):
			mu = np.array([mu])

		# get initial condition y(xL)
		yBoundL = self.left_asyptotic_eigenfunction(mu, xL)

		# get xLax in the interval [xL, xR] with spacing h
		# V.shape = (#values of mu, #points x, size of lax pair, size of lax pair)
		V = self.xLax(mu)[xLIndex:xRIndex+1]

		# solve for the Jost eigenfunction which at xL matches the left asymptotic eigenfunction
		# note that stepsize for Runge Kutta 4th order is 2h since it requires midpoint values
		if ODEIntMethod == 'CRungeKuttaArray':
			yR = np.array([ODE.CRungeKuttaArray(2*h, yBoundL[i], V[i]) for i in range(len(mu))])
		elif ODEIntMethod == 'RungeKuttaArray':
			yR = np.array([ODE.RungeKuttaArray(2*h, yBoundL[i], V[i])[-1] for i in range(len(mu))])

		# M is the number of steps acually taken by Runge Kutta
		M = (V.shape[1]-1)//2

		# so that the real xR is
		xR = xL + 2*h*M

		# calculate the wronskian of the eigenfunction we solve for and
		# the bound sate eigenfunction
		yBoundR = self.right_asyptotic_eigenfunction(mu, xR)
		return yR[:,0]*yBoundR[:,1] - yR[:,1]*yBoundR[:,0]

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
