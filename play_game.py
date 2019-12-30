# Script to play easy21 game and learn optimal value function
import numpy as np
import pandas as pd
import random
from Environment import Environment

import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy_action(q,s,k):
	'''Impliment e-greedy action choice. Based on current state, action value function, and number of elapsed episodes
	make an e-greedy action choice. In this way, the agent policy is updated.
	'''

	# Set epsilon
	e = float(1/k)

	if random.random()<=e:
		# Choose random action
		a = random.choice(np.arange(q[s].size))
	else:
		# Choose greedy
		a = q[s].argmax()

	return a

def play_game(q, n, k):
	'''Play an interation of the game until terminal state is reached. 
	Use total reward received for game to update estimate of state value function.
	'''

	states_visited = []
	game_reward = 0
	nA = 0

	# Initialise a new game
	card_table = Environment()
	s = card_table.state

	# get action, record state
	while card_table.is_state_terminal == False:
		a = random() < 0.5
		#s.append(a) Don't think states need to ne indexed by action as well
		s = s +(a,)
		states_visited.append(s)

		# Take action, get new state and reward
		s, r = card_table.step(a)
		game_reward += r
		nA += 1

	# Assign reward to states, update value function
	for s in states_visited:
		# Convert state expressed in terms of card values and True/False action to indexes for identifying that state in the action-value func
		_s = (s[0]-1, s[1]-1, int(s[2]))

		n[_s] += 1
		q[_s] += (1/float(n[_s]))*(game_reward - q[_s])

	return q, n


def monte_carlo_control(n_iters):
	
	# State space is agent hand x dealer hand x agent actions (22 x 10 x 2)
	q = np.zeros([21,10,2])
	n = np.zeros([21,10,2])
	
	# Function to play many card games in order to estimate value function
	for i in range(n_iters):
		q,n = play_game(q,n)

	return q

def plot_value_function(q):
	# Plot values

	# Create 2d arrays to represent all possible dealer and player hands
	x = np.arange(1,11) * np.ones(10)[:, np.newaxis]
	y = np.ones([10,10]) * np.arange(12,22)[:, np.newaxis]

	# Need to average over actions to get state value function
	v = np.average(q, axis = 2)

	# Select just the states we are interested in plotting
	z = v[11:,:]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	plt.show()

	return ax


q10k = monte_carlo_control(10000)
plot_value_function(q10k)

q50k = monte_carlo_control(50000)
plot_value_function(q50k)

q100k = monte_carlo_control(100000)
plot_value_function(q100k)