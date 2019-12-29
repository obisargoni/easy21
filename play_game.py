# Script to play easy21 game and learn optimal value function
import numpy as np
import pandas as pd
from random import random
from Environment import Environment

import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def play_game(v, n):
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

		if s not in n.keys():
			n[s] = 1
			v[s] = game_reward
		else:
			n[s] += 1
			v[s] += (1/float(n[s]))*(game_reward - v[s])

	return v, n


def estimate_value_function(n_iters):
	
	v = {}
	n = {}
	
	# Function to play many card games in order to estimate value function
	for i in range(n_iters):
		v,n = play_game(v,n)

	# Now unpack value estimation of each state
	k = v.keys()
	#k = list(k)
	v_list = [ list(i) + [v[i]] for i in k]
	df_value = pd.DataFrame(columns = ['hand1', 'hand2', 'action', 'value'], data = v_list)

	return df_value

def plot_value_function(df_value):
	# Plot values

	# Create 2d arrays to represent all possible dealer and player hands
	x = np.arange(1,11) * np.ones(10)[:, np.newaxis]
	y = np.ones([10,10]) * np.arange(12,22)[:, np.newaxis]
	z = np.zeros([10,10])

	for i, r in df_value.iterrows():
	    if r['hand1'] > 11:
	        z[r['hand1']-12,r['hand2']-1] = r['value']

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	plt.show()

	return ax


df_value_10k = estimate_value_function(10000)
plot_value_function(df_value_10k)

df_value_50k = estimate_value_function(50000)
plot_value_function(df_value_50k)

df_value_100k = estimate_value_function(100000)
plot_value_function(df_value_100k)