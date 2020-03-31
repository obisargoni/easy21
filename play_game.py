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

def epsilon_greedy_action(q,n,s):
	'''Impliment e-greedy action choice. Based on current state, action value function, and number of elapsed episodes
	make an e-greedy action choice. In this way, the agent policy is updated.
	'''

	# Set epsilon - updated each time n[s] changes
	e = 100/(100 + n[s].sum())

	if random.random()<=e:
		# Choose random action
		a = random.choice(np.arange(q[s].size))
	else:
		# Choose greedy
		a = q[s].argmax()

	return a

def mc_play_game(q, n, k):
	'''Play an interation of the game until terminal state is reached. 
	Use total reward received for game to update estimate of state value function.
	'''

	state_actions_visited = []
	game_reward = 0

	# Initialise a new game
	card_table = Environment()

	# State is (agent hand, dealer hand)
	s = card_table.state

	# Adjust sate so that it matches with 0 indexed indices of ndarrays
	s = (s[0]-1, s[1]-1)

	# get action, record state
	while card_table.is_state_terminal == False:
		a = epsilon_greedy_action(q,n,s)
		sa = s + (a,)
		state_actions_visited.append(sa)

		# Take action, get new state and reward
		s, r = card_table.step(a)
		s = (s[0]-1, s[1]-1)
		game_reward += r

	# Assign reward to states, update value function
	for sa in state_actions_visited:
		n[sa] += 1
		alpha = (1/float(n[sa]))
		q[sa] += alpha*(game_reward - q[sa])

	return q, n


def monte_carlo_control(n_iters):
	
	# State space is agent hand x dealer hand x agent actions (22 x 10 x 2)
	q = np.zeros([21,10,2])
	n = np.zeros([21,10,2])

	# Episode number
	k = 0
	
	# Function to play many card games in order to estimate value function
	for i in range(n_iters):
		k += 1
		q,n = mc_play_game(q,n,k)

	return q

def sarsa_play_game(table, q, n, E, lam):
	'''Perform a single action and update the state-action value function towards the estimated value using one step lookahead.
	Update the policy
	'''

	gamma = 0.1

	# State is (agent hand, dealer hand)
	s = table.state

	# Adjust sate so that it matches with 0 indexed indices of ndarrays
	s = (s[0]-1, s[1]-1)

	# get action, record state-action
	a = epsilon_greedy_action(q,n,s)
	sa = s + (a,)

	# Take action, get new state and reward
	s_, r = table.step(a)
	s_ = (s_[0]-1, s_[1]-1)

	# Now sample expected future reward from the state agent has arrived in
	if table.is_state_terminal:
		expeced_reward = 0
	else:
		# Need to get a_, action under this or previous policy?
		# Use this policy since that is the action we would expect to take (haven't updated policy yet either)
		a_ = epsilon_greedy_action(q,n,s_)
		sa_ = s_ + (a_,)
		expeced_reward = q[sa_]

	# Update number of times states visited
	n[sa] += 1
	alpha = (1/float(n[sa]))

	# Update eligibility trace
	E = gamma * lam * E
	E[sa] += 1

	# Perform backwards view update
	td_error = r + gamma*expeced_reward - q[sa]
	q += alpha*td_error*E
	
	return q, n, E

def sarsa_control(n_iters, lam):
	# State space is agent hand x dealer hand x agent actions (22 x 10 x 2)
	state_space_size = [21,10,2]
	q = np.zeros(state_space_size)
	n = np.zeros(state_space_size)


	for i in range(n_iters):
		# Initialise a new game
		card_table = Environment()

		# Initialise eligibility trace
		E = np.zeros(state_space_size)
		while card_table.is_state_terminal == False:
			q,n,E = sarsa_play_game(card_table, q, n, E, lam)
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

'''
q10k = monte_carlo_control(10000)
plot_value_function(q10k)

q50k = monte_carlo_control(50000)
plot_value_function(q50k)
'''
# Estiamte true value function using MC
q500k = monte_carlo_control(500000)

# See how sarsa-lambda compares for different values of lambda
sarsa_iter = 1000
results = []
lambda_values = np.arange(0,1,0.1)
for lam in lambda_values:
	q = sarsa_control(sarsa_iter, lam)
	results.append(q)

# Compute mean squared error between each and the estimated true value
def mse(q1,q2):
	return ((q1-q2)*2).mean()

errors = [mse(q_sarsa, q500k) for q_sarsa in results]

# PLot results
plt.scatter(lambda_values, errors)
plt.show()

plot_value_function(q500k)