# Script to play easy21 game and learn optimal value function
import numpy as np
import pandas as pd
from random import random
from Environment import Environment

# State defined by agents hand (1-21) dealers hand (1-21) and agent action (True, False)
v = np.zeros([21,21,2])
v = {}
n = {}

df_value = pd.DataFrame(columns = ["state", "n", "v"])

def play_game():
	'''Play an interation of the game until terminal state is reached. 
	Use total reward received for game to update estimate of state value function.
	'''
	global v
	global n

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


# Function to play many card games in order to estimate value function
for i in range(10000):
	play_game()