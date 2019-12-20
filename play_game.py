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
	global df_value

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
		i = df_value.loc[df_value["state"] == s].index

		assert len(i) in [0,1]

		if len(i) == 0:
			row = {"state":s, "n":1, "v":game_reward}
			df_value = df_value.append(row, ignore_index = True)
		else:
			ind = i[0]
			df_value.loc[ind, "n"] += 1
			df_value.loc[ind, "v"] += (1/float(df_value.loc[ind, "n"]))*(game_reward - df_value.loc[ind, "v"])


# Function to play many card games in order to estimate value function
for i in range(10000):
	play_game()