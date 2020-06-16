# Script to play easy21 game and learn optimal value function
import numpy as np
import pandas as pd
import random
from Environment import Environment

from Agents import sarsa, mc, sarsaL

import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

intermediate_results = None


def train_mc_agent(n_iters):

    # State space is agent hand x dealer hand x agent actions (22 x 22 x 2)
    state_space_size = [22,10,2]

    mc_agent = mc(state_space_size)
    
    # Function to play many card games in order to estimate value function
    for i in range(n_iters):

        # Initialise a new game
        card_table = Environment()
        game_reward = 0
        state_actions_visited = []
        while card_table.is_state_terminal == False:
            s = card_table.state

            # Adjust state so that it matches with 0 indexed indices of ndarrays
            s = (s[0]-1, s[1]-1)

            # agent takes action, gets reward
            a = mc_agent.choose_action(s)

            sa = s + (a,)
            state_actions_visited.append(sa)

            s, r = card_table.step(a)
            game_reward += r

        # Update agents value function at the end of the game
        mc_agent.update_value_function(game_reward)

    return mc_agent


def train_sarsa_agent(n_iters):

    # State space is agent hand x dealer hand x agent actions (22 x 22 x 2)
    state_space_size = [22,10,2]

    # initialise sarsa agent
    sarsa_agent = sarsa(state_space_size, gamma = 0.1)  

    # Train agent
    for i in range(n_iters):
        # initialise the environment
        card_table = Environment()

        # game ends when terminal state is reached
        while card_table.is_state_terminal == False:
            s = card_table.state
            # Adjust sate so that it matches with 0 indexed indices of ndarrays
            s = (s[0]-1, s[1]-1)

            # agent takes action, gets reward
            a = sarsa_agent.choose_action(s)
            s_, r = card_table.step(a)
            s_ = (s_[0]-1, s_[1]-1)

            sarsa_agent.update_value_function(s,a,r,s_)

    # Return the trained agent
    return sarsa_agent

def train_sarsaL_agent(n_iters, lam, record_history = False):

    # State space is agent hand x dealer hand x agent actions (22 x 22 x 2)
    state_space_size = [22,10,2]

    # initialise sarsa agent
    sarsa_agent = sarsaL(state_space_size, lam, gamma = 1)


    # Train agent
    for i in range(n_iters):
        # initialise the environment
        card_table = Environment()

        sarsa_agent.init_etrace()
        # game ends when terminal state is reached
        while card_table.is_state_terminal == False:
            s = card_table.state
            # Adjust sate so that it matches with 0 indexed indices of ndarrays
            s = (s[0]-1, s[1]-1)

            # agent takes action, gets reward
            a = sarsa_agent.choose_action(s)
            s_, r = card_table.step(a)
            s_ = (s_[0]-1, s_[1]-1)

            sarsa_agent.update_value_function(s,a,r,s_)

        if record_history:
            sarsa_agent.log_value_function()

    # Return the trained agent
    return sarsa_agent

def plot_value_function(q):
    # Plot values

    # Create 2d arrays to represent all possible dealer and player hands
    x = np.arange(1,11) * np.ones(10)[:, np.newaxis]
    y = np.ones([10,10]) * np.arange(12,22)[:, np.newaxis]

    # Need to average over actions to get state value function
    v = np.average(q, axis = 2)

    # Select just the states we are interested in plotting
    z = v[11:21,0:10]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

    return ax

# Compute mean squared error between each and the estimated true value
def mse(q1,q2):
    return (np.square(q1-q2)).mean()



mc_agent_500k = train_mc_agent(500000)
q500k = mc_agent_500k.q
plot_value_function(q500k)

# See how sarsa-lambda compares for different values of lambda
sarsa_iter = 1000
results = []
training_log = []
lambda_values = np.arange(0,1.1,0.1)
for lam in lambda_values:
    if lam in [0, 1.0]:
        record_history = True
    else:
        record_history = False

    trained_sarsa_agent = train_sarsaL_agent(sarsa_iter, lam, record_history)
    results.append(trained_sarsa_agent.q)
    training_log.append(trained_sarsa_agent.log)

errors = [mse(q_sarsa, q500k) for q_sarsa in results]

# PLot results - they don't show much of a pattern
plt.figure()
plt.scatter(lambda_values, errors)
plt.show()


# Get learning curve results - lambda = 0 seems to learn faster. would you expect this for this game?
for log in (training_log[0], training_log[-1]):
    errs = [mse(i, q500k) for i in log]
    iter_number =list(range(len(log)))
    plt.figure()
    plt.scatter(iter_number, errs)
    plt.show()