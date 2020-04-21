# Script to play easy21 game and learn optimal value function
import numpy as np
import pandas as pd
import random
from Environment import Environment

from Agents import sarsa, mc, sarsa_lam

import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

intermediate_results = None


def train_mc_agent(n_iters):

    # State space is agent hand x dealer hand x agent actions (22 x 22 x 2)
    state_space_size = [22,22,2]

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
    state_space_size = [22,22,2]

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

def train_sarsa_lam_agent(n_iters, lam):

    # State space is agent hand x dealer hand x agent actions (22 x 22 x 2)
    state_space_size = [22,22,2]

    # initialise sarsa agent
    sarsa_agent = sarsa_lam(state_space_size, lam, gamma = 0.1)  

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
        sarsa_agent.reset_eligibility_trace()

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
    return ((q1-q2)*2).mean()

'''
q10k = monte_carlo_control(10000)
plot_value_function(q10k)

q50k = monte_carlo_control(50000)
plot_value_function(q50k)
'''

mc_agent_500k = train_mc_agent(500000)
plot_value_function(mc_agent_500k.q)

q500k = mc_agent_500k.q

# See how sarsa-lambda compares for different values of lambda
sarsa_iter = 1000
results = []
lambda_values = np.arange(0,1,0.1)
for lam in lambda_values:
    trained_sarsa_agent = train_sarsa_lam_agent(sarsa_iter, lam)
    results.append(trained_sarsa_agent.q)

errors = [mse(q_sarsa, q500k) for q_sarsa in results]

# PLot results - they don't show much of a pattern
plt.figure()
plt.scatter(lambda_values, errors)
plt.show()


# Get learning curve results - lambda = 0 seems to learn faster. would you expect this for this game?
'''
results = sarsa_lambda_control(sarsa_iter, 0, results_to_return = 'all')
errs = [mse(i, q500k) for i in results]
iter_number =list(range(sarsa_iter))
plt.figure()
plt.scatter(iter_number, errs)
plt.show()

results1 = sarsa_lambda_control(sarsa_iter, 1, results_to_return = 'all')
errs = [mse(i, q500k) for i in results1]
iter_number =list(range(sarsa_iter))
plt.figure()
plt.scatter(iter_number, errs)
plt.show()
'''
plot_value_function(q500k)
