# Script to play easy21 game and learn optimal value function
import numpy as np
import pandas as pd
import random
import itertools

from Environment import Environment
from Agents import sarsa, mc, sarsaL, sarsaLApprox
from FeatureVector import FeatureVector

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
    sarsa_agent = sarsaL(state_space_size, lam, gamma = 1, n0 = 10)


    # Train agent
    for i in range(n_iters):
        # initialise the environment
        card_table = Environment()

        sarsa_agent.init_etrace()
        sarsa_agent.init_etrace_log()

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
            sarsa_agent.log_eligibility_trace(s+(a,))

        if record_history:
            sarsa_agent.log_value_function()

    # Return the trained agent
    return sarsa_agent

def train_sarsaLApprox_agent(n_iters, lam, record_history = False):

    # Create feature vector
    agent_features = [range(1,7), range(4,10), range(7,13), range(10,16), range(13, 19), range(16,22)]
    dealer_features = [range(1,5), range(4,8), range(7,11)]

    # Must pass agent features first since agent hand is first in state
    agent_feature_vector = FeatureVector(agent_features, dealer_features)

    # initialise sarsa agent
    sarsa_approx_agent = sarsaLApprox(agent_feature_vector, lam, gamma = 1, n0 = 10)


    # Train agent
    for i in range(n_iters):
        # initialise the environment
        card_table = Environment()

        sarsa_approx_agent.init_etrace()
        sarsa_approx_agent.init_etrace_log()

        # game ends when terminal state is reached
        while card_table.is_state_terminal == False:
            s = card_table.state

            # agent takes action, gets reward
            a = sarsa_approx_agent.choose_action(s)

            s_, r = card_table.step(a)

            sarsa_approx_agent.update_value_function(s,a,r,s_)

        if record_history:
            sarsa_approx_agent.log_weights()

    # Return the trained agent
    return sarsa_approx_agent

def q_from_weights(agent, historic_weight_index = None):

    # Similar to stating the state space size, here need to iterate through the available states
    player_hands = range(1,23)
    dealer_hands = range(1,11)
    actions = range(2)

    state_space_size = [len(player_hands), len(dealer_hands), len(actions)]

    # initialise full value function
    q = np.zeros(state_space_size)

    all_hands = itertools.product(list(player_hands), list(dealer_hands))
    all_hands = itertools.product(player_hands, dealer_hands)

    # Loop through all possible hands, get feature vector, multiply by weights get value for either action
    for s in all_hands:
        for a in actions:
            # convert state to value function index
            ind = tuple(i-1 for i in s) + (a,)
            fvs = agent._fv.state_feature_vector(s)

            q[ind] = agent._q(fvs, a=a, historic_weight_index = historic_weight_index)

    return q

def plot_value_function(q, f = ".\\img\\mc_value_function.png"):
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
    plt.savefig(f)

    return ax

# Compute mean squared error between each and the estimated true value
def mse(q1,q2):
    return (np.square(q1-q2)).mean()



mc_agent_500k = train_mc_agent(500000)
q500k = mc_agent_500k.q
plot_value_function(q500k)

##########################################
#
#
# See how sarsa-lambda compares for different values of lambda
#
#
##########################################

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
    training_log.append((trained_sarsa_agent.log, lam))

errors = [mse(q_sarsa, q500k) for q_sarsa in results]

# PLot results - they don't show much of a pattern
plt.figure()
plt.scatter(lambda_values, errors)
plt.xlabel("lambda")
plt.ylabel("mse")
plt.title("Error after {} episodes".format(sarsa_iter))
plt.show()
plt.savefig(".\\img\\sarsa_lambda_vs_mc.png")


# Get learning curve results - lambda = 0 seems to learn faster. would you expect this for this game?
for log,lam in (training_log[0], training_log[-1]):
    errs = [mse(i, q500k) for i in log]
    iter_number =list(range(len(log)))
    plt.figure()
    plt.scatter(iter_number, errs)
    plt.title("Lambda = {}".format(lam))
    plt.show()
    plt.savefig(".\\img\\sarsa_lambda_{}_learn_rate.png".format(lam))



###############################################
#
#
# Repeate experiment using value function approximation
#
#
###############################################

trained_sarsa_approx_agent = train_sarsaLApprox_agent(1000, 0.1)
plot_value_function(q_from_weights(trained_sarsa_approx_agent), f = ".\\img\\q_sarsa_approx.png")

sarsa_iter = 1000
results = []
training_log = []
lambda_values = np.arange(0,1.1,0.1)
for lam in lambda_values:
    if lam in [0, 1.0]:
        record_history = True
    else:
        record_history = False

    trained_sarsa_agent = train_sarsaLApprox_agent(sarsa_iter, lam, record_history)
    results.append(q_from_weights(trained_sarsa_agent))
    training_log.append((trained_sarsa_agent, lam))

errors = [mse(q_sarsa_approx, q500k) for q_sarsa_approx in results]

# PLot results - they don't show much of a pattern
plt.figure()
plt.scatter(lambda_values, errors)
plt.xlabel("lambda")
plt.ylabel("mse")
plt.title("Error after {} episodes - vfa".format(sarsa_iter))
plt.show()
plt.savefig(".\\img\\sarsa_lambda_vs_mc_vfa.png")


# Get learning curve results - lambda = 0 seems to learn faster. would you expect this for this game?
for trained_agent,lam in (training_log[0], training_log[-1]):
    log = trained_agent.log
    errs = [mse(q_from_weights(trained_agent,historic_weight_index = i), q500k) for i in range(len(log))]
    iter_number =list(range(len(log)))
    plt.figure()
    plt.scatter(iter_number, errs)
    plt.title("Lambda = {}".format(lam))
    plt.show()
    plt.savefig(".\\img\\sarsa_lambda_{}_learn_rate_vfa.png".format(lam))


'''

trained_sarsa_agent = train_sarsaL_agent(100, 1, False)

# Now plot elig trace

def plot_etrace_function(etrace):

    # Need to average over actions to get state value function
    v = np.max(etrace, axis=2)

    # Select just the states we are interested in plotting
    #z = v[11:21,0:10]
    z=v

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
    im = ax.imshow(z)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()

    return ax

'''


##############################################
#
#
# Answering questions
#
#
##############################################
'''
1. Pros vs cons of bootstrapping for easy21?

pros - only get rewards at end state so boot strapping helps avoid outlier estimates of states that are far from end state

cons - seems to give higher error

2. Would bootstrapping help more in balckjack or easy21

i easy21 has more states before end state + reward. More actions to be taken before reciving reward.
- would expect bootstrapping to be more useful in easy21 as can reduce variance of value function approximation, avoid producing widely varying estimates for states
that are 'far' from end state

3. Pros and cons of function approximation in easy21?

- Pro: many states in the game have similar value and so value function approx can make use of this to speed up learning.
- Con: state space isnt large so vfa not necessary

4. How to improve vfa

- many states have similar value where as some states are more important to estimate value of correctly as they are neither clearly safe or close to going bust (ie value function has high gradient at cetain states)
VFA would be improved by accounting for this, with coarser approximation where gradient is low and more granular approximation where gradient is higher.


'''