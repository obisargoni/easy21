# This stored the different learning agents
# Each learning agent is represented as a class
# Each learing agent has attributes which are used to parameterise learning and methods that interact with the environment

import numpy as np 
import random

class mc():
    '''Use monte carlo learning to learn value function
    '''

    # Reward discount factor
    _gamma = None

    # Value function, number of times states visited, number of times states visited this epoch
    _q = None
    _n = None
    _ne = None

    def __init__(self, state_space_size, gamma = 0.1):
        self._q = np.zeros(state_space_size)
        self._n = np.zeros(state_space_size)
        self._ne = np.zeros(state_space_size)
        self._gamma = gamma


    @property
    def q(self):
        return self._q
    
    @property
    def n(self):
        return self._n

    def epsilon_greedy_action(self, s):
        '''Impliment e-greedy action choice. Based on current state, action value function, and number of elapsed episodes
        make an e-greedy action choice. In this way, the agent policy is updated.
        '''

        # Set epsilon - updated each time n[s] changes
        # This is GLIE, since epsilon decays as n[s] increases, number of times a state has been visited increases
        e = 100/(100 + self._n[s].sum())

        if random.random()<=e:
            # Choose random action
            a = random.choice(np.arange(self._q[s].size))
        else:
            # Choose greedy
            a = self._q[s].argmax()

        # Record how many times this state-action pair visited
        sa = s +(a,)
        self._ne[sa] += 1
        return a


    def update_value_function(self, game_reward):
        '''Update the agents value function based on the state the agent was in, the action the agent took, and the reward the agent received

        game_reward: the total reward accumulated through the game
        '''

        self._n += self._ne

        # 1 for all states visited since last update 0 otherwise
        mask_ne = (self._ne > 0).astype(int)
        
        # Reset states visited since last update
        self._ne = np.zeros(self._ne.shape)

        # alhpa controls how much to increment the value function value for each state by, according to number of times those states visited
        alpha = np.divide(1, self._n, out=np.zeros_like(self._n), where=self._n!=0)

        # Update value function for sates visited since last update (these are the states that have lead to this reward)
        self._q += (game_reward - self._q)*alpha*mask_ne

        return None


class sarsa():
    
    # Reward discount factor
    _gamma = None

    # Value function, number of times states visited
    _q = None
    _n = None

    def __init__(self, state_space_size, gamma = 0.1):
        self._q = np.zeros(state_space_size)
        self._n = np.zeros(state_space_size)
        self._gamma = gamma

    @property
    def q(self):
        return self._q
    
    @property
    def n(self):
        return self._n

    def epsilon_greedy_action(self, s):
        '''Impliment e-greedy action choice. Based on current state, action value function, and number of elapsed episodes
        make an e-greedy action choice. In this way, the agent policy is updated.
        '''

        # Set epsilon - updated each time n[s] changes
        # This is GLIE, since epsilon decays as n[s] increases, number of times a state has been visited increases
        e = 100/(100 + self._n[s].sum())

        if random.random()<=e:
            # Choose random action
            a = random.choice(np.arange(self._q[s].size))
        else:
            # Choose greedy
            a = self._q[s].argmax()
        
        # Record how many times this state-action pair visited
        sa = s +(a,)
        self._n[sa] += 1
        return a


    def update_value_function(self,s,a,r,s_):
        '''Update the agents value function based on the state the agent was in, the action the agent took, and the reward the agent received

        s: the state the agent was in
        a: the action the agent took
        r: the reward the agent received
        s_: the new state of the system
        '''

        # Not sure how to check if state is terminal here            
        # Need to get a_, action under this or previous policy?
        # Use this policy since that is the action we would expect to take (haven't updated policy yet either)
        a_ = epsilon_greedy_action(self._q,self._n,s_)
        sa_ = s_ + (a_,)
        expeced_reward = self._q[sa_]

        # Update number of times states visited
        alpha = (1/float(self._n[sa]))

        # Perform backwards view update
        td_error = r + gamma*expeced_reward - self._q[sa]
        self._q += alpha*td_error

        return None


    