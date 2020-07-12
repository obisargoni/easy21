# This stored the different learning agents
# Each learning agent is represented as a class
# Each learing agent has attributes which are used to parameterise learning and methods that interact with the environment

import numpy as np 
import random


class rl_agent():

    def __init__(self, state_space_size, gamma, n0):
        self._sss = state_space_size

        # Value function, number of times states visited, number of times states visited this episode
        self._q = np.zeros(self._sss)
        self._n = np.zeros(self._sss)

        # Reward discount factor
        self._gamma = gamma

        self._n0 = n0

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
        e = self._n0/(self._n0 + self._n[s].sum())

        if random.random()<=e:
            # Choose random action
            a = random.choice(np.arange(self._q[s].size))
        else:
            # Choose greedy
            a = self._q[s].argmax()

        return a

    def choose_action(self, s):

        a = self.epsilon_greedy_action(s)

        # Record how many times this state-action pair visited
        sa = s +(a,)
        self._n[sa] += 1
        return a



class mc(rl_agent):
    '''Use monte carlo learning to learn value function
    '''

    def __init__(self, state_space_size, gamma = 1, n0 = 100):

        super().__init__(state_space_size, gamma, n0)
        self._ne = np.zeros(state_space_size)

    def choose_action(self, s):

        a = self.epsilon_greedy_action(s)

        # Record how many times this state-action pair visited
        sa = s +(a,)
        self._ne[sa] += 1
        return a


    def update_value_function(self, game_reward):
        '''Update the agents value function based on the state the agent was in, the action the agent took, and the reward the agent received

        game_reward: the total reward accumulated through the game
        '''

        # Update th total number of times each state visited with numbers from this episode
        self._n += self._ne

        # alhpa controls how much to increment the value function value for each state by, according to number of times those states visited
        alpha = np.divide(1, self._n, out=np.zeros_like(self._n), where=self._n!=0)

        # 1 for all states visited since last update 0 otherwise
        mask_ne = (self._ne > 0).astype(int)

        # Update value function for sates visited since last update (these are the states that have lead to this reward)
        self._q += (game_reward - self._q)*alpha*mask_ne

        # Reset states visited since last update
        self._ne = np.zeros(self._ne.shape)

        return None


class sarsa(rl_agent):

    def __init__(self, state_space_size, gamma = 1.0, n0 = 100):
        super().__init__(state_space_size, gamma, n0)


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
        a_ = self.choose_action(s_)
        sa_ = s_ + (a_,)
        expeced_reward = self._q[sa_]

        sa = s + (a,)

        # Update number of times states visited
        alpha = (1/float(self._n[sa]))

        # Perform backwards view update
        td_error = r + self._gamma*expeced_reward - self._q[sa]
        self._q[sa] += alpha*td_error

        return None


class sarsaL(rl_agent):

    def __init__(self, state_space_size, lam, gamma = 1, n0 = 100):
        super().__init__(state_space_size, gamma, n0)

        # lambda, use to calculate eligibility trace
        self._lam = lam

        # Log, used to record value functions during training
        self._log = []

        self._Elog = []
        self._slog = []

    @property
    def log(self):
        return self._log

    @property
    def Elog(self):
        return self._Elog

    @property
    def slog(self):
        return self._slog
    

    def init_etrace(self):
        self._E = np.zeros(self._sss)

    def init_etrace_log(self):
        self._Elog = []
        self._slog = []

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
        a_ = self.epsilon_greedy_action(s_)
        sa_ = s_ + (a_,)
        expeced_reward = self._q[sa_]

        sa = s + (a,)

        # Update number of times states visited
        alpha = np.divide(1, self._n, out=np.zeros_like(self._n), where=self._n!=0)

        # Update eligibility trace
        self._E = self._gamma * self._lam * self._E
        self._E[sa] += 1

        # Perform backwards view update - is mask needed here?
        td_error = r + self._gamma*expeced_reward - self._q[sa]
        self._q += alpha*td_error*self._E

        return None

    def log_value_function(self):
        self._log.append(self._q.copy())

    def log_eligibility_trace(self, s):
        self._Elog.append(self._E.copy())
        self._slog.append(s)


class sarsaLApprox():

    def __init__(self, sss, lam, gamma = 1.0, n0 = 100):

    	# Size of feature vector, use as state space size
        self._sss = sss

        # Weights vector
        self._w = np.zeros(self._sss)

        # Number of times state visited - not used due to constant alpha and exploration rate
        self._n = None

        # Reward discount factor
        self._gamma = gamma

        # Constant used to control exploration
        self._n0 = n0

        # lambda, use to calculate eligibility trace
        self._lam = lam

        # Log, used to record value functions during training
        self._log = []

        self._Elog = []
        self._slog = []

    def q(self, s):
        q = np.dot(self._w,s)
        return q

    def q(self, s, a):
        q = np.dot(self._w,s)
        return q[a]    

    @property
    def log(self):
        return self._log

    @property
    def Elog(self):
        return self._Elog

    @property
    def slog(self):
        return self._slog


    def epsilon_greedy_action(self, s):
        '''Impliment e-greedy action choice. Based on current state, action value function, and number of elapsed episodes
        make an e-greedy action choice. In this way, the agent policy is updated.
        '''

        # Set epsilon - updated each time n[s] changes
        # This is GLIE, since epsilon decays as n[s] increases, number of times a state has been visited increases
        #e = self._n0/(self._n0 + self._n[s].sum())
        e = 0.05 # constant rate of exploration 

        if random.random()<=e:
            # Choose random action
            a = random.choice(np.arange(self.q(s).size))
        else:
            # Choose greedy
            a = self.q(s).argmax()

        return a

    def choose_action(self, s):

        a = self.epsilon_greedy_action(s)
        return a
    

    def init_etrace(self):
        self._E = np.zeros(self._sss)

    def update_etrace(self, s, a):
    	# Reduce eligibility of all elements
    	self._E = self._gamma * self._lam * self._E

    	# Get index to update from state vector and increase eligibility of these elements
    	inds = np.where(s==1)[0]
    	self._E[a, inds] += 1

    def init_etrace_log(self):
        self._Elog = []
        self._slog = []

    def update_value_function(self,s,a,r,s_):
        '''Update the agents value function based on the state the agent was in, the action the agent took, and the reward the agent received

        s: the state the agent was in
        a: the action the agent took
        r: the reward the agent received
        s_: the new state of the system
        '''

        # Update eligibility trace
        self.update_etrace(s,a)

        # Not sure how to check if state is terminal here            
        # Need to get a_, action under this or previous policy?
        # Use this policy since that is the action we would expect to take (haven't updated policy yet either)
        a_ = self.epsilon_greedy_action(s_)
        expeced_reward = self.q(s_, a_)

        # Perform backwards view update - is mask needed here?
        td_error = r + self._gamma*expeced_reward - self.q(sa)

        # TD update those ewights corresponding to the feature vector elements
        alpha = 0.01
    	self._w += alpha * td_error * self._E

        return None

    def log_value_function(self):
        self._log.append(self._q.copy())

    def log_eligibility_trace(self, s):
        self._Elog.append(self._E.copy())
        self._slog.append(s)