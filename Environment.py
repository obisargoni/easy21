# Environment class
# Object that form part of the environment the player agent interacts with and gets rewards from

import random
from Deck import Deck
from Dealer import Dealer


class Environment():

    _deck = None
    _dealer = None
    _agent_hand = None
    _terminal = None

    def __init__(self):
        # Initialise a deck of cards and a dealer
        self._deck = Deck()
        self._dealer = Dealer(abs(self._deck.draw()))
        self._agent_hand = abs(self._deck.draw())
        self._terminal = False

    @property
    def agent_hand(self):
        return self._agent_hand
    
    @property
    def dealer_hand(self):
        return self._dealer.hand_value

    @property
    def is_state_terminal(self):
        return self._terminal

    @property
    def state(self):
        # Limit agent and dealer hands to max 22, since at this value or above they go bust. Similarly limit dealer hand to 10 since this is all agent sees.
        ah = min(self.agent_hand, 22)
        dh = min(self.dealer_hand, 10)
        return (ah, dh)
    

    def step(self, a):
        '''
        Function impliments the easy21 game. Takes state (dealers card and players cards) and an action (either hit ot stick)
        as inputs and returns a sample of the next state

        Args:
            s: tuple. Contains the dealers card and the players card
            a: boolean. Either hit (True) or stick (False)

        Returns:
            tuple. Sample of state resunting from input state and action.
        '''
        # Possible actions are True and False
        if isinstance(a, bool):
            pass
        elif a in [0,1]:
            a = bool(a)
        else:
            raise ValueError

        # Check if state is terminal - not sure if this should be here or after each action
        if self._terminal:
            # Reward for terminal state is given when first transition into this state.
            # Subsequent rewards for being in the terminal state are 0
            return (self.state, 0)

        # Not terminal so agent makes action
        # Environment generates a sample of new states
        else:
            # Chosen to stick.
            if a == False:            
                # Play dealer to get its end state
                self._dealer.playDealer(self._deck)

            else:
                # Draw card for agent
                self._agent_hand += self._deck.draw()

            self._terminal = self.isTerminal()
            r = self.reward()
            s = self.state
            
            return (s,r)




    def reward(self):
        '''
        Calculate the reward of a state.

        Args:
            s: tuple. The state of the system which are the cards of the agent and the dealer.

        Returns:
            int. The value of the state

        NEEDS WORK, QUITE MESSY
        '''
        ah = self.agent_hand
        dh = self.dealer_hand

        if self._terminal == False:
            return 0

        if (ah > 21) | (ah < 1):
            r = -1
        elif (dh > 21) | (dh < 1):
            r = 1
        elif (dh > ah):
            r = -1
        elif (dh < ah):
            r = 1
        else:
            r = 0

        return r


    def isTerminal(self):
        '''
        '''
        ah = self.agent_hand
        dh = self.dealer_hand

        if (ah > 21) | (ah < 1) | (dh > 16) | (dh < 1):
            return True
        else:
            return False