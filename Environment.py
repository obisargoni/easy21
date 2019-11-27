# Environment class
# Object that form part of the environment the player agent interacts with and gets rewards from

import random
import .Deck


class Environment():

    deck = None
    dealer = None

    def __init__(self, dck, dlr):
        # Initialise a deck of cards and a dealer
        deck = dck
        dealer = dlr



    def step(s, a):
        '''
        Function impliments the easy21 game. Takes state (dealers card and players cards) and an action (either hit ot stick)
        as inputs and returns a sample of the next state

        Args:
            s: tuple. Contains the dealers card and the players card
            a: boolean. Either hit (True) or stick (False)

        Returns:
            tuple. Sample of state resunting from input state and action.
        '''

        assert isinstance(a,bool)

        # Unpack state to get dealers state and agents state
        sA, sD = s

        # Initialise reward
        r = 0

        # Check if state is terminal - not sure if this should be here or after each action
        if isTerminal(s):
            r = 0 #reward(s) - reward needs to go to zero once terminal state reached
            return (s, r)

        # Not terminal so agent makes action
        else:
            # Chosen to stick.
            if a == False:            
                # Play dealer to get its end state
                sD = playDealer(sD)
                s = (sA, sD)

            else:
                # Generate sample of new state
                card = draw()
                sA += card
                s = (sA, sD)

            r = reward(s)

            return (s,r)




    def reward(s):
        '''
        Calculate the reward of a state.

        Args:
            s: tuple. The state of the system which are the cards of the agent and the dealer.

        Returns:
            int. The value of the state

        NEEDS WORK, QUITE MESSY
        '''
        sA, sD = s

        if isTerminal(s) == False:
            return 0

        if (sA > 21) | (sA < 0):
            r = -1
        elif (sD > 21):
            r = 1
        elif (sD > sA):
            r = -1
        elif (sD < sA):
            r = 1
        else:
            r = 0

        return r


    def isTerminal(s):
        '''
        '''
        sA, sD = s

        if (sA > 21) | (sA < 0) | (sD >= 17):
            return True
        else:
            return False

    def firstDraw():
        return (abs(draw()), abs(draw()))