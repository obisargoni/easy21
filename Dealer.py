# Deck class
# Object that models the dealers moves in the easy21 game

class Dealer():

    _twist_limit = None
    _hand_value = None

    def __init__(self, hv, tl = 17):
        self._twist_limit = tl
        self._hand_value = hv

    @property    
    def hand_value(self):
        return self._hand_value

    def playDealer(self, deck):
        '''
        '''
        while self._hand_value < self._twist_limit:
            card = deck.draw()
            self._hand_value += card

        return self.hand_value