# Deck class
# Object that models deck of cards in the easy21 game


class Deck():

	def __init__(self):
		pass

	def draw():
    '''
    Draw card. Cards have value of 1-10, uniformly distributed and color
    black or red, 2/3 black and 1/3 red

    Returns:
        tuple. (number, colour)
    '''

	    n = random.randint(1,10)

	    p = random.random()

	    if (p <= 1.0/3.0):
	        # If card is red value is -ve
	        c = -1*n
	    else:
	        c = n

	    return c

