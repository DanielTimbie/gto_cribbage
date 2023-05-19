import numpy as np
import random

suits = ['hearts', 'diamonds', 'clubs', 'spades']

class Card: 
    def __init__(self,value,suit):
        self.value = value
        self.suit = suit

class Deck:
    def __init__(self):
        self.deck = [Card(value,suit) for value in range(1, 14) for suit in suits]
        self.shuffle()

    def shuffle(self):
        self.deck = random.shuffle(self.deck)

    def cut(self):
        card = random.sample(self.deck)
        return(card)

    def draw(self): 
        card = self.deck[0]
        self.deck = self.deck[1:]
        return(card)
    
    #enter the number of cards you want to deal
    def deal(self,number):
        pile = []
        k = 1
        while k <= number:
            pile = pile + [self.draw()]
            k += 1
        return(pile)
    
class Hand:
    def __init__(self):
        self.hand = []



        



        

