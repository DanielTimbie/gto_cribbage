import numpy as np
import random

#suits = ['hearts', 'diamonds', 'clubs', 'spades']
suits = [1, 2, 3, 4]


class Card: 
    def __init__(self,value,suit):
        self.value = value
        self.suit = suit
        self.isused = False

class Deck:
    def __init__(self):
        self.deck = []
        for value in range(1, 14):
            for suit in suits:
                self.deck = self.deck + [Card(value,suit)]
        self.shuffle()

    def shuffle(self):
        self.deck = random.sample(self.deck, len(self.deck))

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
        self.cribbed = []
        self.isdeal = False
        self.isgo = False
        #self.score = 0

    def dealt(self,cards):
        self.hand = self.hand + cards

class Peg_pile:
    def __init__(self):
        self.pile = []
        self.pileval = 0
        self.resetct = 0 #number of cards to discount in peg pile for runs
        self.playlast = None
    
    def add_card(self,card):

        val = card.value

        #check if >31
        if val<=10: 
            if self.pileval + val > 31:
                return(False)
            else:
                self.pile = self.pile + [val]
                self.pileval = self.pileval + val
                #print('accepted')
                return(True)

        if val>10:
            if self.pileval + 10 > 31:
                return(False)
            else:
                self.pile = self.pile + [val]
                self.pileval = self.pileval + 10
                #print('accepted')
                return(True)



        



        

