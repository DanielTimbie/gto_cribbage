# from datetime import datetime
import random
import math
# import torch
import numpy as np
from itertools import combinations
# from collections import deque
# from keras.layers import Input, Dense
# from keras.optimizers import Adam, RMSprop
# from keras.models import Model, load_model
# import tensorflow
# import matplotlib.pyplot as plt

import deck_actions
import cribbage_scoring
from preset_policy import random_play_policy, random_discard_policy, max_discard_policy

#function changing a hand into a state for play phase
def get_play_state(hand, cutcard, score, optscore, pile):
    #first four for hand cards, last six for pile data
    state = [0, 0, 0, 0, cutcard[0].value, 0, score, optscore, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0,4):
        if hand.hand[i].isused == False:
            state[i] = hand.hand[i].value
    if hand.isdeal == True:
        state[5] = 1
    for i in range(0,len(pile.pile)):
        state[8+i] = pile.pile[i]
    return state

def get_score_state(hand):
    #first eight for value, final four cribbed
    state = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    if hand.isdeal == True:
        state[6] = [1, 1]
    if hand.cribbed == []:
        for i in range(0,6):
            state[i][0] = hand.hand[i].value
            state[i][1] = hand.hand[i].suit
    else:
        for i in range(0,4):
            state[i][0] = hand.hand[i].value
            state[i][1] = hand.hand[i].suit
        for i in range(0,2):
            state[i + 4][0] = hand.cribbed[i].value
            state[i + 4][1] = hand.cribbed[i].suit
    return(state)

# #-----------------------------------------------------------
#initializing:
score = 0
opscore = 0
D = deck_actions.Deck() #deck init
H = deck_actions.Hand() #hero hand init
H.isdeal = True
V = deck_actions.Hand() #villain hand init
C = deck_actions.Hand() #the crib
P = deck_actions.Peg_pile() #peg pile init
P.playlast = 1
H.dealt(D.deal(6))
V.dealt(D.deal(6))

print('player 1 hand:')
print(get_score_state(H))
print('player 2 hand:')
print(get_score_state(V))

#choosing which cards to discard
C.dealt(random_discard_policy(H))
C.dealt(random_discard_policy(V))

print('player 1 hand after discarding:')
print(get_score_state(H))
print('player 2 hand after discarding:')
print(get_score_state(V))

cut_card = D.deal(1)
print('Cut Card:')
print(str(cut_card[0].value) + ' ' + str(cut_card[0].suit))
if cut_card[0].value == 11:
    if H.isdeal == True:
        score += 1
        #print("Jack turned! One point to Hero")
    if V.isdeal == True:
        opscore += 1
        #print("Jack turned! One point to Villain")

# #sumulated play phase, simulating if only pegging: 
while len(P.pile) < 8:    
    while H.isgo == False or V.isgo == False:  
        #print(P.pileval)
        if (H.isgo == False and P.playlast == 1) or (V.isgo == True and P.playlast == 0):
            #print(get_play_state(H, cut_card, score, opscore, P))            
            random_play_policy(H,P)
            # print(get_play_state(H, cut_card, score, opscore, P))            
            # print('score: ' + str(cribbage_scoring.score_peg(P)))
            score = score + cribbage_scoring.score_peg(P)
            P.playlast = 0
            if P.pileval == 31:
                H.isgo, V.isgo = True, True
        #print(P.pileval)
        if (V.isgo == False and P.playlast == 0) or (H.isgo == True and P.playlast == 1):
            #print(get_play_state(V, cut_card, score, opscore, P))            
            random_play_policy(V,P) 
            # print(get_play_state(V, cut_card, score, opscore, P))            
            # print('score: ' + str(cribbage_scoring.score_peg(P)))
            opscore = opscore + cribbage_scoring.score_peg(P)
            P.playlast = 1
            if P.pileval == 31:
                H.isgo, V.isgo = True, True  
            
    if P.playlast == 0 and P.pileval != 31:
        #print('last card +1 for Hero')
        score += 1

    if P.playlast == 1 and P.pileval != 31:
        #print('last card +1 for Villain')
        opscore += 1

    P.pileval = 0
    P.resetct = len(P.pile)
    H.isgo, V.isgo = False, False

print('states indicating the results of the play phase')
print(get_play_state(H, cut_card, score, opscore, P))            
print(get_play_state(V, cut_card, score, opscore, P))            

# #scoring the final hands


opscore = opscore + cribbage_scoring.score_hand(V.hand + cut_card)
score = score + cribbage_scoring.score_hand(H.hand + cut_card)
score = score + cribbage_scoring.score_hand(C.hand + cut_card)

print('final score:')
print(str(score) + ', ' + str(opscore))

# #-----------------------------------------------------------

