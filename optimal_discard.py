# from datetime import datetime
import random
import math
import matplotlib.pyplot as plt
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

def discard_function(disc_act,hand):
     #let 1 = [0,0], 2 = [0,1] .... 15 = [4,5]
     if disc_act == 0:
          disc = [0,1]
     if disc_act == 1:
          disc = [0,2]
     if disc_act == 2:
          disc = [0,3]
     if disc_act == 3:
          disc = [0,4]
     if disc_act == 4:
          disc = [0,5]
     if disc_act == 5:
          disc = [1,2]
     if disc_act == 6:
          disc = [1,3]
     if disc_act == 7:
          disc = [1,4]
     if disc_act == 8:
          disc = [1,5]
     if disc_act == 9:
          disc = [2,3]
     if disc_act == 10:
          disc = [2,4]
     if disc_act == 11:
          disc = [2,5]
     if disc_act == 12:
          disc = [3,4]
     if disc_act == 13:
          disc = [3,5]
     if disc_act == 14:
          disc = [4,5]
     discards = []
     discards = discards + [hand.hand[disc[0]]]
     discards = discards + [hand.hand[disc[1]]]
     hand.cribbed = hand.cribbed + [hand.hand[disc[0]]]
     hand.cribbed = hand.cribbed + [hand.hand[disc[1]]]
     del hand.hand[disc[1]]
     del hand.hand[disc[0]]
     return(discards)

# #-----------------------------------------------------------
#initializing:

c1 = deck_actions.Card(2,1)
c2 = deck_actions.Card(3,2)
c3 = deck_actions.Card(9,2)
c4 = deck_actions.Card(9,4)
c5 = deck_actions.Card(9,1)
c6 = deck_actions.Card(12,1)
Dealer = True

avg_list = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[])

ctr = 0
while ctr < 10000:

    disc_var = random.randint(0,14)

    score = 0
    opscore = 0
    D = deck_actions.Deck() #deck init
    H = deck_actions.Hand() #hero hand init
    H.isdeal = Dealer
    V = deck_actions.Hand() #villain hand init
    C = deck_actions.Hand() #the crib
    V.isdeal = not Dealer

    ctr_ = 0
    for i in D.deck:
        if i.value == c1.value and i.suit == c1.suit or i.value == c2.value and i.suit == c2.suit or i.value == c3.value and i.suit == c3.suit or \
            i.value == c4.value and i.suit == c4.suit or i.value == c5.value and i.suit == c5.suit or i.value == c6.value and i.suit == c6.suit: 
                D.deck.pop(ctr_)
                print(i.value,i.suit)
        ctr_ += 1

    H.dealt([c1,c2,c3,c4,c5,c6])
    V.dealt(D.deal(6))

    print('player 1 hand:')
    print(get_score_state(H))
    print('player 2 hand:')
    print(get_score_state(V))

    #choosing which cards to discard
    C.dealt(discard_function(disc_var,H))
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

    # #scoring the final hands
    opscore = opscore + cribbage_scoring.score_hand(V.hand + cut_card)
    score = score + cribbage_scoring.score_hand(H.hand + cut_card)
    if Dealer:  
        score = score + cribbage_scoring.score_hand(C.hand + cut_card)
    if not Dealer:  
        opscore = score + cribbage_scoring.score_hand(C.hand + cut_card)

    print('final score:')
    print(str(score) + ', ' + str(opscore))

    avg_list[disc_var].append(score)
    ctr +=1

score_list = []
for i in avg_list:
     score_list.append(sum(i)/len(i))

xx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
labels = ['C1, C2', 'C1, C3', 'C1, C4', 'C1, C5','C1, C6','C2, C3','C2, C4','C2, C5','C2, C6', 'C3, C4', 'C3, C5', 'C3, C6', 'C4, C5', 'C4, C6', 'C5, C6']
plt.bar(xx,score_list)
plt.xticks(xx, labels, rotation='vertical')
plt.show()