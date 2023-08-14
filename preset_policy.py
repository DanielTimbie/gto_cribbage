import random
import math
import numpy as np
from itertools import combinations
import deck_actions
import cribbage_scoring

#function to play a random card 
def random_play_policy(hand, pile):
    tcounter = [0, 0, 0, 0]
    while tcounter != [1, 1, 1, 1]:
            selector = random.randint(0,3)
            val = hand.hand[selector].value
            if val >=10:
                val = 10
            if pile.pileval + val > 31:
                tcounter[selector] = 1
            elif hand.hand[selector].isused == True:
                tcounter[selector] = 1
            elif pile.pileval + val <= 31 and hand.hand[selector].isused == False:
                pile.add_card(hand.hand[selector])
                hand.hand[selector].isused = True
                # print(hand.hand[selector].value)
                retstate = [0,0,0,0]
                retstate[selector] = 1
                return(retstate)
    
    if tcounter == [1, 1, 1, 1]:
            hand.isgo = True
            # print('go!')

def play_function(hand, pile, play_act):
     tcounter = [0, 0, 0, 0]
     first_act = True
     while tcounter != [1, 1, 1, 1]:
               if first_act == True:
                    selector = np.argmax(play_act)
               else:
                    selector = random.randint(0,3)
               val = hand.hand[selector].value
               first_act = False
               if val >=10:
                    val = 10
               if pile.pileval + val > 31:
                    tcounter[selector] = 1
               elif hand.hand[selector].isused == True:
                    tcounter[selector] = 1
               elif pile.pileval + val <= 31 and hand.hand[selector].isused == False:
                    pile.add_card(hand.hand[selector])
                    hand.hand[selector].isused = True
                    # print(hand.hand[selector].value)
                    retstate = [0,0,0,0]
                    retstate[selector] = 1
                    return(retstate)
    
     if tcounter == [1, 1, 1, 1]:
               hand.isgo = True
               return([0,0,0,0])
               # print('go!')

     
     
#function to discard two random cards
def random_discard_policy(hand):
    discards = []
    selector = random.randint(0,5)
    discards = discards + [hand.hand[selector]]
    hand.cribbed = hand.cribbed + [hand.hand[selector]]
    del hand.hand[selector]
    selector = random.randint(0,4)
    discards = discards + [hand.hand[selector]]
    hand.cribbed = hand.cribbed + [hand.hand[selector]]
    del hand.hand[selector]
    return(discards)

#for the model, random discard policy
def random_discard_action_qlearn():
     discard_action_state = np.zeros(15)
     discard_action_state[random.randint(0,14)] = 1
     return(discard_action_state)

#discard policy acting on a hand
def discard_function(disc_act_state,hand):
     #let 1 = [0,0], 2 = [0,1] .... 15 = [4,5]
     disc_act = np.array(disc_act_state).argmax()
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

#function to maximize points from discard w/ random cut card... not quite optimal strategy but close
def max_discard_policy(hand):
    rand_deck = deck_actions.Deck()
    rand_cut = rand_deck.deal(1)
    #print(str(rand_cut[0].value) + ' ' + rand_cut[0].suit)
    k = 0
    # for i in hand.hand:
    #     print(str(i.value) + ' ' + i.suit)
    while k < 5:
        if hand.hand[k].value == rand_cut[0].value and hand.hand[k].suit == rand_cut[0].suit:
            k = 0
            rand_cut = rand_deck.deal(1)
        else:
            k += 1
    #print(str(rand_cut[0].value) + ' ' + rand_cut[0].suit)
    combos = list(combinations(hand.hand,4))
    combolist = []
    for i in combos:
        combolist = combolist + [cribbage_scoring.score_hand(list(i) + rand_cut)]
    kept = combos[np.array(combolist).argmax()]
    hand_temp = []
    for i in kept:
        hand.hand.remove(i)
        hand_temp = hand_temp + [i]
    discards = hand.hand
    hand.cribbed = discards
    hand.hand = hand_temp
    return(discards)