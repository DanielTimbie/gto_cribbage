import numpy as np
from itertools import combinations

#input a five card hand and score it. 
#the last card in the hand is the cut card
def score_hand(cards):
    if len(cards) > 5: 
        print("error: more than five cards in your hand")

    score = 0
    val_list = [cards[0].value, cards[1].value, cards[2].value, cards[3].value, cards[4].value]
    run_list = [cards[0].value, cards[1].value, cards[2].value, cards[3].value, cards[4].value]
    suit_list = [cards[0].suit, cards[1].suit, cards[2].suit, cards[3].suit, cards[4].suit]

    #checking for pairs, trips, quads: 
    #a pair will make 2, trips will make 6, quads will make 12
    for i in cards:
        for j in cards:
            if i.value == j.value:
                score += 1
    score -= 5

    #checking for fifteens:
    val_list_fifteen = val_list
    ctr_fifteen = 0
    for i in val_list_fifteen:
        if i > 10:
            val_list_fifteen[ctr_fifteen] = 10
        ctr_fifteen += 1 
    for i in range(2,len(cards) + 1):
        num_combo = list(combinations(val_list_fifteen,i))
        for i in num_combo:
            s = sum(i)
            if s == 15:
                score += 2
                #print('fifteen')

    #checking for flushes:
    ctr_suit = 0 
    suit_init = suit_list[0]
    for i in suit_list:
        if i == suit_init:
            ctr_suit += 1
    if ctr_suit == 5:
        score +=5
        #print('5c flush')
    elif ctr_suit ==4:
        ctr_suit4 = 0 
        suit_init = suit_list[0]
        for i in suit_list[:-1]:
            if i == suit_init:
                ctr_suit4 += 1
        if ctr_suit4 == 4:
            score +=4
            #print('4c flush')

    #checking if jack in hand matches cut card
    for i in cards[:-1]: 
        if i.value == 11 and i.suit == cards[4].suit:
            #print('j match cut')
            score +=1

    #checking for runs
    val_list_runs = run_list
    fivecr = False
    fourcr = False
    run_combo = []
    for i in range(3, len(cards) + 1):
        run_combo = run_combo + list(combinations(val_list_runs,i))
    run_combo.reverse()
    for j in run_combo:
        k = list(j)
        k.sort()
        if len(k) == 5 and k[0] + 1 == k[1] and k[1] + 1 == k[2]\
            and k[2] + 1 == k[3] and k[3] + 1 == k[4]:
            score += 5
            #print('5c run')
            fivecr = True
        elif fivecr == False and len(k) == 4 and k[0] + 1 == k[1] and k[1] + 1 == k[2]\
            and k[2] + 1 == k[3]:
            score += 4
            #print('4c run')
            fourcr = True
        elif fourcr == False and fivecr == False and len(k) == 3 and k[0] + 1 == k[1] and k[1] + 1 == k[2]:
            score += 3
            #print('3c run')
                
    return(score)

def score_peg(peg_pile_entry):

    peg_pile = peg_pile_entry.pile
    score = 0 

    #check for 31
    if peg_pile_entry.pileval == 31:
        score += 2
        #print('31 for 2')

    #checking for pairs
    if len(peg_pile) - peg_pile_entry.resetct > 1:
        if peg_pile[-2] == peg_pile[-1]:
            score += 2
            #print('pair')
    
    #checking for trips
    if len(peg_pile) - peg_pile_entry.resetct > 2:
        if peg_pile[-2] == peg_pile[-1] and peg_pile[-3] == peg_pile[-2]:
            score += 6
            #print('trips')

    #checking for quads
    if len(peg_pile) - peg_pile_entry.resetct > 3:
        if peg_pile[-2] == peg_pile[-1] and peg_pile[-3] == peg_pile[-2] and peg_pile[-4] == peg_pile[-3]:
            score += 12
            #print('quads')

    # #checking for fifteen:
    if peg_pile_entry.pileval == 15:
        score += 2
        #print('fifteen')

    #checking for runs:
    val_list_runs = peg_pile
    scoreinit = score
    for i in range(0,len(peg_pile) -2 - peg_pile_entry.resetct):
        run_combo = list(combinations(val_list_runs[-(len(peg_pile) - peg_pile_entry.resetct - i):],len(peg_pile) - peg_pile_entry.resetct - i))
        if score != scoreinit:
            break
        for j in run_combo:
            k = list(j)
            k.sort()
            if score != scoreinit:
                break
            if len(k) == 7 and k[0] + 1 == k[1] and k[1] + 1 == k[2]\
                and k[2] + 1 == k[3] and k[3] + 1 == k[4] and k[4] + 1 == k[5]\
                    and k[5] + 1 == k[6]:
                score += 7
                #print('run of seven')
            elif len(k) == 6 and k[0] + 1 == k[1] and k[1] + 1 == k[2]\
                and k[2] + 1 == k[3] and k[3] + 1 == k[4] and k[4] + 1 == k[5]:
                score += 6
                #print('run of six')
            elif len(k) == 5 and k[0] + 1 == k[1] and k[1] + 1 == k[2]\
                and k[2] + 1 == k[3] and k[3] + 1 == k[4]:
                score += 5
                #print('run of five')
            elif len(k) == 4 and k[0] + 1 == k[1] and k[1] + 1 == k[2]\
                and k[2] + 1 == k[3]:
                score += 4
                #print('run of four')
            elif len(k) == 3 and k[0] + 1 == k[1] and k[1] + 1 == k[2]:
                score += 3
                #print('run of three')

    return(score)


