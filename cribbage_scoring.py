import numpy as np
from itertools import combinations

#input a four card hand and score it
def score_hand(cards):
    if len(cards) > 3: 
        print("error: more than four cards in your hand")

    score = 0

    #checking for pairs, trips, quads: 
    dup_count = 0 #a pair will make 2, trips will make 6, quads will make 12
    for i in cards:
        for j in cards:
            if i == j:
                dup_count += 1
    score += dup_count

    #checking for fifteens:
    num_combo = 
    


            

    
