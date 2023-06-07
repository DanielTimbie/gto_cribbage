# gto_cribbage

Using Deep Q-Learning to Optimize Cribbage Gameplay 

Cribbage is a popular two person card game with a relatively simple yes distinct ruleset that can be easily modeled. When the game begins, each player is dealt six cards, two of which are chosen to be placed into the ‘crib’, which is a bonus hand claimed by the dealer at the end of each round. A fifth card is then unveiled from the deck, completing a five-card hand for each player (plus the bonus crib for the dealer). Combinations of pairs, triplets, quadruplets, runs, flushes, and combinations of fifteen are used to score points.

The first phase of gameplay consists of each player alternating playing a card in attempts to get as close to the total value 31. Pairs/triplets/quadruplets played during this phase count towards a player’s total score, as do runs and combinations of 15. If a player is the last person to play a card without breaking 31, they receive one point. If a player plays a card that brings the total to 31, they receive two points. The second phase of gameplay consists of tallying up points from pairs, triplets, quadruplets, runs, flushes, and combinations of fifteen. There are other small rules that occasionally come up - a jack of the same suit as the turn card counts for one at the end of the hand. A Jack of clubs unveiled as the turn card grants the dealer one point. The first player to reach 61 points wins the game. 

I will use deep Q-Learning methods to optimize for cribbage gameplay. I will model each hand as an episode, and maximize points for the player in each episode. If time allows, I will model entire 61 point games as episodes - I would expect strategy to change slightly when accounting for different gameplay present at the end of each round (for example, you may want to prevent your opponent from getting any points rather than maximize your own points due to their advantageous position on the board, taking into consideration who plays their hand first to gain points in the hand). Results will be compared to existing optimal models for cribbage. Potential improvements/strategies will be explored. 

# Files:

The program has several relevant files for operation. A short description for each one follows, as well as some notes about what still needs to be done.

# ceck_actions.py

deck_actions.py defines all possible objects relevant to the game, including cards, hands, decks, and piles for the play phase. 'Card' objects have three attributes: value, suit, and a variable called 'isused' which is False by default but set to True if it has been played during the play phase, rendering it inelligible to be played again. 'Deck' objects consist of 52 cards, and are shuffled upon generation. Using 'shuffle', 'cut', 'draw', and 'deal' internal functions of 'Deck' one can perform those actions. 'Hand' begins with an empty list corresponding to its attribute 'hand', and 'isgo' determines whether or not the hand has defaulted to the other player during the play phase due to inability to play a card. 'Peg_pile' objects keep track of cards and scores during the play phase, and can be fed into the scoring function to get current scores. 

# cribbage_scoring.py 

cribbage_scoring.py takes care of scoring during both the play phase and scoring phase. score_peg and score_hand take care of soring for both of these plases respectively. 