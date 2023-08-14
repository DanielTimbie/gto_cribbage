from datetime import datetime
import random
import math
import time
import numpy as np
from itertools import combinations
from collections import deque
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from keras.models import Model, load_model
import tensorflow
import matplotlib.pyplot as plt
from scipy import stats

import deck_actions
import cribbage_scoring
from preset_policy import *

#discard phase q-learning looks to pick which cards to discard while maximizing the player's score 
#differential with the opponent. total scores are not taken into consideration. 

# def hand_from_state(state):
#     gen_hand = deck_actions.Hand()
#     if state[6] == [1, 1]:
#         gen_hand.isdeal = True
#     for i in range(4):
#         gen_hand.hand = gen_hand.hand + [deck_actions.Card(state[i][0], state[i][1])]
#         #print(state[i][0], state[i][1])
#     for i in range(2):
#         gen_hand.cribbed = gen_hand.cribbed + [[deck_actions.Card(state[i + 4][0], state[i + 4][1])]]
#         #print(state[i + 4][0], state[i + 4][1])
#     return(gen_hand)

# def state_unflatten(flat_state):
#     state = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
#     for i in range(6):
#         state[i] = [flat_state[2*i], flat_state[2*i + 1]]
#     return(state)    

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

def state_flatten(state):
    return(np.array(state).flatten())

#copying from cartpole program to do the q learning
plotting = []
epsplot = []

# #policy network:
def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(?) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    # Output Layer with # of actions: 4 nodes
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='cribbage_score_model')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # model.summary()
    return model

class DQNAgent:
    def __init__(self):

        self.state_size = 16 
        self.action_size = 4 
        self.EPISODES = 50
        self.hands_per_ep = 10
        self.memory = deque(maxlen=500000)
        self.gamma = 1  # discount rate (no discounting here - quick hands, no benefit to speed)
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999 #0.99995# 0.999
        self.batch_size = 64
        self.train_start = 200

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        global epsplot
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            epsplot = epsplot + [self.epsilon]
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # implement the epsilon-greedy policy
    def act(self, hand, cut, score, opscore, pile):
        state = get_play_state(hand, cut, score, opscore, pile)
        state_for_model = state_flatten(state)
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        math.exp(-1. * len(self.memory) / self.epsilon_decay) 
        print('acting')
        if sample > eps_threshold: 
            print('acting calculated')
            state_for_model = get_play_state(hand, cut, score, opscore, pile)
            state_for_model = tensorflow.reshape(state_for_model,shape=(1,self.state_size))
            action_to_take = np.zeros(4)
            action_to_take[np.argmax(self.model.predict([state_for_model]))] = 1
            return (action_to_take)

        else: #implement the random policy to pick any two cards
            print('acting random')
            random_act = random_play_policy(hand,pile)
            return (random_act)

    # implement the Q-learning
    def replay(self):
        #print(self.memory)
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory... how do i generate more states to learn from?
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        # assign data into state, next_state, action, reward and done from minibatch
        # compute value function of current(call it target) and value function of next state(call it target_next)
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            next_state[i] = minibatch[i][3]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            done.append(minibatch[i][4])###**
        target = self.model.predict(state)###**
        # target_next = target + self.gamma*self.model.predict(next_state[i])
        target_next = self.model.predict(next_state)###**
            # q(s,a) = r + gamma Q(s',pi(s'))

        for i in range(self.batch_size):
            # correction on the Q value for the action used,
            # if done[i] is true, then the target should be just the final reward
            if done[i]:
                # print(action[i])
                # print(list(action[i]).index(1.0))
                #target[i][action[i]] = reward[i] #modifying acion[i] to return the value for which action[i] is 1, so we can index
                target[i][list(action[i]).index(1.0)] = reward[i]

            else: #this never happens because it finishes immediately
                target[i][action[i]] = reward[i] + self.gamma * np.max(target_next[i])
                # else, use Bellman Equation
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # target = max_a' (r + gamma*Q_target_next(s', a'))

        # Train the Neural Network with batches where target is the value function
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def training(self):
        global plotting
        # state = self.env.reset()
        counter = 0
        while counter < self.EPISODES:
            runs = 0
            while runs < self.hands_per_ep:
                #initializing:
                score = 0
                opscore = 0
                D = deck_actions.Deck() #deck init
                H = deck_actions.Hand() #hero hand init
                V = deck_actions.Hand() #villain hand init
                if (counter % 2) == 0:
                    H.isdeal = True
                else:
                    V.isdeal = True
                C = deck_actions.Hand() #the crib
                P = deck_actions.Peg_pile() #peg pile init
                P.playlast = 1
                H.dealt(D.deal(6))
                H.order()
                V.dealt(D.deal(6))    
                V.order()
                C.dealt(random_discard_policy(V))
                C.dealt(random_discard_policy(H))

                # state = get_score_state_nosuit(H)
                # print(H.hand)
                # print(state)
                cut_card = D.deal(1)

                done = False 

                i = 0 
                while not done:
                    while H.isgo == False or V.isgo == False:  
                        if (H.isgo == False and P.playlast == 1) or (V.isgo == True and P.playlast == 0):
                            
                            prevscore = score
                            state = get_play_state(H, cut_card, score, opscore, P) #finds the state
                            
                            action = self.act(H, cut_card, score, opscore, P)
                            action = play_function(H, P, action)
                            print(action)

                            score = score + cribbage_scoring.score_peg(P)

                            next_state = get_play_state(H, cut_card, score, opscore, P) #finds the next state
                            reward = score - prevscore
                            # if H.isgo == False:
                            self.remember(state, action, reward, next_state, done)

                            P.playlast = 0
                            if P.pileval == 31:
                                H.isgo, V.isgo = True, True
                        if (V.isgo == False and P.playlast == 0) or (H.isgo == True and P.playlast == 1):
                            random_play_policy(V,P) 
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

                    i += 1

                    if len(P.pile) >= 8:
                        done = True

                    if done and (runs + 1) == self.hands_per_ep:
                        dateTimeObj = datetime.now()
                        timestampStr = dateTimeObj.strftime("%H:%M:%S")
                        print("episode: {}/{}, score: {}, e: {:.2}, time: {}".format(counter + 1, self.EPISODES, score, self.epsilon,
                                                                                    timestampStr))
                        plotting = plotting + [score]
                        # save model option
                        # if i >= 500:
                        #     print("Saving trained model as cribbage_discard_training.h5")
                        #     self.save("./save/cribbage_discard_training.h5")
                        #     return # remark this line if you want to train the model longer

                runs += 1
                print(counter)
                self.replay()
            counter += 1

    def test(self):
        self.load("./save/cribbage_discard_training.h5")
        #untested so far.... need to rebuild this for cribbage model as opposed to cartpole

        # for e in range(self.EPISODES):
        #     state = self.env.reset()
        #     state = np.reshape(state[0], [1, self.state_size])
        #     done = False
        #     i = 0
        #     while not done:
        #         # self.env.render()
        #         action = np.argmax(self.model.predict(state))
        #         next_state, reward, done, _ = self.env.step(action)
        #         state = np.reshape(next_state[0], [1, self.state_size])
        #         i += 1
        #         if done:
        #             print("episode: {}/{}, score: {}".format(e + 1, self.EPISODES, i))
        #             break


if __name__ == "__main__":
    agent = DQNAgent()
    agent.training()
    # print(agent.memory)
    # agent.test()

# def linearfit(x, slope, intercept):
#     yret = slope*x + intercept
#     return(yret)

# x = np.arange(0,len(plotting))
# slope, intercept, r_value, p_value, std_err = stats.linregress(x, plotting)

# window = 100
# average_data = []
# for ind in range(len(plotting) - window + 1):
#     average_data.append(np.mean(plotting[ind:ind+window]))
# for ind in range(window - 1):
#     average_data.insert(0, np.nan)

plt.plot(plotting)
# plt.plot(x, average_data)
# print('slope:')
# print(slope)
# plt.plot(x,linearfit(x, slope, intercept))
plt.title('Score vs. number of attempt')
plt.xlabel('Number of training sessions')
plt.ylabel('Score')
plt.show()

plt.plot(epsplot)
plt.title('Epsilon value vs. number of training sessions')
plt.xlabel('Number of training sessions')
plt.ylabel('Epsilon value')
plt.show()

print(np.mean(plotting[:100]))
print(np.mean(plotting[-100:]))