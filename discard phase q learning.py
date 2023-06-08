from datetime import datetime
import random
import math
import numpy as np
from itertools import combinations
from collections import deque
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from keras.models import Model, load_model
import tensorflow
import matplotlib.pyplot as plt

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

#state with collapsed suit and value attributes
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

def get_score_state_nosuit(hand):
    #temporary function for scoring
    state = [0, 0, 0, 0, 0, 0, 0]
    if hand.isdeal == True:
        state[6] = 1
    if hand.cribbed == []:
        for i in range(0,6):
            state[i] = hand.hand[i].value
    else:
        for i in range(0,4):
            state[i] = hand.hand[i].value
        for i in range(0,2):
            state[i + 4] = hand.cribbed[i].value
    return(state)

def state_flatten(state):
    return(np.array(state).flatten())

#copying from cartpole program to do the q learning
plotting = []

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
    # Output Layer with # of actions: 15 nodes
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='cribbage_score_model')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # model.summary()
    return model

class DQNAgent:
    def __init__(self):

        self.state_size = 7 
        self.action_size = 15 
        self.EPISODES = 50000
        self.memory = deque(maxlen=40000)
        self.gamma = 1  # discount rate (no discounting here - quick hands, no benefit to speed)
        self.epsilon = 0.999
        self.epsilon_min = 0.001
        self.epsilon_decay = 10000 #0.999 #play with this number - more chances to take random actions when larger
        self.batch_size = 64
        self.train_start = 4000000

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # implement the epsilon-greedy policy
    def act(self, hand_state, crib):
        #state = get_score_state(hand_state)
        state = get_score_state_nosuit(hand_state)
        state_for_model = state_flatten(state)
        sample = random.random()
        print(sample)
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        math.exp(-1. * len(self.memory) / self.epsilon_decay) 
        print(eps_threshold)
        if sample > eps_threshold: 
            state_for_model = tensorflow.reshape(state_for_model,shape=(1,self.state_size))
            action_to_take = np.zeros(15)
            action_to_take[np.argmax(self.model.predict([state_for_model]))] = 1
            crib.dealt(discard_function(action_to_take,hand_state))
            return (action_to_take)

        else: #implement the random policy to pick any two cards
            random_act = random_discard_action_qlearn()
            crib.dealt(discard_function(random_act,hand_state))
            print('random action')
            return (random_act)

    # implement the Q-learning
    def replay(self):
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
                target[i][action[i]] = reward[i]

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

            state = get_score_state_nosuit(H)
            # print(H.hand)
            # print(state)
            cut_card = D.deal(1)

            #score for jack on turn for dealer
            if cut_card[0].value == 11:
                if H.isdeal == True:
                    score += 1
                if V.isdeal == True:
                    opscore += 1

            done = False #only one step for each of these... to be expanded on...

            i = 0 #really only running once, but keep this format for future complexity
            while not done:
                action = self.act(H,C) 
                #next_state = get_score_state(H)
                next_state = get_score_state_nosuit(H)
                # print(next_state) next state not important here
                # #scoring the final hands
                opscore = opscore + cribbage_scoring.score_hand(V.hand + cut_card)
                score = score + cribbage_scoring.score_hand(H.hand + cut_card)
                if (counter % 2) == 0:
                    score = score + cribbage_scoring.score_hand(C.hand + cut_card)
                else:
                    opscore = opscore + cribbage_scoring.score_hand(C.hand + cut_card)
                reward = score
                done = True
                print(state, action, reward, next_state, done)
                self.remember(state, action, reward, next_state, done)
                #state = next_state

                #i += 1   <--- previously used to reward longer sessions, now undesired. temp: i = reward
                i = reward

                if done:
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print("episode: {}/{}, score: {}, e: {:.2}, time: {}".format(counter + 1, self.EPISODES, i, self.epsilon,
                                                                                 timestampStr))
                    plotting = plotting + [i]
                    # save model option
                    # if i >= 500:
                    #     print("Saving trained model as cribbage_discard_training.h5")
                    #     self.save("./save/cribbage_discard_training.h5")
                    #     return # remark this line if you want to train the model longer
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

plt.plot(plotting)
plt.title('Score vs. number of attempt: random policy')
plt.xlabel('Number of training sessions')
plt.ylabel('Score')
plt.show()

print(np.mean(plotting[:1000]))
print(np.mean(plotting[-1000:]))