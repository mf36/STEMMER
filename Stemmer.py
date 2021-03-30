"""

README

The below code implements
- uniform random experience replay; 
    set algo = 0 - for "RANDOM"
    
- prioritized experience replay;
    set algo = 1 - for "PER"
    
- the replay algo "STEMMER" proposed in the report;
    set algo = 2 - for "STEMMER"

    
weights of the networks 
- are stored in a local pickle file during training
- are loaded during testing 


"""

algo = ["RANDOM", "PER", "STEMMER"]
algo = algo[2]




import random
import pandas as pd 
import numpy as np
import gym 
import os
import pickle

from collections import deque

# import tensorflow.compat.v1 as tf 
import tensorflow as tf 
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
# from prioritized_memory import Memory
from tensorflow import Variable

from datetime import datetime
from sklearn import preprocessing



"""Set Seeds"""
RANDOM_SEED = 1
random.seed(RANDOM_SEED) # set random seed for reproducability
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


stats = [] # container to store rewards over all episodes



"""Utils"""

# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    """
    Binary Tree for quicker sampling
    https://github.com/rlcode/per 
    """

    write = 0


    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0


    # update to the root node
    def propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)


    # find sample on leaf node
    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    # get number of entries in tree
    def total(self):
        return self.tree[0]


    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1


    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self.propagate(idx, change)


    # get priority and sample
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    """Memory class for handling access to Tree 
    https://github.com/rlcode/per
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001


    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity


    def get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a


    def add(self, error, sample):
        p = self.get_priority(error)
        self.tree.add(p, sample)


    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight


    def update(self, idx, error):
        p = self.get_priority(error)
        self.tree.update(idx, p)
        
            
def get_model(input_shape, output_shape, model_type = "dense"):
        
        model = tf.keras.models.Sequential([
        
        layers.Dense(24, activation='relu', input_shape=(input_shape)),
        # layers.Dense(64, activation='relu'),
        # Add another:
        layers.Dense(24, activation='relu'),
        # layers.Dense(24, activation='relu'),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(64, activation='relu'),
        
        layers.Dense(output_shape, activation = "linear")])
        
        model.compile(optimizer=Adam(lr=0.001),
              loss="mse",
              metrics=['accuracy'])
        model.summary()
        return model
    
    
class Agent():
    def __init__(self):
        
        """PARAMETERS"""
        
        """Instantiate Environment and Set Seeds"""
        # self.env = gym.make('SpaceInvaders-ram-v0')
        self.env = gym.make('CartPole-v1')
        self.env.seed(RANDOM_SEED)
        self.env.action_space.seed(RANDOM_SEED)
        
        """Set Environment Variables"""
        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_size = self.action_space.n
        self.episodes = 2000

        """Routing of selected Algorithm"""
        global algo
        self.algo = algo


        """Set Learning Hyper Parameters"""
        self.gamma = 0.95
        self.epsilon  = 1.0
        # self.epsilon_decay = 0.998
        # self.epsilon_decay = 0.99
        # self.explore_step = 5000
        self.explore_step = 10000
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.rand = 0 # initialize random number for e-greedy selection
        self.lr = 0.001
        
        """Set Replay and Buffer Parameters"""
        self.batch_size = 64 # number of samples to consider for each replay 
        self.train_start = 1000 # start training by replay after n records in memory
        # self.buffer_size =  2000 # number of max samples in memory
        self.buffer_size =  20000 # number of max samples in memory
        self.target_update  = 20 # update target network with action weights every n steps
        if self.algo == "RANDOM" or self.algo == "STEMMER":
            self.memory = deque(maxlen = self.buffer_size)
        elif self.algo == "PER" :
            self.memory = Memory(self.buffer_size)

            
            
        self.optimizer = tf.keras.optimizers.Adam(lr = self.lr) 
        
        """Instantiate Action and Target Network"""
        self.action_network = get_model(self.state_space.shape, self.action_space.n)
        self.target_network = get_model(self.state_space.shape, self.action_space.n)
                
        """Hyper Paramters for PER"""
        self.p_e = 0.01
        self.p_a = 0.6
        self.p_beta = 0.4
        self.p_beta_increment_per_sampling = 0.001
 
    def write_pickle(self, obj, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(obj, handle)
    
    
    def read_pickle(self, filepath):
        with open(filepath, 'rb') as handle:
            unserialized_data = pickle.load(handle)
        return unserialized_data
            
        
    def load(self, name):
        self.action_network = load_model(name)
    
    
    def save(self, name):
        self.action_network.save(name, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None)


    #get e-gredy action
    def get_action(self, state):
        self.rand = np.random.random() # get random number in [0,1]
        if self.rand <= self.epsilon: # explore if rand below epsilon
            return random.randrange(self.action_space.n)
        else:
            # return np.argmax(self.action_network.predict(np.expand_dims(state, axis=0))) # else predict action with target-model
            return np.argmax(self.action_network.predict(np.expand_dims(state, axis=0))) # else predict action with target-model
        
        
    #store experiences in memory    
    def remember_random(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    
    def remember_prioritized(self, state, action, reward, next_state, done):
        target = self.action_network.predict(np.expand_dims(state, 0))
        old_val = target[0][action]
        target_next = self.target_network.predict(np.expand_dims(next_state, 0))

        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * np.amax(target_next)
            
        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))
    
    
    
    def replay_train_random(self):
        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= self.epsilon_decay
            
        # randomly sample from memory or stemmer
        if(self.algo == "RANDOM"):
            minibatch = np.array(random.sample(list(self.memory), self.batch_size))
        elif(self.algo == "STEMMER"): # if stemmer, calculate priority exp function
            a = 0.9
            m = 1
            #get exponential distribution 
            stemmer_prio = []
            sum_x = 0
            for x in range(len(self.memory)):
                i = round((a**(m*x)), 10)
                stemmer_prio.append(i)
                sum_x  +=i
            
            
            prio = []
            for i in range(len(stemmer_prio)):
                prio.append(stemmer_prio[i]/sum_x)


            #flip the axis since lowest index in memory is oldest item
            prio = np.flip(prio)

            try:
                prio_indices = np.random.choice(range(len(self.memory)), self.batch_size,replace = True, p = prio) #sample from memory according to probability
            except:
                prio_indices = np.random.choice(range(len(self.memory)), self.batch_size,replace = True) # in seldom cases, values do not exactly sum up to 1 after normalization 
                
            # index = np.random.choice(((self.memory)), self.batch_size, replace = True, p = stemmer_prio)
            minibatch = np.array(self.memory)[prio_indices]
        # randomly sample from memory
        # minibatch = np.array(random.sample(list(self.memory), self.batch_size))

        state = tf.convert_to_tensor(list(minibatch[:,0]))
        action = tf.convert_to_tensor(list(minibatch[:,1]))
        reward = tf.convert_to_tensor(list(minibatch[:,2]))
        next_state = tf.convert_to_tensor(list(minibatch[:,3]))
        done = tf.convert_to_tensor(list(minibatch[:,4]))
    
        target = self.action_network.predict(state)
        # target_next = self.action_network.predict(next_state)
        target_next = self.target_network.predict(next_state)
        
        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i] 
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i])) # action network selects action by max Q value; target network evaluates next best action
            
        #fit model with samples
        self.action_network.fit(state, target, verbose=0)
    
    
    def replay_train_prioritized(self):
        if not self.memory.tree.n_entries >= self.train_start:
            return
 
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= self.epsilon_decay
        minibatch, idxs, is_weights = self.memory.sample(self.batch_size)
        minibatch = np.array(minibatch).transpose()
               
        state = np.vstack(minibatch[0])
        action = (minibatch[1])
        reward = list(minibatch[2])
        next_state = np.vstack(minibatch[3])
        done = minibatch[4]
        done = done.astype(int)
        
        target_next = self.target_network.predict(next_state)
        pred_online = self.action_network.predict(state)

        a = tf.convert_to_tensor(action.reshape(-1, 1).astype(int))
        a = np.squeeze(action.reshape(-1, 1).astype(int))
        one_hot_action = np.zeros((self.batch_size, self.action_size))
        
        #one hot encoding to mark best action
        for i in range(self.batch_size):
            if a[i] == 0:
                one_hot_action[i] = [1,0]
            else:
                one_hot_action[i] = [0,1]
        
        pred_online = tf.math.reduce_sum(pred_online * one_hot_action, axis = 1)
        target = reward + (1 - done) * self.gamma * np.amax(target_next)
        errors = tf.convert_to_tensor(abs(pred_online - target))
        
        pred_online = Variable(pred_online)
        target = Variable(target)
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        
        def loss():
            return tf.reduce_mean(tf.convert_to_tensor(is_weights) * tf.losses.mean_squared_error(target, pred_online))
        
        # self.optimizer.minimize(loss, [pred_online, target])
        self.optimizer.minimize(loss, [pred_online])
        
        # maybe using heapsort instead of binary tree
        # for all experiences sampled, calculate abs difference between target and online network. 
            #- |δt| — Magnitude of our TD error
            #- e — constant assures that no experience has 0 probability to be taken
        # resample over all experiences, but sample experiences with higher difference k times so often; k is a tunable hyper parameter 
            # P(i) = pi  / ∑kpak  normalized by sum of all priority values - see https://medium.com/analytics-vidhya/reinforcement-learning-d3qn-agent-with-prioritized-experience-replay-memory-6d79653e8561
            
            
    def train(self):
        print(datetime.now()) # traing started
        # self.load("cartpole-dqn-online.h5")
        print("=========")
        print(self.algo)
        print("=========")
        global t0
        write_data = 5
        reward = 0
        for e in range(self.episodes+1):

            done= False
            state = self.env.reset() # reset env for new episode
            i = 0
            reward_sum = 0

            while not done: # execute while not dead
                action = self.get_action(state) # get action                
                next_state, reward, done, info = self.env.step(action) # perform action

                # self.env.render()
                
                """for space invaders only"""
                # lives = info["ale.lives"] # get current lives
                # done = (lives !=3) # done if live is lost, not if game is over
                

                reward_sum += reward # add to sum for statistics
                
                if not done or i == self.env._max_episode_steps -1: # save reward if not done, or if max iterations reached
                    reward += reward
                else:
                    reward = -10 # negative reward for dying
                
                # store current experience in memory and train
                if self.algo == "RANDOM" or self.algo == "STEMMER":
                    self.remember_random(state, action, reward, next_state, done)
                    self.replay_train_random()
                elif self.algo == "PER":
                    self.remember_prioritized(state, action, reward, next_state, done)
                    self.replay_train_prioritized()
                
                    
                state = next_state # update state for next iteration
                i+=1

            #udpate epsilon after each episode

            
                reward = 0
            
            self.target_network.set_weights(self.action_network.get_weights()) # perform weights update from action to target
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.episodes, reward_sum, self.epsilon))

            #stats for analysis plottin
            stats.append((e, reward_sum))
                
                
            if(e % write_data == 0): # every k steps
                pd.DataFrame(stats).to_csv("stats_" + self.algo + "_" + t0.strftime("%b %d %Y %H-%M-%S") +".csv", sep = ";", decimal = ",") # write learning to csv
                # print("Saving trained model as cartpole-dqn.h5") 
                self.save("dqn-action_network.h5") # store model locally
            """save model weights every 50 episodes"""
            if(e % 50 == 0): 
                action_network_weights = self.action_network.get_weights()
                target_network_weights = self.target_network.get_weights()
                self.write_pickle(action_network_weights,("action_weights_"+self.algo+".pickle"))
                self.write_pickle(target_network_weights,("target_weights_"+self.algo+".pickle"))



    def test(self):
        print(datetime.now()) # testing started
        # self.load("dqn-action_network.h5")
        self.action_network = get_model(self.state_space.shape, self.action_space.n)
        self.target_network = get_model(self.state_space.shape, self.action_space.n)
        
        self.action_network.set_weights(self.read_pickle("action_weights_"+self.algo+".pickle"))
        self.target_network.set_weights(self.read_pickle("target_weights_"+self.algo+".pickle"))
        
        # self.load("cartpole-dqn - Kopie.h5")
        for e in range(self.episodes):
            # state = image_preprocess(self.env.reset())
            state = (self.env.reset())
            # state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.action_network.predict(np.expand_dims(state, axis=0)))
                next_state, reward, done, _ = self.env.step(action)
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.episodes, i))
                    break
                

"""Instantiate Agent and perform training and/or testing"""                
agent = Agent()
t0 = datetime.now()

agent.train()    
# agent.test()



"""
Code used for the plots can be found below

"""
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

import pandas as pd

rcParams.update({'font.size': 14})


plot_all = False # create either all in one or single algorithms plots

windows  = [1, 2, 5, 10, 20, 50, 100, 200]
line_colors = ['y', 'm', 'c']


for window in windows:




    path_random= r"C:\Users\Marc\stats_RANDOM_May 21 2020 19-42-04.csv"
    path_per= r"C:\Users\Marc\stats_PER_May 21 2020 17-30-17.csv"
    path_stemmer= r"C:\Users\Marc\stats_STEMMER_May 22 2020 05-13-40.csv"

    paths = [path_random, path_per,path_stemmer]    

    data_random = pd.read_csv(path_random, decimal = ",",sep =";" ).iloc[0:2000] # only keep first 2000 records
    data_per = pd.read_csv(path_per, decimal = ",",sep =";" ).iloc[0:2000] # only keep first 2000 records
    data_stemmer = pd.read_csv(path_stemmer, decimal = ",",sep =";" ).iloc[0:1200] # only keep first 2000 records

    len_x = len(data_random)
    len_x_stemmer = len(data_stemmer)
    datas = [data_random, data_per, data_stemmer]
    
    names = ["Random", "PER", "STEMMER"]

    means = []
    stds = []
    #for data in datas:
    for id, data in enumerate(datas):
        
        y = data.iloc[:,2]
        y = np.array(y).reshape(int((len(y)/window)), int((len(y)/int((len(y)/window)))))
        

        if(id !=2):
            x = np.arange(0,len_x, window)
        else:
            x = np.arange(0,len_x_stemmer, window)

        #calculate meand and standard deviation per row

        mean = np.array([np.mean(y[i]) for i in range(len(y))])
        std = np.array([np.std(y[i]) for i in range(len(y))])
        axes = plt.gca()
        axes.set_ylim([0,600])

        m, b = np.polyfit(x, mean, 1)

        plt.xlabel("No. of Episodes")
        plt.ylabel("Total Reward")
        plt.plot(x, mean, label = names[id])
        plt.fill_between(x, mean-std, mean+std, alpha= 0.1)
        
        if( not plot_all):    
            m, b = np.polyfit(x, mean, 1)
            plt.plot(x, m*x+b)
            plt.title(names[id] + " Rewards - average of " + str(window))
            #plt.legend()
            plt.show
            plt.savefig(r"C:\Users\Marc\AI_FINAL\\" + names[id] + "_"+str(window)+".png")
            plt.clf()
    if(plot_all):
        
        plt.title("Rewards over time - average of " + str(window))
        plt.legend()
        plt.show
        plt.savefig(r"C:\Users\Marc\AI_FINAL\\" + "all"+ "_"+str(window)+".png")
        plt.clf()