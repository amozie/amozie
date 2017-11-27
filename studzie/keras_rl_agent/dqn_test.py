import numpy as np
import matplotlib.pyplot as plt

import gym
import time
from prettytable import PrettyTable
import copy

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, Input, Reshape, concatenate, Merge
from keras.optimizers import Adam, RMSprop
from keras.callbacks import History
from keras import backend as K
import tensorflow as tf

from gym import Env, Space, spaces
from gym.utils import seeding
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.agents.cem import CEMAgent
from rl.agents import SARSAAgent
from rl.callbacks import TrainEpisodeLogger, CallbackList


env = gym.make('MountainCar-v0')
env.seed()
nb_actions = env.action_space.n

x = Input((1,) + env.observation_space.shape)
y = Flatten()(x)
y = Dense(16)(y)
y = Activation('relu')(y)
y = Dense(16)(y)
y = Activation('relu')(y)
y = Dense(16)(y)
y = Activation('relu')(y)
y = Dense(nb_actions)(y)
y = Activation('linear')(y)
model = Model(x, y)

memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000, gamma=.9,
               enable_dueling_network=False, dueling_type='avg', target_model_update=.1, policy=policy)
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=.01, decay=.001), metrics=['mae'])

rewards = []
callback = [TrainEpisodeLogger(), History()]
hist = dqn.fit(env, nb_steps=100000, visualize=False, verbose=2, callbacks=None)
rewards.extend(hist.history.get('episode_reward'))
plt.plot(rewards)

dqn.test(env, nb_episodes=5, visualize=True)

state = env.reset()
action = env.action_space.sample()
print(action)
state_list= []
for i in range(300):
    state_list.append(state)
    # action = np.argmax(dqn.model.predict(np.expand_dims(np.expand_dims(state, 0), 0))[0])
    state, reward, done, _ = env.step(2)
    env.render()
env.render(close=True)

state_arr = np.array(state_list)
plt.plot(state_arr)