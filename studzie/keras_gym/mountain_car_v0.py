import numpy as np
import matplotlib.pyplot as plt

import gym
import time
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


class MountainCarEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self) -> None:
        self.env = gym.make('MountainCar-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _step(self, action):
        step = self.env.step(action)
        step = list(step)
        step[1] = np.abs(step[0][1]) - 0.05
        return tuple(step)

    def _reset(self):
        return self.env.reset()

    def _seed(self, seed=None):
        return self.env.seed(seed)

    def _render(self, mode='human', close=False):
        return self.env.render(mode, close)

    def _close(self):
        return self.env.close()

env = MountainCarEnv()
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

memory = SequentialMemory(limit=10000, window_length=1)
# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, gamma=.9, batch_size=32,
               enable_dueling_network=False, dueling_type='avg', target_model_update=.1, policy=policy)
dqn.compile(Adam(), metrics=['mae'])

hist = dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, callbacks=None)

state = env.reset()
action = env.action_space.sample()
print(action)
state_list= []
for i in range(500):
    action = np.argmax(dqn.model.predict(np.expand_dims(np.expand_dims(state, 0), 0))[0])
    state, reward, done, _ = env.step(2)
    state_list.append(reward)
    env.render()
env.render(close=True)

dqn.test(env, nb_episodes=5, visualize=True)
env.render(close=True)