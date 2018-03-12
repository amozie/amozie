import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import keras
from keras import backend as K
import tensorflow as tf

from rl.agents import *
from rl.policy import *
from rl.memory import *
from rl.random import *
import gym
from gym import Env, Space, spaces


class TestEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num, step_size) -> None:
        self.num = num
        self.step_size = step_size

        self.data = np.cumsum(np.random.choice([-1, 0, 1], self.num))
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        self.step_begin = 0
        self.step_end = self.step_begin + self.step_size

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, (step_size,), float)

    def step(self, action):
        self.step_begin += 1
        self.step_end += 1

        state = self.data[self.step_begin:self.step_end]

        price_del = self.data[self.step_end] - self.data[self.step_end - 1]
        if action == 1:
            reward = price_del
        elif action == 0:
            reward = -price_del
        else:
            raise Exception()

        if self.step_end < self.data.size - 1:
            finished = False
        else:
            finished = True

        return state, reward, finished, {}

    def reset(self):
        self.step_begin = 0
        self.step_end = self.step_begin + self.step_size
        return self.data[self.step_begin:self.step_end]

    def seed(self, seed=None):
        pass

    def render(self, mode='human', close=False):
        if not close:
            plt.plot(self.data)

    def close(self):
        pass

env = TestEnv(100, 10)
env.data = np.load('./weights/keras_rl_random_walk.npy')
env.render()
plt.show()

window_length = 1
x = Input((window_length, ) + env.observation_space.shape)
y = Flatten()(x)
y = Dense(16, activation='elu')(y)
y = Dense(16, activation='elu')(y)
y = Dense(env.action_space.n)(y)
model = Model(x, y)

memory = SequentialMemory(limit=100000, window_length=window_length)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=1000, gamma=.95, batch_size=32,
               enable_dueling_network=True, dueling_type='avg', target_model_update=.1, policy=policy)
dqn.compile(Adam(), metrics=['mae'])

dqn.load_weights('./weights/keras_rl_random_walk.hdf5')

dqn.test(env)
