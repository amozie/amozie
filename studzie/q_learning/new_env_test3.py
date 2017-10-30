import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from prettytable import PrettyTable
import copy

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, Input, Reshape, concatenate
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf

from gym import Env, Space, spaces
from gym.utils import seeding
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


class DQNAgentZZ:
    def __init__(self, env) -> None:
        self.gamma = 0.99
        self.epsilon_greedy = 0.1

        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.input_reshape = (1, ) + self.input_shape
        self.nb_actions = self.env.action_space.n

        self.model = None
        self.memory = None
        self.target_list = None

        self.memory_limit = None
        self.batch_size = None

    def _init_model(self):
        model = Sequential()
        model.add(Dense(16, input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        print(model.summary())
        self.model = model

    def compile(self):
        self._init_model()
        self.model.compile(Adam(), 'mse')
        self.memory = []
        self.target_list = []

    def fit(self, steps, memory_limit, max_episode_steps=None, batch_size=32):
        self.memory_limit = memory_limit
        self.batch_size = batch_size
        episode = 0
        step_sum = 0
        while True:
            episode += 1
            state = self.env.reset()
            target = 0
            step = 0
            while True:
                step += 1
                if np.random.uniform() < self.epsilon_greedy:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(state.reshape(self.input_reshape))[0])
                next_state, reward, done, _ = self.env.step(action)
                if len(self.memory) < self.memory_limit:
                    pass
                else:
                    self.memory = self.memory[-self.memory_limit:]
                self.memory.append((state, action, reward, next_state, done))
                state = copy.deepcopy(next_state)

                target += reward
                step_sum += 1
                if done:
                    self.target_list.append(target)
                    print('steps:{0} episode:{1} step:{2} target{3}'.format(step_sum, episode, step, target))
                    break
                if step_sum >= steps:
                    break
                if max_episode_steps is not None and step >= max_episode_steps:
                    break
            self._replay()
            if step_sum >= steps:
                break

    def _replay(self):
        batches_size = min(self.batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches_size)
        for k in batches:
            state_k, action_k, reward_k, next_state_k, done_k = self.memory[k]
            if done_k:
                target = reward_k
            else:
                target = reward_k + self.gamma * np.amax(
                    self.model.predict(next_state_k.reshape(self.input_reshape))[0])
            target_f = self.model.predict(state_k.reshape(self.input_reshape))
            target_f[0][action_k] = target
            self.model.fit(state_k.reshape(self.input_reshape), target_f, epochs=1, verbose=0)

if __name__ == '__main__':
    ENV_NAME = 'CartPole-v0'
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    dqn = DQNAgentZZ(env)
    dqn.compile()
    dqn.fit(50000, 100000)
    plt.plot(dqn.target_list)

    state = env.reset()
    for i in range(200):
        action = np.argmax(dqn.model.predict(state.reshape(1, 4))[0])
        state, reward, done, _ = env.step(action)
        env.render()
    env.render(close=True)