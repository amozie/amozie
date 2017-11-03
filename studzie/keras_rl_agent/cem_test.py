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
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.agents.cem import CEMAgent
from rl.agents import SARSAAgent


env = gym.make('CartPole-v0')
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

memory = EpisodeParameterMemory(limit=50000, window_length=1)
cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=2000, batch_size=50, train_interval=50, elite_frac=0.05)
cem.compile()

rewards = []
hist = cem.fit(env, nb_steps=50000, visualize=False, verbose=2)
rewards.extend(hist.history.get('episode_reward'))
plt.plot(rewards)

cem.test(env, nb_episodes=5, visualize=True)

state = env.reset()
action = env.action_space.sample()
print(action)
for i in range(500):
    action = np.argmax(cem.model.predict(state.reshape(1, 1, 6))[0])
    state, reward, done, _ = env.step(action)
    env.render()
env.render(close=True)
