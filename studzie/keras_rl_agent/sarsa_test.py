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


env = gym.make('Acrobot-v0')
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

policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

rewards = []
hist = sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)
rewards.extend(hist.history.get('episode_reward'))
plt.plot(rewards)

sarsa.test(env, nb_episodes=5, visualize=True)

state = env.reset()
action = env.action_space.sample()
print(action)
for i in range(300):
    # action = np.argmax(sarsa.model.predict(np.expand_dims(np.expand_dims(state, 0), 0))[0])
    state, reward, done, _ = env.step(action)
    env.render()
env.render(close=True)