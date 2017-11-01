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


env = gym.make('CartPole-v1')
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
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

rewards = []
hist = dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)
rewards.extend(hist.history.get('episode_reward'))
plt.plot(rewards)

dqn.test(env, nb_episodes=5, visualize=False)

state = env.reset()
for i in range(500):
    # action = np.argmax(dqn.model.predict(state.reshape(1, 1, 4))[0])
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()
env.render(close=True)
