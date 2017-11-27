import numpy as np
import matplotlib.pyplot as plt

import gym
import time
from prettytable import PrettyTable
import copy
import tushare as ts

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, Input, Reshape, concatenate, \
    Merge, LSTM, Embedding
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

t = np.linspace(0, 25, 2500)
f = np.sin(t)
f = np.cumsum(np.random.choice([-1, 0, 1], 2000))

valid_spilit = 0.2
window_size = 60

length = f.size
spilit = int(length * (1 - valid_spilit))

t = np.arange(length)

t_train = t[:spilit][window_size:]
t_test = t[spilit:]

f_train = f[:spilit]
f_test = f[spilit - window_size:]

x_train = [f_train[i:i+window_size] for i in range(f_train.shape[0]-window_size)]
x_train = np.array(x_train)
x_train = np.expand_dims(x_train, -1)
y_train = f_train[window_size:]

x_test = [f_test[i:i+window_size] for i in range(f_test.shape[0]-window_size)]
x_test = np.array(x_test)
x_test = np.expand_dims(x_test, -1)
y_test = f_test[window_size:]

x = Input((window_size, 1))
y = LSTM(32, return_sequences=True)(x)
y = LSTM(32, return_sequences=True)(y)
y = LSTM(32)(y)
y = Dense(1)(y)
y = Activation('linear')(y)
model = Model(x, y)
model.summary()

x = Input((window_size, 1), batch_shape=(32, window_size, 1))
y = LSTM(16, return_sequences=True, stateful=True)(x)
y = LSTM(16, return_sequences=True, stateful=True)(y)
y = LSTM(16, stateful=True)(y)
y = Dense(1)(y)
y = Activation('linear')(y)
model = Model(x, y)
model.summary()

model.compile('adam', 'mse')
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

model.evaluate(x_test, y_test)

#######################
y_train_pred = model.predict(x_train).flatten()
y_test_pred = model.predict(x_test).flatten()

plt.plot(t_train, y_train, label='train')
plt.plot(t_train, y_train_pred, label='train_pred')
plt.plot(t_test, y_test, label='test')
plt.plot(t_test, y_test_pred, label='test_pred')
plt.legend()
########################
x_cur = x_train[0:1]
y_pred = model.predict(x_train[0:1])
