import numpy as np
import matplotlib.pyplot as plt

import gym
import time
from prettytable import PrettyTable
import copy

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

