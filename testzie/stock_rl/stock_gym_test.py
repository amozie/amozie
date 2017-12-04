from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, GlobalAveragePooling2D, \
    Flatten, RepeatVector, Permute, Reshape, GlobalMaxPooling2D
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from skimage import transform, color
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical

from gym import Env, Space, spaces
from gym.utils import seeding
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class StockEnv(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, series) -> None:
        assert series.ndim == 2
        assert series.shape[0] == 1
        self.series = series
        self.action_space = spaces.Discrete(2)

    def _render(self, mode='human', close=False):
        super()._render(mode, close)

    def _step(self, action):
        pass

    def _reset(self):
        self.current_i = 0
        return self.series[self.current_i]

    def _close(self):
        super()._close()

    def _seed(self, seed=None):
        return super()._seed(seed)

