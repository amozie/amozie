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
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class ListDiscrete(Space):
    def __init__(self, l) -> None:
        self.l = list(set(l))
        self.n = len(self.l)

    def sample(self):
        if len(self.l) == 0:
            return None
        return np.random.choice(self.l)

    def contains(self, x):
        return x in self.l

    def get_rid_of(self, x):
        if x in self.l:
            self.l.remove(x)


class TestEnv(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self) -> None:
        self.__max_step = 9

        # 0-8分别代表此次落子位置
        self.action_space = ListDiscrete(range(9))
        # 下标0为：初始为1，1、2交替变换分别代表黑白两方（1黑方先行），下标1-9为：初始全零，0空、1黑方、2白方
        self.observation_space = spaces.Box(
            np.zeros(9, int), np.zeros(9, int) + 2)
        # 代表AI所属方（1或2）
        self.ai = None
        self._seed()
        self.state = None
        self.__step = None

    def _seed(self, seed=None):
        self.ai = np.random.choice([1, 2])
        self.op = 3 - self.ai

    def _reset(self):
        self.__step = 0
        self.seed()
        self.action_space = ListDiscrete(range(9))
        self.state = list(np.zeros(9, int))
        if self.ai == 2:
            op_action = self.action_space.sample()
            self.action_space.get_rid_of(op_action)
            self.state[op_action] = self.op
        print(self.state)
        return np.array(self.state)

    def _step(self, action):
        # 不可重复落子
        if action is None:
            return np.array(self.state), 0, True, {}
        if self.state[action] != 0:
            return np.array(self.state), -100, True, {}

        self.state[action] = self.ai
        self.action_space.get_rid_of(action)
        self.__step += 1

        # 胜负
        is_win = lambda b: \
            b[0:3].all() or b[3:6].all() or b[6:9].all() or \
            b[0::3].all() or b[1::3].all() or b[2::3].all() or \
            b[0::4].all() or b[2:7:2].all()

        if is_win(np.array(self.state) == self.ai):
            return np.array(self.state), 20, True, {}

        if self.__step < self.__max_step:
            op_action = self.action_space.sample()
            self.action_space.get_rid_of(op_action)
            self.state[op_action] = self.op
            self.__step += 1

            if is_win(np.array(self.state) == self.op):
                return np.array(self.state), -20, True, {}

            if self.__step < self.__max_step:
                done = False
            else:
                done = True
        else:
            done = True
        return np.array(self.state), 0, done, {}

    def _render(self, mode='ansi', close=False):
        table = PrettyTable()
        table.header = False
        grid = self.state[:]
        table.add_row(grid[0:3])
        table.add_row(grid[3:6])
        table.add_row(grid[6:9])
        print(table)

    def _close(self):
        pass


if __name__ == '__main__':
    env = TestEnv()
    state = env.reset()
    nb_actions = env.action_space.n

    x = Input((1,) + env.observation_space.shape)
    y = Flatten()(x)
    y = Dense(32)(y)
    y = Activation('relu')(y)
    y = Dense(32)(y)
    y = Activation('relu')(y)
    y = Dense(32)(y)
    y = Activation('relu')(y)
    y = Dense(nb_actions)(y)
    y = Activation('linear')(y)
    # z = Reshape((10, ))(x)
    # w = concatenate([y, z])
    # w = Lambda(lambda x: x[:, :9] + K.cast(K.greater(x[:, 10:], 0), 'float32')*-100)(w)
    model = Model(x, y)

    memory = SequentialMemory(limit=15000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    hist = dqn.fit(env, nb_steps=16000, visualize=False, verbose=2, nb_max_start_steps=1000, nb_max_episode_steps=9)

    plt.plot(hist.history.get('episode_reward'))

    state = env.reset()
    action = np.argmax(model.predict(state.reshape(1, 1, 9))[0])
    print(action)
    state, reward, done, _ = env.step(action)
    print(action, reward, done)
    env.render()