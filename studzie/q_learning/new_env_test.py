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
        if self.n == 0:
            return None
        return np.random.choice(self.l)

    def contains(self, x):
        return x in self.l


class TestEnv(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self) -> None:
        self.__max_step = 9

        # 0-8分别代表此次落子位置
        self.action_space = ListDiscrete(range(9))
        # 下标0为：初始为1，1、2交替变换分别代表黑白两方（1黑方先行），下标1-9为：初始全零，0空、1黑方、2白方
        self.observation_space = spaces.Box(
            np.zeros(10, int), np.zeros(10, int) + 2)
        # 代表AI所属方（1或2）
        self.ai = None
        self._seed()
        self.state = None
        self.__step = None
        # 当前落子方
        self.current = None
        # 已经落子格子
        self.stated = None
        self.__all_stated_set = set(range(9))

    def _seed(self, seed=None):
        self.ai = np.random.choice([1, 2])

    def _reset(self):
        self.__step = 0
        self.seed()
        self.current = 1
        self.state = [self.ai]
        self.state.extend(np.zeros(9, int).tolist())
        self.stated = []
        print(self.state)
        return np.array(self.state)

    def _step(self, action):
        if action is None:
            return np.array(self.state), 0, True, {}
        grid = self.state[1:]
        # 不可重复落子
        if action in self.stated:
            return np.array(self.state), -100, True, {}
        # 黑方先手
        grid = np.array(grid, int)
        dif = np.sum(grid == 1) - np.sum(grid == 2)
        if (dif == 0 and self.current == 1) or (dif == 1 and self.current == 2):
            pass
        else:
            return np.array(self.state), -100, True, {}
        # 胜负
        current = self.current
        grid[action] = current
        self.current = 3 - current
        self.state = [self.ai]
        self.state.extend(grid.tolist())
        self.stated.append(action)
        self.action_space = ListDiscrete(self.__all_stated_set.difference(self.stated))

        is_win = lambda b: \
            b[0:3].all() or b[3:6].all() or b[6:9].all() or \
            b[0::3].all() or b[1::3].all() or b[2::3].all() or \
            b[0::4].all() or b[2:7:2].all()
        if is_win(grid == current):
            reward = 20
            # if self.ai == current:
            #     reward = 20
            # else:
            #     reward = -20
            return np.array(self.state), reward, True, {}

        reward = 0
        self.__step += 1
        if self.__step < self.__max_step:
            done = False
        else:
            done = True
        return np.array(self.state), reward, done, {}

    def _render(self, mode='ansi', close=False):
        table = PrettyTable()
        table.header = False
        grid = self.state[1:]
        table.add_row(grid[0:3])
        table.add_row(grid[3:6])
        table.add_row(grid[6:9])
        print(table)

    def _close(self):
        pass


def foo(x):
    # return K.concatenate((K.zeros((K.ndim(x), 1)), x), 1)
    y1 = K.sum(K.cast(K.equal(x[:, 1:], 1), 'float32'), 1)
    y2 = K.sum(K.cast(K.equal(x[:, 1:], 2), 'float32'), 1)
    y = K.not_equal(y1, y2)

    return K.equal(y1, y2)


if __name__ == '__main__':
    state = np.array([2, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0])
    env = TestEnv()
    env.reset()

    x = Input((1,) + env.observation_space.shape)
    y = Flatten()(x)
    y = Lambda(foo)(y)
    model = Model(x, y)
    model.predict(state.reshape(-1, 1, 10))

# if __name__ == '__main__':
#     env = TestEnv()
#     state = env.reset()
#     nb_actions = env.action_space.n
#
#     state = state.reshape(-1, 1, 10)
#
#     x = Input((1,) + env.observation_space.shape)
#     y = Flatten()(x)
#     y = Lambda(foo)(y)
#     model = Model(x, y)
#     model.predict(state.reshape(-1, 1, 10))
#
#     y = Dense(32)(y)
#     y = Activation('relu')(y)
#     y = Dense(32)(y)
#     y = Activation('relu')(y)
#     y = Dense(32)(y)
#     y = Activation('relu')(y)
#     y = Dense(nb_actions)(y)
#     y = Activation('linear')(y)
#     z = Reshape((10, ))(x)
#     w = concatenate([y, z])
#     w = Lambda(lambda x: x[:, :9] + K.cast(K.greater(x[:, 10:], 0), 'float32')*-100)(w)
#     model = Model(x, w)
#     model.predict(state.reshape(-1, 1, 10))
#
#     memory = SequentialMemory(limit=5000, window_length=1)
#     policy = BoltzmannQPolicy()
#     dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
#                    target_model_update=1e-2, policy=policy)
#     dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#
#     hist = dqn.fit(env, nb_steps=6000, visualize=False, verbose=0, nb_max_start_steps=1000, nb_max_episode_steps=9)
#
#     plt.plot(hist.history.get('episode_reward'))
#
#     state = env.reset()
#     state = np.array([2,0,0,0,0,1,2,1,0,0, 1,1,0,0,0,2,2,1,0,0])
#     model.predict(state.reshape(2, 1, 10))
#     action = np.argmax(model.predict(state.reshape(1, 1, 10))[0])
#     state, reward, done, _ = env.step(action)
#     print(action, done)
#     env.render()