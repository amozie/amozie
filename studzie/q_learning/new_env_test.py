import numpy as np
import gym
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from gym import Env, spaces
from gym.utils import seeding
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class TestEnv(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self) -> None:
        self.__max_step = 10

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0]), np.array([1]))

        self._seed()
        self.state = None
        self.__step = None
        self.state_list = None

    def _seed(self, seed=None):
        pass

    def _step(self, action):
        state = self.state
        if state != action:
            reward = 1
        else:
            reward = 0
        self.state = action
        self.state_list.append(action)
        self.__step += 1
        if self.__step < self.__max_step:
            done = False
        else:
            done = True
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.__step = 0
        self.state = np.random.randint(2)
        self.state_list = []
        self.state_list.append(self.state)
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        super()._close()


if __name__ == '__main__':
    env = TestEnv()
    env.reset()
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Dense(16, input_dim=1))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=500, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=500, visualize=True, verbose=2)

