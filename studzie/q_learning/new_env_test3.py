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
        self.epsilon = 1.0
        self.epsilon_min = 0.01

        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.nb_actions = self.env.action_space.n

        self.model = None
        self.trainable_model = None
        self.memory = None
        self.target_list = None

        self.epsilon_decay = None
        self.memory_limit = None
        self.batch_size = None

    def _init_model(self):
        model = Sequential()
        model.add(Dense(16, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        print(model.summary())
        self.model = model
        self.model.compile(optimizer='sgd', loss='mse')

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = .5 * K.square(y_true - y_pred)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        y_pred = self.model.output
        y_true = Input((self.nb_actions, ))
        mask = Input((self.nb_actions, ))
        loss = Lambda(clipped_masked_error, (1, ))([y_pred, y_true, mask])
        train_model = Model([self.model.input, y_true, mask], [loss, y_pred])
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        self.train_model = train_model
        self.train_model.compile(Adam(), loss=losses)

    def compile(self):
        self._init_model()
        self.memory = []
        self.target_list = []

    def fit(self, steps, memory_limit, max_episode_steps=None, batch_size=32, decay_episodes=100):
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / decay_episodes
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
                q_values = self.model.predict(np.array([state.reshape(self.input_shape)]))[0]
                q_values = q_values.astype('float64')
                exp_values = np.exp(np.clip(q_values, -500.0, 500.0))
                probs = exp_values / np.sum(exp_values)
                action = np.random.choice(len(q_values), p=probs)
                # GreedyQPolicy
                # if np.random.uniform() < self.epsilon:
                #     action = self.env.action_space.sample()
                # else:
                #     action = np.argmax(self.model.predict(state.reshape(self.input_reshape))[0])
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
        state_batch = []
        target_batch = []
        mask_batch = []
        dummy_targets = []
        for k in batches:
            state_k, action_k, reward_k, next_state_k, done_k = self.memory[k]
            if done_k:
                dummy_target = reward_k
            else:
                dummy_target = reward_k + self.gamma * np.amax(
                    self.model.predict(np.array([next_state_k.reshape(self.input_shape)]))[0])
            target = self.model.predict(np.array([state_k.reshape(self.input_shape)]))[0]
            target = np.zeros_like(target, 'float32')
            target[action_k] = dummy_target
            mask = np.zeros_like(target, 'float32')
            mask[action_k] = 1.0
            # self.model.train_on_batch(np.array([state_k.reshape(self.input_shape)]), np.array([target]))
            state_batch.append(state_k)
            target_batch.append(target)
            mask_batch.append(mask)
            dummy_targets.append(dummy_target)
        state_batch = np.array(state_batch)
        target_batch = np.array(target_batch)
        mask_batch = np.array(mask_batch)
        dummy_targets = np.array(dummy_targets)
        self.train_model.train_on_batch([state_batch, target_batch, mask_batch],
                                        [dummy_targets, target_batch])
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

if __name__ == '__main__':
    ENV_NAME = 'CartPole-v0'
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    dqn = DQNAgentZZ(env)
    dqn.compile()
    dqn.fit(50000, 50000)
    plt.plot(dqn.target_list)

    state = env.reset()
    for i in range(500):
        action = np.argmax(dqn.model.predict(state.reshape(1, 4))[0])
        state, reward, done, _ = env.step(action)
        env.render()
    env.render(close=True)