import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from prettytable import PrettyTable

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, Input, concatenate
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


def foo(x):
    # return K.argmax(x[:, :-1] * (1 - K.cast(K.greater(K.sum(K.one_hot(K.cast(x[:, -2:], 'int32'), 4), 1), 0.5), 'float32')))
    return K.argmax(x[:, :4] * (1 - K.cast(K.greater(K.sum(K.one_hot(K.cast(x[:, 4:], 'int32'), 4), 1), 0.5), 'float32')))

x1 = Input((4, ), name='input1')
x2 = Input((None, ), name='input2')
y = Lambda(lambda x: K.softmax(x), name='lambda')(x1)
y = concatenate([y, x2])
y = Lambda(foo)(y)
model = Model(inputs=[x1, x2], outputs=y)
model.predict([np.array([[5, 4, 3, 1], [6, 5, 7, 9]]), np.array([[4, 1], [2, 4]])])
