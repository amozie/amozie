import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from prettytable import PrettyTable

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras import backend as K


main_input = Input(shape=(100, ), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)