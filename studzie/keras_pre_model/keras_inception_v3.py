import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3

input_tensor = Input(shape=(224, 224, 3))

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

model = InceptionV3(weights='imagenet', include_top=True)

plot_model(model, show_shapes=True)
