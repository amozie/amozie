import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten, Reshape, \
    Conv2D, MaxPooling2D, concatenate, GlobalMaxPooling2D
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from skimage import transform, color

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
