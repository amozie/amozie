import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam

from keras.applications.resnet50 import ResNet50

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 28, 28).transpose(0, 2, 1, 3).reshape(28*3, 28*3), 'gray')

model = ResNet50(weights='imagenet', include_top=False)
