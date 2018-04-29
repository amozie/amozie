import keras
from keras.layers import *
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

x = Input((28, 28))
y = Reshape((28, 28, 1))(x)
y = Conv2D(32, (5, 5), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(32, (5, 5), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
# y = Flatten()(x)
# y = Dense(16, activation='relu')(y)
# y = Dense(16, activation='relu')(y)
# y = Dense(16, activation='relu')(y)
# y = Dense(16, activation='relu')(y)
y = Flatten()(y)
y = Dense(28*28)(y)
y = Reshape((28, 28))(y)
model = keras.Model(x, y)

model.summary()

model.compile('adam', 'mse', ['accuracy'])

model.fit(X_train, X_train, 50, 4)

pred = model.predict(X_test[:32])
pred = pred.astype(int)
pred = np.clip(pred, 0, 255)

for i in range(32):
    ax = plt.subplot(8, 8, i*2+1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.imshow(X_test[i])

    ax = plt.subplot(8, 8, i*2+2)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.imshow(pred[i])
