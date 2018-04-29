import keras
from keras.layers import *
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

x = Input((32, 32, 3))
y = Conv2D(32, (5, 5), padding='same')(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = AveragePooling2D()(y)
y = Conv2D(32, (5, 5), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = AveragePooling2D()(y)
y = UpSampling2D()(y)
y = Conv2DTranspose(32, (5, 5), padding='same')(y)
y = UpSampling2D()(y)
y = Conv2DTranspose(3, (5, 5), padding='same')(y)
# y = Flatten()(y)
# y = Dense(32*32*3)(y)
# y = Reshape((32, 32, 3))(y)
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
