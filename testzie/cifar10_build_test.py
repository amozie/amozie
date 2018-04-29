import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import *
from sklearn import preprocessing

model = keras.applications.ResNet50()
model = keras.applications.DenseNet121()

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

y_one = keras.utils.to_categorical(y_train, 10)
y_test_one = keras.utils.to_categorical(y_test, 10)

X_mean = np.mean(X_train, (0, 1, 2)).reshape(1, 1, 1, 3)
X_fix = X_train - X_mean
X_test_fix = X_test - X_mean

def residual_layer(x, n):
    y1 = BatchNormalization()(x)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(n, (3, 3), padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(n, (3, 3), padding='same')(y1)

    y2 = BatchNormalization()(x)
    y2 = Conv2D(n, (3, 3), padding='same')(y2)

    y = keras.layers.add([y1, y2])
    return y


x = Input((32, 32, 3))
y = BatchNormalization()(x)
y = Conv2D(64, (3, 3), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = residual_layer(y, 64)
y = residual_layer(y, 64)
y = MaxPooling2D()(y)
y = residual_layer(y, 128)
y = residual_layer(y, 128)
y = GlobalMaxPooling2D()(y)
y = Dense(10, activation='softmax')(y)
model = keras.Model(x, y)

model.summary()

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

model.fit(X_train, y_one, 50, 2)

model.evaluate(X_test, y_test_one, 50)