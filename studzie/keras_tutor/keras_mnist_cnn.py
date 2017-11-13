import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam


(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 28, 28).transpose(0, 2, 1, 3).reshape(28*3, 28*3), 'gray')

x = Input((28, 28))
y = Reshape((28, 28, 1))(x)
y = Conv2D(32, (5, 5))(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(64, (5, 5))(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = Flatten()(y)
y = Dense(256)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(10)(y)
y = Activation('softmax')(y)
model = Model(x, y)
model.summary()

model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)
