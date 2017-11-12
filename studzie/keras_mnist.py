import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
from keras.callbacks import EarlyStopping


mnist = input_data.read_data_sets('./dataset/', one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
images_test = mnist.test.images
lables_test = mnist.test.labels
plt.imshow(images[0:9].reshape(28*9, 28), 'gray')

x = Input((784, ))
y = Dense(16)(x)
y = Activation('relu')(y)
y = Dense(16)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(10)(y)
y = Activation('softmax')(y)
model = Model(x, y)
model.summary()

model.compile('adam', 'mse', ['accuracy'])
early_stopping = EarlyStopping(patience=2)
hist = model.fit(images, labels, validation_split=0.1, batch_size=32, epochs=10, verbose=2, callbacks=[early_stopping])
hist = model.fit(images, labels, batch_size=32, epochs=20, verbose=2)

model.evaluate(images_test, lables_test)
