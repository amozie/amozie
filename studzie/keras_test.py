import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data


def foo1():
    mnist = input_data.read_data_sets('./dataset/', one_hot=True)
    print(mnist.train.images[0])
    print(mnist.train.labels[0])
    plt.imshow(mnist.train.images[0].reshape(28, 28), 'gray')
    plt.show()


def foo2():
    model = Sequential()
    model.add(Dense(20, input_shape=()))


if __name__ == '__main__':
    foo1()
