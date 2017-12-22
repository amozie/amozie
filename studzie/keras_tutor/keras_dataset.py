from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import imdb
from keras.datasets import reuters
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import boston_housing
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

(X_train, y_train), (X_test, y_test) = imdb.load_data()

(X_train, y_train), (X_test, y_test) = reuters.load_data()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
