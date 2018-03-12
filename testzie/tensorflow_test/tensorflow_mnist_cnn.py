import tensorflow as tf
import tensorlayer as tl
import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_ = tf.placeholder(tf.float32, (None, 28, 28))
y_ = tf.placeholder(tf.int32, (None,))

sess = tf.InteractiveSession()

outputs = tf.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(X_)
outputs = tf.layers.MaxPooling2D(()
