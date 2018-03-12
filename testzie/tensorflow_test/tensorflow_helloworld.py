from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorlayer as tl
import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_ = tf.placeholder(tf.float32, (None, 28, 28))
y_ = tf.placeholder(tf.int32, (None,))

sess = tf.InteractiveSession()

outputs = tf.layers.Flatten()(X_)
outputs = tf.layers.Dense(16, activation=tf.nn.relu)(outputs)
outputs = tf.layers.Dense(16, activation=tf.nn.relu)(outputs)
outputs = tf.layers.Dense(10)(outputs)
y_one_hot = tf.one_hot(y_, 10)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=outputs)
loss = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer().minimize(loss)
acc = tf.equal(tf.argmax(outputs, 1, output_type=tf.int32), y_)
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

sess.run(tf.global_variables_initializer())

sess.run(outputs, {X_:X_train, y_:y_train})
sess.run(acc, {X_:X_train, y_:y_train})

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
it = dataset.shuffle(X_train.shape[0]).batch(50).repeat(20).make_one_shot_iterator()
element = it.get_next()

max_steps = (X_train.shape[0] - 1) // 50 + 1

step = 0
epoch = 0
losses_train = []
losses_test = []
try:
    while True:
        batch = sess.run(element)
        sess.run(opt, {X_: batch[0], y_: batch[1]})
        step += 1
        # print(step)
        if step == max_steps:
            step = 0
            epoch += 1
            print(epoch)
            loss_train = sess.run(loss, {X_:X_train, y_:y_train})
            acc_train = sess.run(acc, {X_:X_train, y_:y_train})
            loss_test = sess.run(loss, {X_: X_test, y_: y_test})
            acc_test = sess.run(acc, {X_: X_test, y_: y_test})

            losses_train.append(losses_train)
            losses_test.append(losses_test)
except tf.errors.OutOfRangeError:
    pass

dataset = tf.data.Dataset.from_tensor_slices((list(range(0, 100)), list(range(100, 200))))
it = dataset.shuffle(100).batch(3).repeat(2).make_one_shot_iterator()

sess.run(it.get_next())

keras.Model().evaluate()

tl.utils.evaluation()