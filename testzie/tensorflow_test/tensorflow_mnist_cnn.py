import tensorflow as tf
import tensorlayer as tl
import keras
import time
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

y_one_hot = tf.one_hot(y_train, 10).eval()
y_test_one_hot = tf.one_hot(y_test, 10).eval()

X_ = tf.placeholder(tf.float32, (None, 28, 28))
y_ = tf.placeholder(tf.int32, (None, 10))

sess = tf.InteractiveSession()

init = tf.contrib.layers.xavier_initializer()

outputs = tf.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(X_[:, :, :, tf.newaxis])
outputs = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(outputs)
outputs = tf.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(outputs)
outputs = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(outputs)
outputs = tf.layers.Flatten()(outputs)
outputs = tf.layers.Dense(10)(outputs)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=outputs)
loss = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer().minimize(loss)
acc = tf.equal(tf.argmax(outputs, 1, output_type=tf.int32), tf.argmax(y_, 1, output_type=tf.int32))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

sess.run(tf.global_variables_initializer())

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_one_hot))
make_one = dataset.shuffle(X_train.shape[0]).batch(50).repeat(4).make_one_shot_iterator()
element = make_one.get_next()

max_steps = (X_train.shape[0] - 1) // 50 + 1

epoch = 0
step = 0
begin_time = time.time()
try:
    while True:
        batch = sess.run(element)
        sess.run(opt, {X_:batch[0], y_:batch[1]})
        # step += 1
        # if step == max_steps:
        #     step = 0
        #     epoch += 1
        #     train_loss = sess.run(loss, {X_:batch[0], y_:batch[1]})
        #     print(f'epoch: {epoch}, loss: {train_loss}')
except tf.errors.OutOfRangeError:
    pass
end_time = time.time()
print(end_time - begin_time)
####

model_x = keras.layers.Input((28, 28))
model_y = keras.layers.Reshape((28, 28, 1))(model_x)
model_y = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(model_y)
model_y = keras.layers.MaxPooling2D((2, 2), padding='same')(model_y)
model_y = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(model_y)
model_y = keras.layers.MaxPooling2D((2, 2), padding='same')(model_y)
model_y = keras.layers.Flatten()(model_y)
model_y = keras.layers.Dense(10, activation='softmax')(model_y)
model = keras.models.Model(model_x, model_y)
keras.models.Model(model_x, model_y)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

begin_time = time.time()
model.fit(X_train, y_one_hot, batch_size=50, epochs=4, verbose=2)
end_time = time.time()
print(end_time - begin_time)

model.evaluate(X_test, y_test_one_hot)
y_pred = model.predict(X_test)
np.mean(np.equal(np.argmax(y_pred, 1), y_test))