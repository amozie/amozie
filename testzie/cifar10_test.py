import keras
import numpy as np
from matplotlib import pyplot as plt

from skimage import transform

model = keras.applications.InceptionResNetV2(include_top=False)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


def generate(X, y, batch_size):
    while True:
        choices = np.random.choice(X.shape[0], (batch_size,))
        imgs = []
        for i in choices:
            img = transform.resize(X[i], (299, 299), mode='constant')
            imgs.append(img)
        imgs = np.array(imgs)
        pred = model.predict(imgs)
        yield pred, y[choices]


y_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

model_x = keras.layers.Input((8, 8, 1536))
model_y = keras.layers.AveragePooling2D((8, 8))(model_x)
model_y = keras.layers.Flatten()(model_y)
model_y = keras.layers.Dense(10, activation='softmax')(model_y)
model_new = keras.models.Model(model_x, model_y)

model_new.compile('adam', 'categorical_crossentropy', ['accuracy'])

model_new.fit_generator(generator=generate(X_train, y_one_hot, 50), steps_per_epoch=50, epochs=1)

model_new.evaluate_generator(generator=generate(X_test, y_test_one_hot, 1), steps=50)

model_new.save_weights('./weights/weight_cifar10_test.hdf5')

model_new.save('./weights/model_cifar10_test.hdf5')

model_x = keras.layers.Input((8, 8, 1536))
model_y = keras.layers.GlobalAveragePooling2D()(model_x)
model_y = keras.layers.Dense(10, activation='softmax')(model_y)
model_new_2 = keras.models.Model(model_x, model_y)

model_new_2.compile('adam', 'categorical_crossentropy', ['accuracy'])

model_new_2.fit_generator(generator=generate(X_train, y_one_hot, 50), steps_per_epoch=100, epochs=1)

model_new_2.evaluate_generator(generator=generate(X_test, y_test_one_hot, 1), steps=50)

model_new_2.save_weights('./weights/weight_cifar10_test_2.hdf5')

model_new_2.save('./weights/model_cifar10_test_2.hdf5')
