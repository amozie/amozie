import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten, GlobalMaxPooling2D
from keras.datasets import mnist, cifar10
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from skimage import transform, color

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 32, 32, 3).swapaxes(1, 2).reshape(32*3, 32*3, 3))

base_model = InceptionV3(weights='imagenet', include_top=False)
y = GlobalMaxPooling2D()(base_model.output)
y = Dense(1024)(y)
y = Activation('relu')(y)
y = Dense(10)(y)
y = Activation('softmax')(y)
model = Model(base_model.input, y)

for layer in base_model.layers:
    layer.trainable = False

model.compile(RMSprop(), 'categorical_crossentropy', ['accuracy'])

for layer in base_model.layers[172:]:
    layer.trainable = True

model.compile(SGD(1e-4), 'categorical_crossentropy', ['accuracy'])


def img_generator_one(images, labels):
    while True:
        i = np.random.choice(images.shape[0])
        image = images[i]
        label = labels[i]
        image = transform.resize(image, (299, 299))
        # image = color.gray2rgb(image)
        yield np.expand_dims(image, 0), np.expand_dims(label, 0)

hist = model.fit_generator(img_generator_one(X_train, y_train), 32, 3, verbose=2,
                           validation_data=img_generator_one(X_test, y_test),
                           validation_steps=32)
plt.plot()
