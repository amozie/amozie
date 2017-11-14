from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, GlobalAveragePooling2D, \
    Flatten, RepeatVector, Permute, Reshape, GlobalMaxPooling2D
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from skimage import transform, color
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 28, 28).transpose(0, 2, 1, 3).reshape(28*3, 28*3), 'gray')

vgg16 = VGG16(weights='imagenet', include_top=False)
y = vgg16.output
y = GlobalMaxPooling2D()(y)
# y = Flatten()(y)
y = Dense(256)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(10)(y)
y = Activation('softmax')(y)
model = Model(vgg16.input, y)
model.summary()

img = X_train[0]
img = transform.resize(img, (224, 224))
img = color.gray2rgb(img)
plt.imshow(img)

'''
def img_generator(images, labels, batch_size):
    while True:
        batch_i_arr = np.random.choice(images.shape[0], batch_size)
        image_batch = images[batch_i_arr]
        label_batch = labels[batch_i_arr]
        image_batch = image_batch.reshape((-1, 28, 28))
        image_batch = image_batch.transpose(1, 2, 0)
        image_batch = transform.resize(image_batch, (224, 224))
        image_batch = image_batch.transpose(2, 0, 1)
        image_batch = color.gray2rgb(image_batch)
        yield image_batch, label_batch
'''


def img_generator_one(images, labels):
    while True:
        i = np.random.choice(images.shape[0])
        image = images[i]
        label = labels[i]
        image = transform.resize(image, (224, 224))
        image = color.gray2rgb(image)
        yield np.expand_dims(image, 0), np.expand_dims(label, 0)

for layer in vgg16.layers:
    layer.trainable = False

for layer in vgg16.layers[-4:]:
    layer.trainable = True

for layer in model.layers:
    print(layer.trainable, end=' ')

model.compile(Adam(1e-3), 'categorical_crossentropy', ['accuracy'])

model.compile(Adam(1e-4), 'categorical_crossentropy', ['accuracy'])

hist = model.fit_generator(img_generator_one(X_train, y_train), 32, 10, verbose=2,
                           validation_data=img_generator_one(X_test, y_test),
                           validation_steps=32)

model.evaluate_generator(img_generator_one(X_test, y_test), 32)

plot_model(vgg16, to_file='model_0.png', show_shapes=True)
