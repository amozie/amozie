from keras.applications import ResNet50
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.datasets import cifar10
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD
import cv2

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 32, 32, 3).swapaxes(1, 2).reshape(32*3, 32*3, 3))

base_model = ResNet50(weights='imagenet', include_top=False, pooling=None, input_shape=(197, 197, 3), classes=10)

for layer in base_model.layers:
    layer.trainable = False

y = base_model.output
y = Flatten()(y)
y = Dense(10, activation='softmax')(y)

model = Model(base_model.input, y)
sgd = SGD(lr=0.003, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

def img_generator_one(images, labels):
    while True:
        i = np.random.choice(images.shape[0])
        image = images[i]
        label = labels[i]
        image = cv2.resize(image, (197, 197))
        yield np.expand_dims(image, 0), np.expand_dims(label, 0)

history = model.fit_generator(generator=img_generator_one(X_train, y_train),
                              steps_per_epoch=128,
                              epochs=10,
                              verbose=1,
                              validation_data=img_generator_one(X_test, y_test),
                              validation_steps=32)