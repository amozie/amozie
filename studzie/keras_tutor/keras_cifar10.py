import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.datasets import cifar10
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam

from skimage import transform, color
from keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 32, 32, 3).swapaxes(1, 2).reshape(32*3, 32*3, 3))

# vgg like: loss: 0.5700 - acc: 0.8007 - val_loss: 0.7638 - val_acc: 0.7455
x = Input((32, 32, 3))
y = Conv2D(16, 2, padding='same', activation='relu')(x)
y = Conv2D(16, 2, padding='same', activation='relu')(y)
y = MaxPool2D()(y)
y = Conv2D(64, 2, padding='same', activation='relu')(y)
y = Conv2D(64, 2, padding='same', activation='relu')(y)
y = MaxPool2D()(y)
y = Conv2D(256, 2, padding='same', activation='relu')(y)
y = Conv2D(256, 2, padding='same', activation='relu')(y)
y = MaxPool2D()(y)
y = GlobalMaxPool2D()(y)
y = Dense(10, activation='softmax')(y)
model = Model(x, y)
model.summary()

model.compile(Adam(1e-3), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
plt.plot()


# googleNet like: loss: 0.5194 - acc: 0.8233 - val_loss: 0.8489 - val_acc: 0.7148
def lenet_top(x):
    y = Conv2D(16, 2, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPool2D()(y)
    y = Conv2D(32, 2, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPool2D()(y)
    return y


def lenet(filter, x):
    y1 = Conv2D(filter, 1)(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y2 = AveragePooling2D(1)(x)
    y2 = Conv2D(filter, 1)(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y3 = Conv2D(filter, 1)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(filter, 2, padding='same')(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)

    y4 = Conv2D(filter, 1)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = Conv2D(filter, 2, padding='same')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = Conv2D(filter, 2, padding='same')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)

    y = concatenate([y1, y2, y3, y4])
    return y

x = Input((32, 32, 3))
y = lenet_top(x)
y = lenet(16, y)
y = lenet(32, y)
y = lenet(64, y)
y = GlobalMaxPool2D()(y)
y = Dense(10, activation='softmax')(y)
model = Model(x, y)
model.summary()

model.compile(Adam(1e-3), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=4, validation_data=(X_test, y_test), verbose=1)
plt.plot()


# resNet like: loss: 0.5618 - acc: 0.8003 - val_loss: 0.9090 - val_acc: 0.6994
def resnet_top(filter, x):
    y = Conv2D(filter, 3, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPool2D()(y)
    return y


def resnet(filter, x):
    z = BatchNormalization()(x)
    z = Activation('relu')(z)
    z = Conv2D(filter, 2, strides=(2, 2))(z)
    ####
    y = BatchNormalization()(z)
    y = Activation('relu')(y)
    y = Conv2D(filter//4, 1)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filter//4, 3, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filter, 1)(y)
    y = add([y, z])
    return y

x = Input((32, 32, 3))
y = resnet_top(32, x)
y = resnet(32, y)
y = resnet(64, y)
y = resnet(128, y)
y = resnet(256, y)
y = GlobalAvgPool2D()(y)
y = Dense(10, activation='softmax')(y)

model = Model(x, y)
model.summary()

model.compile(Adam(1e-3, amsgrad=True), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
plt.plot()
##########################
plot_model(model, show_shapes=True)
