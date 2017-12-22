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
y = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
y = MaxPooling2D((2, 2), (2, 2))(y)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(y)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(y)
y = MaxPooling2D((2, 2), (2, 2))(y)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(y)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(y)
y = MaxPooling2D((2, 2), (2, 2))(y)
y = Flatten()(y)
y = Dense(1024, activation='relu')(y)
y = Dense(100, activation='relu')(y)
y = Dense(10, activation='softmax')(y)
model = Model(x, y)
model.summary()

model.compile(Adam(1e-4), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=4, validation_data=(X_test, y_test), verbose=1)
plt.plot()


# googleNet like:
def lenet_top(x):
    y = Conv2D(32, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D((3, 3), 2, padding='same')(y)
    y = Conv2D(64, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(128, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D((3, 3), 2, padding='same')(y)
    return y


def lenet(filter, x):
    y1 = Conv2D(filter, (1, 1))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y2 = AveragePooling2D((1, 1))(x)
    y2 = Conv2D(filter, (1, 1))(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y3 = Conv2D(filter, (1, 1))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(filter, (3, 3), padding='same')(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)

    y4 = Conv2D(filter, (1, 1))(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = Conv2D(filter, (3, 3), padding='same')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    y4 = Conv2D(filter, (3, 3), padding='same')(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)

    y = concatenate([y1, y2, y3, y4])
    return y

x = Input((32, 32, 3))
y = lenet_top(x)
y = lenet(64, y)
y = lenet(128, y)
y = lenet(256, y)
y = GlobalMaxPool2D()(y)
y = Dense(256, activation='relu')(y)
y = Dense(10, activation='softmax')(y)
model = Model(x, y)
model.summary()

model.compile(Adam(1e-4), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test), verbose=1)
plt.plot()

# resNet like:
##########################
plot_model(model, show_shapes=True)
