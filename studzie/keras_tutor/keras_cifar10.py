import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten, Reshape, \
    Conv2D, MaxPooling2D, concatenate, GlobalMaxPooling2D, BatchNormalization, AveragePooling2D
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from skimage import transform, color
from keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
plt.imshow(X_train[0:9].reshape(3, 3, 32, 32, 3).swapaxes(1, 2).reshape(32*3, 32*3, 3))

x = Input((32, 32, 3))
y = Conv2D(32, (3, 3), padding='same')(x)
y = Activation('relu')(y)
y = Conv2D(32, (3, 3), padding='same')(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = Activation('relu')(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = Activation('relu')(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = Activation('relu')(y)
y = MaxPooling2D()(y)
y = Flatten()(y)
y = Dense(256)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(10)(y)
y = Activation('softmax')(y)
model = Model(x, y)
model.summary()

model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])

hist = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=1)
plt.plot()

##########################

x = Input((32, 32, 3))
y1 = Conv2D(16, (1, 1), padding='same')(x)
y1 = Activation('relu')(y1)
y1 = Conv2D(16, (3, 3), padding='same')(y1)
y1 = Activation('relu')(y1)

y2 = Conv2D(16, (1, 1), padding='same')(x)
y2 = Activation('relu')(y2)
y2 = Conv2D(16, (5, 5), padding='same')(y2)
y2 = Activation('relu')(y2)

y3 = MaxPooling2D(strides=(1, 1), padding='same')(x)
y3 = Conv2D(16, (1, 1), padding='same')(y3)
y3 = Activation('relu')(y3)
y = concatenate([y1, y2, y3], axis=1)

y = Conv2D(16, (1, 1), padding='same')(y)
y = Activation('relu')(y)
y = Conv2D(16, (5, 5), padding='same')(y)
y = Activation('relu')(y)

y = MaxPooling2D()(y)
y = Flatten()(y)
y = Dense(2048)(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
y = Dense(10)(y)
y = Activation('softmax')(y)

model = Model(x, y)
model.summary()

#########################

x = Input((32, 32, 3))
y = Conv2D(32, (3, 3), padding='same')(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(32, (3, 3), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = MaxPooling2D((3, 3), 2, padding='same')(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = MaxPooling2D((3, 3), 2, padding='same')(y)

z1 = Conv2D(24, (1, 1))(y)
z1 = BatchNormalization()(z1)
z1 = Activation('relu')(z1)

z2 = AveragePooling2D((1, 1))(y)
z2 = Conv2D(24, (1, 1))(z2)
z2 = BatchNormalization()(z2)
z2 = Activation('relu')(z2)

z3 = Conv2D(24, (1, 1))(y)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)
z3 = Conv2D(24, (3, 3), padding='same')(z3)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)

z4 = Conv2D(24, (1, 1))(y)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)
z4 = Conv2D(24, (3, 3), padding='same')(z4)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)
z4 = Conv2D(24, (3, 3), padding='same')(z4)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)

y = concatenate([z1, z2, z3, z4])

z1 = Conv2D(24, (1, 1))(y)
z1 = BatchNormalization()(z1)
z1 = Activation('relu')(z1)

z2 = AveragePooling2D((1, 1))(y)
z2 = Conv2D(24, (1, 1))(z2)
z2 = BatchNormalization()(z2)
z2 = Activation('relu')(z2)

z3 = Conv2D(24, (1, 1))(y)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)
z3 = Conv2D(24, (3, 3), padding='same')(z3)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)

z4 = Conv2D(24, (1, 1))(y)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)
z4 = Conv2D(24, (3, 3), padding='same')(z4)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)
z4 = Conv2D(24, (3, 3), padding='same')(z4)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)

y = concatenate([z1, z2, z3, z4])

z1 = Conv2D(24, (1, 1))(y)
z1 = BatchNormalization()(z1)
z1 = Activation('relu')(z1)

z2 = AveragePooling2D((1, 1))(y)
z2 = Conv2D(24, (1, 1))(z2)
z2 = BatchNormalization()(z2)
z2 = Activation('relu')(z2)

z3 = Conv2D(24, (1, 1))(y)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)
z3 = Conv2D(24, (3, 3), padding='same')(z3)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)

z4 = Conv2D(24, (1, 1))(y)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)
z4 = Conv2D(24, (3, 3), padding='same')(z4)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)
z4 = Conv2D(24, (3, 3), padding='same')(z4)
z4 = BatchNormalization()(z4)
z4 = Activation('relu')(z4)

y = concatenate([z1, z2, z3, z4])
y = GlobalMaxPooling2D()(y)
y = Dense(10)(y)
y = Activation('softmax')(y)

model = Model(x, y)
model.summary()

plot_model(model, show_shapes=True)
