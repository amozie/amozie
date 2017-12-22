from keras.datasets import boston_housing
from keras.utils import to_categorical
from sklearn import preprocessing

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_fix = scaler.transform(X_train)
X_train_fix[:, 3] = X_train[:, 3]
X_test_fix = scaler.transform(X_test)
X_test_fix[:, 3] = X_test[:, 3]

x = Input((13,))
y = Dense(16, activation='tanh')(x)
y = BatchNormalization()(y)
y = Dense(16, activation='tanh')(y)
y = BatchNormalization()(y)
y = Dense(1)(y)

model = Model(x, y)

model.compile('adam', 'mse', ['accuracy'])

# hist = model.fit(X_train_fix, y_train, epochs=200, validation_data=(X_test_fix, y_test), verbose=2)

ax = plt.subplot(111)
epoch_start = 0
hist = model.fit(X_train_fix, y_train, epochs=300, validation_data=(X_test_fix, y_test), verbose=2,
                 callbacks=[LambdaCallback(on_epoch_end=
                                           lambda epoch, logs:
                                           ax.scatter(epoch+epoch_start, np.log(logs['val_loss']), c='b', marker='.'))])
epoch_start += 300

y_pred = model.predict(X_test_fix)
y_pred = y_pred.flatten()

plt.plot(list(zip(y_test, y_pred)))
plt.plot(y_test - y_pred)
np.linalg.norm(y_test - y_pred)
