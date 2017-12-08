import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import RFE

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))
y_train_arg = np.argmax(y_train, 1)
y_test_arg = np.argmax(y_test, 1)

# 神经网络
x = Input((784,))
y = Dense(32)(x)
y = Activation('tanh')(y)
y = Dense(32)(y)
y = Activation('tanh')(y)
y = Dropout(0.5)(y)
y = Dense(10)(y)
y = Activation('softmax')(y)
model = Model(x, y)
# model.summary()
model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])
hist = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=2, batch_size=128,
                 callbacks=[TensorBoard('c:/users/dell/appdata/local/temp/tf.log')])

# 多项式贝叶斯
model = MultinomialNB(alpha=0.001)
model.fit(X_train, y_train_arg)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test_arg, y_pred)

# 伯努利贝叶斯
model = BernoulliNB(alpha=0.001)
model.fit(X_train, y_train_arg)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test_arg, y_pred)

# KNN
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# LR
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# 随机森林
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

feature = model.feature_importances_
plt.bar(np.arange(1, feature.size + 1), feature)
plt.imshow(feature.reshape(28, 28))

# 决策树
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# GBDT
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# SVM
model = SVC(probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# RFE 特征选择
model = RFE(RandomForestClassifier())
fit = model.fit(X_train, y_train_arg)
