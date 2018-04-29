import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import *
from sklearn import preprocessing
import cv2

# model = keras.applications.ResNet50()
model = keras.applications.DenseNet121()

ws = model.layers[2].get_weights()[0]

for i in range(64):
    ws_i = ws[:, :, :, i].reshape(-1, 3)
    ws_i = preprocessing.minmax_scale(ws_i).reshape(7, 7, 3)
    ws_i = cv2.resize(ws_i, (28, 28))
    ax = plt.subplot(8, 8, i+1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.imshow(ws_i)


