import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
from time import sleep

img = cv2.imread('f:/tyt.png')
cv2.imshow('', img)

img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('', gray)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)

img[dst>0.01*dst.max()] = [0, 0, 255]

cv2.matchTemplate()