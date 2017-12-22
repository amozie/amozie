import numpy as np


DIS = 3**0.5/2


def point():
    theta = 2 * np.pi * np.random.rand()
    return np.array([np.cos(theta), np.sin(theta)])


def once():
    p1 = point()
    p2 = point()
    dis = np.linalg.norm(p1 - p2)
    if dis > DIS:
        return 1
    else:
        return 0


def many_times(num):
    true_count = 0
    all_count = 0
    for i in range(num):
        true_count += once()
        all_count += 1
    print(true_count, all_count, true_count/all_count)

import sympy as sp
from scipy import stats
import matplotlib.pyplot as plt

stats.multivariate_normal
n = stats.norm()
u = stats.uniform()
x = np.linspace(-5, 5, 100)
plt.plot(x, n.cdf(x))