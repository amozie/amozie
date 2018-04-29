from numba import jit, int32, float32
import numpy as np


def f(arr):
    s = 0
    for i in np.nditer(arr):
        s += i
    return s


@jit
def fj(arr):
    s = 0
    for i in np.nditer(arr):
        s += i
    return s

