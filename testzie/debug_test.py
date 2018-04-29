import pdb
import numpy as np


def foo():
    s = 0
    x = 10
    for i in range(x):
        pdb.set_trace()
        s += x
        print(s)
    return s