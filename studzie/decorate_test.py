# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import wraps

class Parent():
    def decorate(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            print('start')
            f(*args, **kwargs)
            print('end')
        return decorated
    
    @decorate
    def run(self, text):
        print('parent run: {}'.format(text))

class Son(Parent):
    def run(self, text):
        print('son run: {}'.format(text))

def dec(fun):
    def decorated(*args, **kwargs):
        print('start')
        res = fun(*args, **kwargs)
        print('end')
        return res
    return decorated

@dec
def foo(a, b):
    return a*b

if __name__ == '__main__':
    res = foo(3,2)
    print(res)