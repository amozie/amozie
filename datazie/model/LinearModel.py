# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


model_sm = 'sm'


class ModelException(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message


class LinearModel():
    def __init__(self, y, x=None, model=model_sm, add_constant=None):
        self.model = model
        self.add_constant = add_constant
        self.__features = None
        if x is None:
            x = np.arange(y.size)
        self.x = self.__pretreat_params(x)
        self.y = y

    def __pretreat_params(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not self.__features:
            if 1 == x.ndim:
                self.__features = 1
            elif 2 == x.ndim:
                self.__features = x.shape[1]
            else:
                raise ModelException('dimension of x is error')
        if 2 != x.ndim:
            x = x.reshape(-1, self.__features)
        if self.add_constant is None:
            if model_sm == self.model:
                x = self.__add_constant(x)
        elif self.add_constant:
            x = self.__add_constant(x)
        return x

    def __add_constant(self, x):
        # 样本数为1时，sm.add_constant存在bug，没有添加常数返回原数组
        if 1 == x.shape[0]:
            return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        else:
            return sm.add_constant(x)

    def __fit_sm(self):
        self.res = sm.OLS(self.y, self.x).fit()

    def fit(self):
        if model_sm == self.model:
            self.__fit_sm()

    def predict(self, x=None, alpha=0.05):
        if x is not None:
            x = self.__pretreat_params(x)
        ret = [self.res.predict(x)]
        ret.extend(wls_prediction_std(self.res, exog=x, alpha=alpha))
        return np.array(ret).T

    def summary(self):
        print(self.res.summary())

    def plot(self):
        fig = plt.figure()
        fig.suptitle('LINEAR MODEL')
        ax = plt.subplot(111)
        y_prd = self.predict(alpha=0.1)
        ax.plot(self.x[:, 1], self.y, '*', label='sample')
        ax.plot(self.x[:, 1], y_prd[:, 0], label='predict')
        ax.plot(self.x[:, 1], y_prd[:, 2], 'r--', label='std')
        ax.plot(self.x[:, 1], y_prd[:, 3], 'r--', label='std')
        plt.show()


if __name__ == '__main__':
    x = np.linspace(0, 10, 21)
    y = 3*x + 2
    y += np.random.randn(x.size)
    lm = LinearModel(y, x)
    lm.fit()
    lm.summary()
    print(lm.predict())
    lm.plot()
