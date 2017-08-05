# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tushare as ts
import stockzie as sz


def plotkv(code, ktype='D'):
    sns.set()

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('K-LINE-DIAGRAM')

    ax1 = fig.add_axes([0.05, 0.26, 0.93, 0.73])
    ax2 = fig.add_axes([0.05, 0.10, 0.93, 0.15])

    data = sz.data.get(code, ktype=ktype)

    sz.plot.kline(ax1, data=data)
    sz.plot.volume(ax2, data=data)

    ax1.set_xticklabels([])

    plt.show()

    return data

