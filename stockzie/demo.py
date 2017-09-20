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


def plotkv(code=None, data=None, ktype='D'):
    sns.set()
    
    if data is None:
        data = sz.data.get(code, ktype=ktype)

    data = data[0]
    fig, ax = sz.plot.init_fig_axes(2, data)
    ax1, ax2 = ax

    sz.plot.kline(ax1, data=data)
    sz.plot.volume(ax2, data=data)

    ax1.set_xticklabels([])
    plt.show()

