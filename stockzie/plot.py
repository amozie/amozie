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
import talib as tl

from matplotlib.dates import datestr2num
from matplotlib.finance import candlestick2_ohlc
from datetime import datetime, timedelta

import stockzie as sz

FIG_SIZE = (14, 7)
DATE_NUM = 20


class PlotException(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message


def _init(code=None, data=None, ax=None, start=None, end=None, ktype='D', figsize=FIG_SIZE):
    if data is None:
        if code is None:
            raise PlotException('code & data are all None')
        data = sz.data.get(code, start=start, end=end, ktype=ktype)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    return ax, data


def init_xtick_date(ax, data, num=DATE_NUM, label=True):
    length = data.shape[0]
    x_tick_space = length // num
    x_ticks = np.arange(length - 1, 0, -x_tick_space)
    x_ticks = x_ticks[::-1]

    ax.set_xlim([-1, length])
    ax.set_xticks(x_ticks)
    if label:
        ax.set_xticklabels([data.date.iloc[i] for i in x_ticks], rotation=30)
    else:
        ax.set_xticklabels([])


def init_fig_axes(row=1, data=None, top=0.01, down=0.10, left=0.05, right=0.02,
                  space=0.01, sub_height=0.15, num=DATE_NUM):
    if row < 1 or row > 4:
        raise PlotException('row must be >= 1 and <= 4')
    sns.set()
    fig = plt.figure(figsize=FIG_SIZE)
    axes = []
    sub_all_height = sub_height + space
    main_down = down + sub_all_height * (row - 1)
    main_height = 1 - main_down - top
    width = 1 - left - right

    for i in range(row):
        axes.append(fig.add_axes([left,
                                  main_down - sub_all_height * i,
                                  width,
                                  main_height if i == 0 else sub_height]))

    if data is not None:
        for i, ax in enumerate(axes):
            init_xtick_date(ax, data, num, i + 1 == len(axes))

    return fig, axes


def kline(ax=None, code=None, data=None, start=None, end=None, ktype='D', num=DATE_NUM,
          figsize=FIG_SIZE, width=1, colorup='w', colordown='k', alpha=0.5):
    ax, data = _init(code, ax=ax, start=start, end=end, ktype=ktype, data=data, figsize=figsize)

    candlestick2_ohlc(ax, data.open, data.high, data.low, data.close,
                      width, colorup, colordown, alpha)

    # init_xtick_date(ax, data, num)

    return data


def volume(ax=None, code=None, data=None, start=None, end=None, ktype='D', num=DATE_NUM,
           figsize=FIG_SIZE, width=0.75, colorup='gray', colordown='k', alpha=0.5):
    ax, data = _init(code, ax=ax, start=start, end=end, ktype=ktype, data=data, figsize=figsize)

    copy = data.copy()

    copy['v_up'] = copy.volume[copy.close > copy.open]
    copy['v_down'] = copy.volume[copy.close <= copy.open]

    ax.bar(np.arange(copy.v_up.size), copy.v_up, width=width, facecolor=colorup, alpha=alpha)
    ax.bar(np.arange(copy.v_down.size), copy.v_down, width=width, facecolor=colordown, alpha=alpha)

    # init_xtick_date(ax, data, num)

    return data
