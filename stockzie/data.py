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
import stockzie as sz

from matplotlib.dates import datestr2num
from matplotlib.finance import candlestick2_ohlc
from datetime import datetime, timedelta


BACK_DAYS = 360


def get(code, start=None, end=None, ktype='D'):
    if start is None:
        day = BACK_DAYS
        start = datetime.today() - timedelta(days=day)
        start = datetime.strftime(start, '%Y-%m-%d')
    if end is None:
        end = datetime.today()
        end = datetime.strftime(end, '%Y-%m-%d')
    data = ts.get_k_data(code, start=start, end=end, ktype=ktype)

    return data


def add_datenum(data):
    copy = data.copy()
    copy['dateNum'] = copy.date.map(datestr2num)
    return copy
