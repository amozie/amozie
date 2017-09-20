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

_BACK_DAYS = 360

_PERIOD_DICT = {
    'D': 1,
    'W': 5,
    'M': 20,
    '60': 1 / 4,
    '30': 1 / 8,
    '15': 1 / 16,
    '5': 1 / 48
}

_TAIL_NUM = 240


def get(codes, start=None, end=None, ktype='D'):
    if start is None and end is None:
        tail_flag = True
    else:
        tail_flag = False
    if start is None:
        day = int(_BACK_DAYS * _PERIOD_DICT.get(ktype.upper(), 1))
        start = datetime.today() - timedelta(days=day)
        start = datetime.strftime(start, '%Y-%m-%d')
    if end is None:
        end = datetime.today()
        end = datetime.strftime(end, '%Y-%m-%d')
    codes = to_list(codes)
    datas = []
    # for code in codes:
    #     df = ts.get_k_data(code, start=start, end=end, ktype=ktype)
    #     df.set_index('date', False, inplace=True)
    #     data.append(df)
    # data = pd.concat(data, 1, keys=range(len(codes)))
    # data.sort_index(inplace=True)
    # if tail_flag:
    #     data = data.tail(_TAIL_NUM)
    for code in codes:
        df = ts.get_k_data(code, start=start, end=end, ktype=ktype)
        datas.append(df)
    return datas


# def datas_to_list(datas):
#     if isinstance(datas, list):
#         return datas
#     data_list = []
#     try:
#         for i in range(datas.columns.levels[0].size):
#             data_list.append(datas.loc[:, i])
#     except:
#         data_list = []
#     finally:
#         return data_list


def add_datenum(data):
    copy = data.copy()
    copy['dateNum'] = copy.date.map(datestr2num)
    return copy


def to_list(params):
    if isinstance(params, list):
        return params
    return [params]