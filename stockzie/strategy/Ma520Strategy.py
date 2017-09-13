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
import talib as tl
from stockzie.strategy import BaseStrategy


class Ma520Strategy(BaseStrategy):
    def __init__(self, data, cash=100000):
        super().__init__(data, cash)

    def _init_trading(self):
        self.ma5 = []
        self.ma20 = []

    def _handle_trading(self):
        # if self._iter_i == 0:
        #     self.ma5.append(None)
        #     self.ma20.append(None)
        # else:
        #     self.ma5.append(self._sofar_data_list[0].tail(5).close.mean())
        #     self.ma20.append(self._sofar_data_list[0].tail(20).close.mean())

        self.ma5.append(self._sofar_datas.tail(5).close.mean())
        self.ma20.append(self._sofar_datas.tail(20).close.mean())

    def _end_trading(self):
        self._add_plot_dict(self.ma5, 0, label='ma5')
        self._add_plot_dict(self.ma20, 0, label='ma20')
        # 测试MACD
        macds = []


        macd = tl.MACD(self._data_list[0].close.values)
        macd = np.array(macd)
        # for i, v in enumerate(macd):
        #     self._add_plot_dict(v, 2, 0, str(i))

        self._add_plot_dict(macd, 2, 0, label=['a','b','c'])


if __name__ == '__main__':
    data = sz.data.get(['600056', '300383'])
    st = Ma520Strategy(data.tail(200))
    st.run()
    st.plot_demo()

