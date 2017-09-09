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

import datazie as dz


class LinearModelStrategy(BaseStrategy):
    def __init__(self, data, cash=100000):
        super().__init__(data, cash)

    def _init_trading(self):
        self.close = []
        self.close_prd = []
        self.close_std_up = []
        self.close_std_down = []
        self.close_std = []

    def _handle_trading(self):
        N = 20
        if self._iter_i < N:
            lm = None
        else:
            lm = dz.model.LinearModel(self.__sofar_data.tail(N).close.values)
            lm.fit()
        if lm is None:
            self.close_prd.append(None)
            self.close_std_down.append(None)
            self.close_std_up.append(None)
            self.close_std.append(None)
        else:
            prd = lm.predict(N, alpha=0.1)
            self.close_prd.append(prd[0][0])
            self.close_std_down.append(prd[0][2])
            self.close_std_up.append(prd[0][3])
            self.close_std.append(prd[0][1])
        self.close.append(self._iter_datas[0].close)

    def _end_trading(self):
        self.__plot_dicts[0]['close'] = self.close
        self.__plot_dicts[0]['close_prd'] = self.close_prd
        self.__plot_dicts[0]['close_std_down'] = self.close_std_down
        self.__plot_dicts[0]['close_std_up'] = self.close_std_up
        self.__plot_dicts[1]['close_std'] = self.close_std


if __name__ == '__main__':
    data = sz.data.get('600056', ktype='60')
    st = LinearModelStrategy(data.tail(220))
    st.run()
    st.plot_demo(2)
