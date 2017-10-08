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

from datetime import datetime

import stockzie as sz
from stockzie.technique import Technique


class StrategyError(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message


class TechStrategy():
    def __init__(self):
        self.__last_day = None
        self.techniques = {}
        self.iter = None
        self.data_hist = None

    ####################################

    def _init_trading(self, data):
        # 整个交易初始化执行一次
        pass

    def _before_trading(self, data):
        # 交易初始化后以及每天开盘前执行一次
        pass

    def _handle_trading(self, data):
        # 每根K线执行一次
        pass

    def _after_trading(self, data):
        # 每天收盘后执行一次
        pass

    def _end_trading(self, data):
        # 整个交易结束执行一次
        pass

    ####################################

    def add_technique_iter(self, name, value_iter, row=0, style='', width=None, alpha=None, x_axis=None):
        technique_dict = self.techniques.get(name, None)
        if technique_dict is None:
            self.techniques[name] = {
                'name': name,
                'value': [value_iter],
                'row': row,
                'style': style,
                'width': width,
                'alpha': alpha,
                'x_axis': x_axis
            }
        else:
            technique_dict['value'].append(value_iter)

    def run(self, data):
        global curbar
        self.techniques.clear()
        self._init_trading(data)
        self.__last_day = None
        for i in range(data.shape[0]):
            self.iter = i
            self.data_hist = data.iloc[0:i+1]
            try:
                datestr = data.iloc[i].date
            except AttributeError:
                raise StrategyError('data has no attribute \'date\'')
            try:
                date = datetime.strptime(datestr, '%Y-%m-%d')
            except ValueError:
                try:
                    date = datetime.strptime(datestr, '%Y-%m-%d %H:%M')
                except ValueError:
                    try:
                        date = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        raise StrategyError('cannot parse datestr to datetime')

            if self.__last_day != date.day:
                self._before_trading(data)

            self._handle_trading(data)

            if date.hour == 0 or date.hour == 15:
                self._after_trading(data)

            self.__last_day = date.day

        self._end_trading(data)

        return self.techniques


class Trading():
    def __init__(self, cash):
        self.__cash = cash
        self.__market_value = 0
        self.__total_value = cash
        self.__commision = 0.0005
        self.__stamp_tax = 0.001

        self.__total = [1]
        self.__is_position = False

    def buy_all(self, price):
        pass

    def sell_all(self, price):
        pass

