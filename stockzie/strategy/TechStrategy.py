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
        self.__techniques = {}
        self.__trading_simple = {}

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

    def _add_technique(self, name, value, row=0, style='', width=None, alpha=None, x_axis=None):
        self.__techniques[name] = {
                'name': name,
                'value': value,
                'row': row,
                'style': style,
                'width': width,
                'alpha': alpha,
                'x_axis': x_axis
            }

    def _add_technique_iter(self, name, value_iter, row=0, style='', width=None, alpha=None, x_axis=None):
        technique_dict = self.__techniques.get(name)
        if technique_dict is None:
            self.__techniques[name] = {
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
        self.__techniques.clear()
        self.__trading_simple.clear()
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

        self.__calc_trading_simple(data)

        return self.__techniques

    def __trade_simple(self, price, buy):
        trade = self.__trading_simple.get(self.iter)
        if trade is None:
            self.__trading_simple[self.iter] = [
                {
                    'buy': buy,
                    'price': price
                }
            ]
        else:
            trade.append(
                {
                    'buy': buy,
                    'price': price
                }
            )

    def buy_simple(self, price):
        self.__trade_simple(price, True)

    def sell_simple(self, price):
        self.__trade_simple(price, False)

    def __calc_trading_simple(self, data, stop=0.1):
        total = 1
        buy_date = None
        total_trading = []
        stop_price = None
        for i in range(data.shape[0]):
            high = data.high.iloc[i]
            low = data.low.iloc[i]
            trade = self.__trading_simple.get(i)
            if stop_price is not None:
                if low <= stop_price:
                    buy_date = None
                    last_price = data.iloc[i - 1].close
                    total = total * stop_price / last_price
                    stop_price = None
            if trade is not None:
                datestr = data.iloc[i].date.split()[0]
                try:
                    date = datetime.strptime(datestr, '%Y-%m-%d')
                except ValueError:
                    raise StrategyError('cannot parse datestr to datetime')
                for j in trade:
                    if j['buy']:
                        if buy_date is None:
                            buy_price = j['price']
                            if low <= buy_price <= high:
                                buy_date = date
                                cur_price = data.iloc[i].close
                                total = total * cur_price / buy_price
                                if stop > 0:
                                    stop_price = high * (1 - stop)
                    else:
                        if buy_date is not None and date > buy_date:
                            sell_price = j['price']
                            if low <= sell_price <= high:
                                buy_date = None
                                last_price = data.iloc[i-1].close
                                total = total * sell_price / last_price
                                if stop > 0:
                                    stop_price = None
            else:
                if buy_date is not None:
                    last_price = data.iloc[i - 1].close
                    cur_price = data.iloc[i].close
                    total = total * cur_price / last_price

            total_trading.append(total)
        self._add_technique('Asset', total_trading, 2)




class Trading():
    def __init__(self, cash):
        self.__cash = cash
        self.__market_value = 0
        self.__total_value = cash
        self.__commision = 0.0005
        self.__stamp_tax = 0.001

        self.__total = 1
        self.__is_position = False

    def buy_all(self, price):
        pass

    def sell_all(self, price):
        pass

