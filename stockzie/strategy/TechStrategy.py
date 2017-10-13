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

TOL_ERR = 0.001


class StrategyError(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message


class TechStrategy:
    def __init__(self):
        self.__last_day = None
        self.__techniques = {}
        self.__trading = Trading(1.0)

        self.iter = None
        self.data_hist = None
        self.data_i = None

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

    def _add_technique(self, name, value, row=0, style='', width=None, alpha=None, x_axis=None, twin=False):
        self.__techniques[name] = {
                'name': name,
                'value': value,
                'row': row,
                'style': style,
                'width': width,
                'alpha': alpha,
                'x_axis': x_axis,
                'twin': twin
            }

    def _add_technique_iter(self, name, value_iter, row=0, style='', width=None, alpha=None, x_axis=None, twin=False):
        technique_dict = self.__techniques.get(name)
        if technique_dict is None:
            self.__techniques[name] = {
                'name': name,
                'value': [value_iter],
                'row': row,
                'style': style,
                'width': width,
                'alpha': alpha,
                'x_axis': x_axis,
                'twin': twin
            }
        else:
            technique_dict['value'].append(value_iter)

    def run(self, data):
        self._init_trading(data)
        self.__last_day = None
        for i in range(data.shape[0]):
            self.iter = i
            self.data_hist = data.iloc[0:i+1]
            self.data_i = data.iloc[i]
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
            self.__trading.calc_trading_bar(self.data_i, self.iter)

            if date.hour == 0 or date.hour == 15:
                self.__trading.trading_bar_day()
                self._after_trading(data)

            self.__last_day = date.day

        self._end_trading(data)

        self.__calc_trading_tech()

        return self.__techniques

    def buy_soft_percentage(self, price, percentage=1.0):
        self.__trading.buy_soft_percentage(price, percentage)

    def sell_soft_percentage(self, price, percentage=1.0):
        self.__trading.sell_soft_percentage(price, percentage)

    def __calc_trading_tech(self):
        self._add_technique('ASSET', self.__trading.total_list, 1, twin=True)
        self._add_technique('POS', self.__trading.pos_list, 2)
        self._add_technique('BUY', self.__trading.buy_list, 0, 'r.', x_axis=self.__trading.buy_x)
        self._add_technique('SELL', self.__trading.sell_list, 0, 'g.', x_axis=self.__trading.sell_x)
        self._add_technique('STOP', self.__trading.stop_list, 0, 'm.', x_axis=self.__trading.stop_x)


class Trading:
    def __init__(self, cash, stop=0.1):
        self.__commision = 0.0005
        self.__stamp_tax = 0.001
        self.__stop = stop

        self.__cash = cash
        self.__avail_pos = 0.0
        self.__frozen_pos = 0.0
        self.__position = 0.0
        self.__pos_price = 0.0
        self.__equity = 0.0
        self.__total = cash
        self.__last_max_price = 0.0

        self.__trading_bar = []

        self.total_list = []
        self.pos_list = []
        self.buy_list = []
        self.buy_x = []
        self.sell_list = []
        self.sell_x = []
        self.stop_list = []
        self.stop_x = []

    def buy_soft_percentage(self, price, percentage=1.0):
        self._add_trading_bar(price, True, self.__cash * percentage / price)

    def sell_soft_percentage(self, price, percentage=1.0, stop=False):
        self._add_trading_bar(price, False, self.__position * percentage, stop)

    def _add_trading_bar(self, price, buy, quantity, stop=False):
        self.__trading_bar.append(
            {
                'buy': buy,
                'price': price,
                'quantity': quantity,
                'stop': stop
            }
        )

    def update_trading_value(self, price):
        # self.__cash
        # self.__avail_pos
        # self.__frozen_pos
        self.__position = self.__avail_pos + self.__frozen_pos
        self.__pos_price = price
        self.__equity = self.__pos_price * self.__position
        self.__total = self.__equity + self.__cash

    def calc_trading_bar(self, data_i, itr, stop=0.1):
        low = data_i.low
        high = data_i.high
        close = data_i.close
        for trade in self.__trading_bar:
            buy = trade['buy']
            price = trade['price']
            quantity = trade['quantity']
            if low <= price <= high:
                if buy:
                    if self.__cash + TOL_ERR >= price * quantity > TOL_ERR:
                        self.__cash -= price * quantity
                        self.__frozen_pos += quantity
                        self.buy_list.append(price)
                        self.buy_x.append(itr)
                else:
                    if self.__avail_pos + TOL_ERR >= quantity > TOL_ERR:
                        self.__cash += price * quantity
                        self.__avail_pos -= quantity
                        stop = trade['stop']
                        if not stop:
                            self.sell_list.append(price)
                            self.sell_x.append(itr)
                        else:
                            self.stop_list.append(price)
                            self.stop_x.append(itr)
                self.update_trading_value(price)
        self.update_trading_value(close)
        self.__trading_bar.clear()
        self._trading_stop(high)
        self.total_list.append(self.__total)
        self.pos_list.append(self.__position)

    def trading_bar_day(self):
        self.__avail_pos += self.__frozen_pos
        self.__frozen_pos = 0

    def _trading_stop(self, high):
        if self.__avail_pos > TOL_ERR:
            self.__last_max_price = max(high, self.__last_max_price)
            self.sell_soft_percentage(self.__last_max_price * (1 - self.__stop), stop=True)
        else:
            self.__last_max_price = 0.0
