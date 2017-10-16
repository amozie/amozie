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

def is_zero(val):
    return np.isclose(val, 0)


def bigger_than(val1, val2, allow_equal=False):
    err = val1 - val2
    if not allow_equal and err > 0 and not is_zero(err):
        return True
    elif allow_equal and (err >= 0 or is_zero(err)):
        return True
    else:
        return False


def smaller_than(val1, val2, allow_equal=False):
    return bigger_than(val2, val1, allow_equal)


class StrategyError(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message


class TechStrategy:
    def __init__(self, code):
        self.__code = code
        self.__last_day = None
        self.__techniques = {}

        self.trading = Trading(code, 1.0)

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

    def _buy_soft_percentage(self, price, percentage=1):
        self.trading.buy_soft_percentage(price, percentage)

    def _sell_soft_percentage(self, price, percentage=1):
        self.trading.sell_soft_percentage(price, percentage)

    def _get_avail_pos(self):
        return self.trading.avail_pos

    def _get_frozen_pos(self):
        return self.trading.frozen_pos

    def _get_position(self):
        return self.trading.position

    def _get_cash(self):
        return self.trading.cash

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
                self.trading.before_trading_day(self.iter)

            self._handle_trading(data)
            self.trading.calc_trading_bar(self.data_i, self.iter)

            if date.hour == 0 or date.hour == 15:
                self.trading.after_trading_day(self.iter)
                self._after_trading(data)

            self.__last_day = date.day

        self._end_trading(data)
        if bigger_than(self.trading.position, 0):
            # 强行清仓
            self.trading.position_after(data.shape[0] - 1)

        self.trading.trading_summary()
        self.__calc_trading_tech()

        return self.__techniques

    def __calc_trading_tech(self):
        self._add_technique('ASSET', self.trading.total_list, 1, twin=True)
        # self._add_technique('POS', self.trading.pos_list, 2)
        self._add_technique('BUY', self.trading.buy_list, 0, 'r.', x_axis=self.trading.buy_x)
        self._add_technique('SELL', self.trading.sell_list, 0, 'g.', x_axis=self.trading.sell_x)
        self._add_technique('STOP', self.trading.stop_list, 0, 'm.', x_axis=self.trading.stop_x)
        self._add_technique('WIN', self.trading.process_win_y, 1, 'r-',
                            x_axis=self.trading.process_win_x, twin=True)
        self._add_technique('FAIL', self.trading.process_fail_y, 1, 'g-',
                            x_axis=self.trading.process_fail_x, twin=True)


class Trading:
    def __init__(self, code, cash, stop=0.1):
        self.__code = code
        self.__commision = 0.0005
        self.__stamp_tax = 0.001
        self.__stop = stop

        self.cash = cash
        self.avail_pos = 0.0
        self.frozen_pos = 0.0
        self.position = 0.0
        self.pos_price = 0.0
        self.equity = 0.0
        self.total = cash
        self.last_max_price = 0.0
        self.init_total = cash

        self.last_pos = 0.0
        self.__trading_bar = []

        # 绘图变量
        self.total_list = []
        self.pos_list = []
        self.buy_list = []
        self.buy_x = []
        self.sell_list = []
        self.sell_x = []
        self.stop_list = []
        self.stop_x = []

        # 持仓过程变量
        self.__before_total = 0.0
        self.process_total = []
        self.__before_iter = 0
        self.process_iter = []
        # 计算持仓变量
        self.process_total_period = 0
        self.process_profit = []
        self.process_profit_period = []
        self.process_loss = []
        self.process_loss_period = []

        self.process_win_y = []
        self.process_win_x = []
        self.process_fail_y = []
        self.process_fail_x = []

    def buy_soft_percentage(self, price, percentage=1):
        self._add_trading_bar(price, True, self.cash * percentage / price)

    def sell_soft_percentage(self, price, percentage=1, stop=False):
        self._add_trading_bar(price, False, self.position * percentage, stop)

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
        self.position = self.avail_pos + self.frozen_pos
        self.pos_price = price
        self.equity = self.pos_price * self.position
        self.total = self.equity + self.cash

    def calc_trading_bar(self, data_i, itr, stop=0.1):
        low = data_i.low
        high = data_i.high
        close = data_i.close
        op = data_i.open
        trading_bar_sorted = sorted(self.__trading_bar, key=lambda i: i['price'], reverse=op > close)
        for trade in trading_bar_sorted:
            buy = trade['buy']
            price = trade['price']
            quantity = trade['quantity']
            equity = price * quantity
            if low <= price <= high and bigger_than(quantity, 0):
                if buy:
                    if bigger_than(self.cash, equity, True):
                        self.cash -= price * quantity
                        self.frozen_pos += quantity
                        self.buy_list.append(price)
                        self.buy_x.append(itr)
                else:
                    if bigger_than(self.avail_pos, quantity, True):
                        self.cash += price * quantity
                        self.avail_pos -= quantity
                        stop = trade['stop']
                        if not stop:
                            self.sell_list.append(price)
                            self.sell_x.append(itr)
                        else:
                            self.stop_list.append(price)
                            self.stop_x.append(itr)
                self.update_trading_value(price)
                if is_zero(self.last_pos) and bigger_than(self.position, 0):
                    # 持仓前
                    self.position_before(itr)
        self.update_trading_value(close)
        self.__trading_bar.clear()
        self._trading_stop(high)
        self.total_list.append(self.total)
        self.pos_list.append(self.position)

    def before_trading_day(self, itr):
        self.last_pos = self.position

    def after_trading_day(self, itr):
        self.avail_pos += self.frozen_pos
        self.frozen_pos = 0.0

        if bigger_than(self.last_pos, 0) and is_zero(self.position):
            # 空仓后
            self.position_after(itr)

    def _trading_stop(self, high):
        if bigger_than(self.avail_pos, 0):
            self.last_max_price = max(high, self.last_max_price)
            self.sell_soft_percentage(self.last_max_price * (1 - self.__stop), stop=True)
        else:
            self.last_max_price = 0.0

    def position_before(self, itr):
        self.__before_total = self.total
        self.__before_iter = itr

    def position_after(self, itr):
        self.process_total.append((self.__before_total, self.total))
        self.process_iter.append((self.__before_iter, itr))

    def init_trading_summary(self):
        self.process_total_period = 0
        self.process_profit = []
        self.process_profit_period = []
        self.process_loss = []
        self.process_loss_period = []

        self.process_win_y = []
        self.process_win_x = []
        self.process_fail_y = []
        self.process_fail_x = []

    def calc_trading_summary(self):
        self.init_trading_summary()
        for (before_total, after_total), (before_period, after_period) in \
                zip(self.process_total, self.process_iter):
            dif_total = after_total - before_total
            dif_period = after_period - before_period
            self.process_total_period += dif_period
            if bigger_than(dif_total, 0):
                self.process_profit.append(dif_total)
                self.process_profit_period.append(dif_period)

                self.process_win_y.append(before_total)
                self.process_win_y.append(after_total)
                self.process_win_y.append(np.nan)
                self.process_win_x.append(before_period)
                self.process_win_x.append(after_period)
                self.process_win_x.append(np.nan)
            else:
                self.process_loss.append(dif_total)
                self.process_loss_period.append(dif_period)

                self.process_fail_y.append(before_total)
                self.process_fail_y.append(after_total)
                self.process_fail_y.append(np.nan)
                self.process_fail_x.append(before_period)
                self.process_fail_x.append(after_period)
                self.process_fail_x.append(np.nan)

    def trading_summary(self):
        self.calc_trading_summary()
        print('#' * 20)

        print('股票代码：{0}'.format(self.__code))
        print('总周期：{0}'.format(len(self.total_list)))
        print('总资产：{0}'.format(self.total))
        # print('净利润：{0}'.format(self.total - self.init_total))
        # print('平均每日净利润：{0}'.format((self.total - self.init_total)/self.process_total_period))
        #
        # print('-' * 10)
        #
        # if len(self.process_profit) > 0:
        #     print('总利润：{0}'.format(sum(self.process_profit)))
        #     print('平均每日利润：{0}'.format(sum(self.process_profit)/sum(self.process_profit_period)))
        # if len(self.process_loss) > 0:
        #     print('总损失：{0}'.format(sum(self.process_loss)))
        #     print('平均每日损失：{0}'.format(sum(self.process_loss)/sum(self.process_loss_period)))

        print('#' * 20)
