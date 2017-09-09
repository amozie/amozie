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


class StrategyError(Exception):
    def __init__(self, message):
        super().__init__(self)
        self.message = message


class BaseStrategy():
    def __init__(self, data, cash=100000):
        self.__data = data
        self._datas = sz.data.data_to_list(data)
        self._trading = Trading(cash)
        self.output_dict = {}
        self.__plot_dicts = []

    ####################################
    '''
    _datas              用于回测的所有数据（列表，避免使用）
    _sofar_datas        从回测开始至当前（不包含）可用的data数据（列表，常用）
    _trading            交易对象
    _iter_i             当前遍历步数（从零开始）
    _iter_datas         当前时间点的data数据（列表）
    __plot_dicts        自动绘图数据存放字典，序号0-3对应ax1-4
                        按照{'label':list}格式存入相应序号字典
                        label为字符串，list为长度与data一致的列表
    output_dict         输出字典，需要返回可用
    '''

    def _init_trading(self):
        # 整个交易初始化执行一次
        pass

    def _before_trading(self):
        # 交易初始化后以及每天开盘前执行一次
        pass

    def _handle_trading(self):
        # 每根K线执行一次
        pass

    def _after_trading(self):
        # 每天收盘后执行一次
        pass

    def _end_trading(self):
        # 整个交易结束执行一次
        pass

    ####################################

    def _run(self):
        if not self.__last_day:
            self._init_trading()

        try:
            datestr = self.__iter_data[0]
        except AttributeError:
            raise BaseStrategy('data has no attribute \'date\'')

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

        if not (self.__last_day == date.day):
            self._before_trading()

        self._handle_trading()

        if date.hour == 0 or date.hour == 15:
            self._after_trading()

        self.__last_day = date.day
        self._iter_i += 1
        self.__sofar_data = self.__data[:self._iter_i]
        self._sofar_datas = sz.data.data_to_list(self.__sofar_data)
        try:
            self.__iter_data = next(self.__iter)
            self._iter_datas = sz.data.data_to_list(self.__iter_data)
        except StopIteration:
            self._iter_datas = []
            self._end_trading()
            return
        self._run()

    def run(self):
        self.__iter = self.__data.itertuples()
        self.__last_day = None
        self._iter_i = 0
        self.__sofar_data = None
        self._sofar_datas = []
        try:
            self.__iter_data = next(self.__iter)
            self._iter_datas = sz.data.data_to_list(self.__iter_data)
        except StopIteration:
            self._iter_datas = []
            return
        self._run()

    def _add_plot_dict(self, fig_i, ax_i, y, label=None, options='', x=None):
        dt = {
            'fig_i': fig_i,
            'ax_i': ax_i,
            'y': y,
            'label': label,
            'x': x,
            'options': options
        }
        self.__plot_dicts.append(dt)

    def _plot_dict(self, axes):
        for dt in self.__plot_dicts:
            fig_i = dt['fig_i']
            ax_i = dt['ax_i']
            y = dt['y']
            label = dt['label']
            x = np.arange(len(dt['y'])) if dt['x'] is None else dt['x']
            options = dt['options']
            axes[fig_i][ax_i].plot(x, y, options, label=label)
        for ax in axes:
            for i in ax:
                i.legend()

    def plot_demo(self, row=2, kline=True, volume=True):
        sns.set()
        axes = []
        row_dict = {}
        for dt in self.__plot_dicts:
            fig_i = dt['fig_i']
            ax_i = dt['ax_i']
            row_dict[fig_i] = np.max([row_dict.get(fig_i, row), ax_i + 1])
        for i, data in enumerate(self._datas):
            fig, ax = sz.plot.init_fig_axes(row_dict.get(i, row), data)
            fig.suptitle(data.code.dropna()[0])
            axes.append(ax)
            if row >= 1 and kline:
                sz.plot.kline(ax[0], data=data)
            if row >= 2 and volume:
                sz.plot.volume(ax[1], data=data)
        self._plot_dict(axes)

        plt.show()


class Trading():
    def __init__(self, cash):
        self.__cash = cash
        self.__market_value = 0
        self.__total_value = cash
        self.__commision = 0.0005
        self.__stamp_tax = 0.001


if __name__ == '__main__':
    print('main...')
    data = sz.data.get('600056')
    bs = BaseStrategy(data.tail(200))
    bs.run()
    bs.plot_demo()
