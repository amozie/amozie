from stockzie.strategy.TechStrategy import *
import numpy as np
import pywt


class TestTechStrategy(TechStrategy):
    def _handle_trading(self, data):
        self._add_technique_iter('test', data.iloc[self.iter].close)


class MA520TechStrategy(TechStrategy):
    def _init_trading(self, data):
        self.ma5_list = []
        self.ma10_list = []
        self.ma20_list = []

    def _handle_trading(self, data):
        con5 = 5
        con10 = 10
        con20 = 3
        ma5 = np.average(self.data_hist.close[-con5:])
        ma10 = np.average(self.data_hist.close[-con10:])
        ma20 = np.average(self.data_hist.close[-con20:])
        self.ma5_list.append(ma5)
        self.ma10_list.append(ma10)
        self.ma20_list.append(ma20)
        ma5_cross = np.average(self.data_hist.close.iloc[-con5:-1])
        ma10_cross = np.average(self.data_hist.close.iloc[-con10:-1])
        ma20_cross = np.average(self.data_hist.close.iloc[-con20:-1])
        try:
            last5_price = self.data_hist.close.iloc[-con5-1]
            last10_price = self.data_hist.close.iloc[-con10-1]
            last20_price = self.data_hist.close.iloc[-con20-1]
            ma5_hist = self.ma5_list[-con5-1]
            ma10_hist = self.ma10_list[-con10-1]
            ma20_hist = self.ma20_list[-con20-1]
        except IndexError:
            last5_price = np.nan
            last10_price = np.nan
            last20_price = np.nan
            ma5_hist = np.nan
            ma10_hist = np.nan
            ma20_hist = np.nan
        if self.iter > 1:
            price_cross_trend(self, ma20_cross)
            # if ((self.data_i.open <= ma20_cross < self.data_i.close) or
            #         (self.data_i.open > ma20_cross >= self.data_i.low)):
            #     self.buy_soft_percentage(ma20_cross)
            #     ma520 = 1
            # elif ((self.data_i.open >= ma20_cross > self.data_i.close) or
            #         (self.data_i.open < ma20_cross <= self.data_i.high)):
            #     self.sell_soft_percentage(ma20_cross)
            #     ma520 = -1
            # else:
            #     ma520 = 0
            # if ma5_cross > self.ma5 and ma10_cross > self.ma10 and ma20_cross > self.ma20:
            #     ma520 = 1
            #     self.buy_soft_percentage(ma20_cross)
            # elif ma20_cross < self.ma20:
            #     ma520 = -1
            #     self.sell_soft_percentage(ma5_cross)
            # else:
            #     ma520 = 0
        # self._add_technique_iter('ma1', ma5)
        # self._add_technique_iter('ma2', ma10)
        self._add_technique_iter('ma3', ma20_cross)
        self.ma5 = ma5
        self.ma10 = ma10
        self.ma20 = ma20

    def _end_trading(self, data):
        print('ma predict: {0}'.format(np.average(self.data_hist.close.iloc[-6 + 1:])))


class MultiMATechStrategy(TechStrategy):
    def _handle_trading(self, data):
        short = 3
        middle = 6
        long = 12
        ma_short_cross = np.average(self.data_hist.close.iloc[-short:-1])
        ma_middle_cross = np.average(self.data_hist.close.iloc[-middle:-1])
        ma_long_cross = np.average(self.data_hist.close.iloc[-long:-1])
        if self.iter >= short:
            price_cross_trend(self, ma_short_cross, ma_middle_cross, ma_long_cross)
        self._add_technique_iter('ma_s', ma_short_cross)
        self._add_technique_iter('ma_m', ma_middle_cross)
        self._add_technique_iter('ma_l', ma_long_cross)


class TwoMATechStrategy(TechStrategy):
    def _handle_trading(self, data):
        short = 3
        long = 6
        ma_short_cross = np.average(self.data_hist.close.iloc[-short:-1])
        ma_long_cross = np.average(self.data_hist.close.iloc[-long:-1])
        if self.iter >= short:
            price_cross_two_trend(self, ma_short_cross, ma_long_cross)
        self._add_technique_iter('ma_s', ma_short_cross)
        self._add_technique_iter('ma_l', ma_long_cross)


def price_cross_trend(self, *args):
    data_i = self.data_i
    op = data_i.open
    high = data_i.high
    low = data_i.low
    close = data_i.close

    for trend in args:
        if op <= trend < close or (op > trend > low and close > trend):
            self._buy_soft_percentage(trend)
        elif op >= trend > close or (op < trend < high and close < trend):
            self._sell_soft_percentage(trend)


def stg_price_cross_trend(self, *args):
    data_i = self.data_i
    op = data_i.open
    high = data_i.high
    low = data_i.low
    close = data_i.close

    try:
        last_trade = self.last_trade
    except AttributeError:
        self.last_trade = 0
        last_trade = self.last_trade

    for arg in args:
        if isinstance(arg, list) or isinstance(arg, tuple):
            trend, direct = arg
        else:
            trend = arg
            direct = 0
        if last_trade == -1 and op < trend:
            self._sell_soft_percentage(op)
        if op < trend and direct >= 0:
            self._buy_soft_percentage(trend)
            if close < trend:
                self.last_trade = -1
        elif op > trend and direct <= 0:
            self._sell_soft_percentage(trend)
            if close > trend:
                self._buy_soft_percentage(close)
                self.last_trade = 1


def price_cross_two_trend(self, t1, t2):
    data_i = self.data_i
    op = data_i.open
    high = data_i.high
    low = data_i.low
    close = data_i.close
    tmin = min(t1, t2)
    tmax = max(t1, t2)

    self._sell_soft_percentage(tmax)
    self._buy_soft_percentage(tmin)

