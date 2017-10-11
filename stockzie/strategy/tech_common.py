from stockzie.strategy.TechStrategy import TechStrategy
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
        con20 = 20
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
        if self.iter < con20 + 1:
            ma520 = 0
        else:
            if ((self.data_i.open <= ma20_cross < self.data_i.close) or
                    (self.data_i.open > ma20_cross >= self.data_i.low)):
                self.buy_simple(ma20_cross)
                ma520 = 1
            elif ((self.data_i.open >= ma20_cross > self.data_i.close) or
                    (self.data_i.open < ma20_cross <= self.data_i.high)):
                self.sell_simple(ma20_cross)
                ma520 = -1
            else:
                ma520 = 0
            # if ma5_cross > self.ma5 and ma10_cross > self.ma10 and ma20_cross > self.ma20:
            #     ma520 = 1
            #     self.buy_simple(ma20_cross)
            # elif ma20_cross < self.ma20:
            #     ma520 = -1
            #     self.sell_simple(ma5_cross)
            # else:
            #     ma520 = 0
        self._add_technique_iter('ma1', ma5)
        self._add_technique_iter('ma2', ma10)
        self._add_technique_iter('ma3', ma20)
        self._add_technique_iter('ma520', ma520, 3)
        self.ma5 = ma5
        self.ma10 = ma10
        self.ma20 = ma20

    def _after_trading(self, data):
        test = data.close.values
        test = np.insert(test, 0 , [np.nan]*20)
        # self._add_technique('test', test, x_axis=np.arange(test.size))


class WaveHisTechStrategy(TechStrategy):
    def _init_trading(self, data):
        self.wavelet = 'db2'
        self.level = 8

    def _handle_trading(self, data):
        close = self.data_hist.close.values[:-1]
        wla = np.nan
        trend = 0
        if self.iter > 1:
            wp = pywt.WaveletPacket(close, self.wavelet, maxlevel=self.level)
            new_wp = pywt.WaveletPacket(None, self.wavelet, maxlevel=self.level)
            node = 'a' * self.level
            new_wp[node] = wp[node]
            wl = new_wp.reconstruct()
            try:
                wla = wl[close.size]
                wla_0 = wl[close.size - 1]
            except IndexError:
                wla = wl[close.size - 1]
                wla_0 = wl[close.size - 2]
            trend = wla - wla_0
        if trend > 0:
            trend = 1
        elif trend < 0:
            trend = -1
        else:
            trend = 0
        self._add_technique_iter('wl', wla)
        self._add_technique_iter('trend', trend, 3)

        trade = 0
        if self.iter > 1:
            trade = price_cross_trend(self, wla)
        self._add_technique_iter('trade', trade, 3)


def price_cross_trend(self, trend_price):
    data_i = self.data_i
    if data_i.open <= trend_price < data_i.close or (data_i.open > trend_price > data_i.low and data_i.close > trend_price):
        self.buy_simple(trend_price)
        return 1
    elif data_i.open >= trend_price > data_i.close or (data_i.open < trend_price < data_i.high and data_i.close < trend_price):
        self.sell_simple(trend_price)
        return -1
    else:
        return 0


