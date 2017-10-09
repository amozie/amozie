from stockzie.strategy.TechStrategy import TechStrategy
import numpy as np


class TestTechStrategy(TechStrategy):
    def _handle_trading(self, data):
        self._add_technique_iter('test', data.iloc[self.iter].close)


class MA520TechStrategy(TechStrategy):
    def __init__(self):
        super().__init__()
        self.ma5_list = []
        self.ma10_list = []
        self.ma20_list = []

    def _init_trading(self, data):
        super()._init_trading(data)
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
            if ma5_cross > self.ma5 and ma10_cross > self.ma10 and ma20_cross > self.ma20:
                ma520 = 1
                self.buy_simple(ma20_cross)
            elif ma20_cross < self.ma20:
                ma520 = -1
                self.sell_simple(ma5_cross)
            else:
                ma520 = 0
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
