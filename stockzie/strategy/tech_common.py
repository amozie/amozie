from stockzie.strategy.TechStrategy import TechStrategy
import numpy as np


class TestTechStrategy(TechStrategy):
    def _handle_trading(self, data):
        self._add_technique_iter('test', data.iloc[self.iter].close)


class MA520TechStrategy(TechStrategy):
    def _handle_trading(self, data):
        ma5 = np.average(self.data_hist.close.tail(5))
        ma10 = np.average(self.data_hist.close.tail(10))
        ma20 = np.average(self.data_hist.close.tail(20))
        if self.iter == 0:
            ma520 = 0
        elif self.ma5 < self.ma10 and ma5 > ma10 and self.ma5 < ma5 and self.ma20 < ma20:
            ma520 = 1
            self.buy_simple(data.iloc[self.iter].close)
        elif self.ma5 > self.ma10 and ma5 < ma10:
            ma520 = -1
            self.sell_simple(data.iloc[self.iter].close)
        else:
            ma520 = 0
        self._add_technique_iter('ma5', ma5)
        self._add_technique_iter('ma10', ma10)
        self._add_technique_iter('ma20', ma20)
        self._add_technique_iter('ma520', ma520, 3)
        self.ma5 = ma5
        self.ma10 = ma10
        self.ma20 = ma20
