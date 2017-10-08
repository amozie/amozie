from stockzie.strategy.TechStrategy import TechStrategy
import numpy as np


class TestTechStrategy(TechStrategy):
    def _handle_trading(self, data):
        self._add_technique_iter('test', data.iloc[self.iter].close)


class MA520TechStrategy(TechStrategy):
    def _handle_trading(self, data):
        self._add_technique_iter('ma5', np.average(self.data_hist.close.tail(5)))
