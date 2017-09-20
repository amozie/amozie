import numpy as np
import talib
from quantdigger.technicals import TechnicalBase
from quantdigger.technicals.base import tech_init, ndarray
from quantdigger.technicals.techutil import register_tech


@register_tech('TEST')
class TEST(TechnicalBase):
    @tech_init
    def __init__(self, data, name='TEST',
                 style='y', lw=1):
        super(TEST, self).__init__(name)
        self._args = [ndarray(data), 1]

    def _rolling_algo(self, data, n, i):
        return data[i],

    def _vector_algo(self, data, n):
        self.values = data

    def plot(self, widget):
        self.widget = widget
        self.plot_line(self.values, self.style, lw=self.lw)

@register_tech('MACD')
class MACD(TechnicalBase):
    @tech_init
    def __init__(self, data, name='MACD',
                 style=('y','b','g', 'k--'), lw=1):
        super(MACD, self).__init__(name)
        self._args = [ndarray(data), 1]

    def _rolling_algo(self, data, n, i):
        s1, s2, s3 = talib.MACD(data)
        return s1[i], s2[i], s3[i], 0

    def _vector_algo(self, data, n):
        s1, s2, s3 = talib.MACD(data)
        self.values = {
            's1': s1,
            's2': s2,
            's3': s3,
            's4': np.zeros(len(s1))
        }

    def plot(self, widget):
        self.widget = widget
        self.plot_line(self.values['s1'], self.style[0], lw=self.lw)
        self.plot_line(self.values['s2'], self.style[1], lw=self.lw)
        self.plot_line(self.values['s3'], self.style[2], lw=self.lw)
        self.plot_line(self.values['s4'], self.style[3], lw=self.lw)