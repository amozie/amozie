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

@register_tech('MAZ')
class MAZ(TechnicalBase):
    """ 移动平均线指标。 """
    @tech_init
    def __init__(self, data, n, name='MA',
                 style='y', lw=1):
        """ data (NumberSeries/np.ndarray/list) """
        super(MAZ, self).__init__(name)
        # 必须的函数参数
        self._args = [ndarray(data), n]

    def _rolling_algo(self, data, n, i):
        """ 逐步运行函数。"""
        ## @todo 因为用了向量化方法，速度降低
        return (talib.SMA(data, n)[i], )

    def _vector_algo(self, data, n):
        """向量化运行, 结果必须赋值给self.values。

        Args:
            data (np.ndarray): 数据
            n (int): 时间窗口大小
        """
        ## @NOTE self.values为保留字段！
        # 绘图和指标基类都会用到self.values
        self.values = talib.SMA(data, n)

    def plot(self, widget):
        """ 绘图，参数可由UI调整。 """
        self.widget = widget
        self.plot_line(self.values, self.style, lw=self.lw)