from stockzie.technique import Technique
import talib
import numpy as np
import statsmodels.api as sm


class TestTechnique(Technique):
    def run(self, data) -> list:
        super()._add_technique('close', data.close, 0, 'r--', 1, 0.8)
        super()._add_technique('open', data.open, 0, 'b--', 1, 0.8)
        return super().run(data)


class MA520Technique(Technique):
    def run(self, data) -> list:
        ma5 = talib.MA(data.close.values, 5)
        ma20 = talib.MA(data.close.values, 20)
        super()._add_technique('ma5', ma5)
        super()._add_technique('ma20', ma20)
        return super().run(data)


class MA520LineTechnique(Technique):
    def run(self, data) -> list:
        index = np.arange(data.index.size)
        ma_fast = talib.MA(data.close.values, 12)
        ma_slow = talib.MA(data.close.values, 18)
        close = data.close.values
        x_up = []
        y_up = []
        x_down = []
        y_down = []
        cross = ma_fast - ma_slow
        flag = None
        for i in index:
            if i == 0:
                if cross[i] > 0:
                    flag = True
                else:
                    flag = False
            else:
                if not flag and (i == len(index)-1 or cross[i] > 0):
                    x_up.append(i)
                    y_up.append(close[i])
                    x_down.append(i)
                    y_down.append(close[i])
                    x_down.append(np.nan)
                    y_down.append(np.nan)
                    flag = True
                if flag and (i == len(index)-1 or cross[i] < 0):
                    x_down.append(i)
                    y_down.append(close[i])
                    x_up.append(i)
                    y_up.append(close[i])
                    x_up.append(np.nan)
                    y_up.append(np.nan)
                    flag = False
        super()._add_technique('up', y_up, x_axis=x_up, width=3)
        super()._add_technique('down', y_down, x_axis=x_down, width=3)
        super()._add_technique('ma5', ma_fast, 2, width=1, alpha=0.5)
        super()._add_technique('ma20', ma_slow, 2, width=1, alpha=0.5)
        return super().run(data)


class LineTechnique(Technique):
    def _evaluate(self):
        pass

    def run(self, data) -> list:
        index = np.arange(data.index.size)
        close = data.close.values
        start = 0

        return super().run(data)