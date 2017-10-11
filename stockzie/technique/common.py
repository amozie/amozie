from stockzie.technique import *
import talib
import numpy as np
import pywt
import statsmodels.api as sm


class TestTechnique(Technique):
    def run(self, data) -> list:
        self._add_technique('close', data.close, 0, 'r--', 1, 0.8)
        self._add_technique('open', data.open, 0, 'b--', 1, 0.8)
        return super().run(data)


class MA520Technique(Technique):
    def run(self, data) -> list:
        ma5 = talib.MA(data.close.values, 5)
        ma20 = talib.MA(data.close.values, 20)
        signal = []
        for i in range(len(ma5)):
            if i > 0 and ma5[i-1] < ma20[i-1] and ma5[i] > ma20[i]:
                signal.append(1)
            elif i > 0 and ma5[i-1] > ma20[i-1] and ma5[i] < ma20[i]:
                signal.append(-1)
            else:
                signal.append(0)
        trade, asset = cal_trade_asset(data, signal, True, 1)

        self._add_technique('ma5', ma5)
        self._add_technique('ma20', ma20)
        self._add_technique('trading', trade, 2, '*')
        self._add_technique('asset', asset, 3)
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
        self._add_technique('up', y_up, x_axis=x_up, width=3)
        self._add_technique('down', y_down, x_axis=x_down, width=3)
        self._add_technique('ma5', ma_fast, 2, width=1, alpha=0.5)
        self._add_technique('ma20', ma_slow, 2, width=1, alpha=0.5)
        return super().run(data)


class ATRTechnique(Technique):
    def run(self, data) -> list:
        close = data.close.values
        high = data.high.values
        low = data.low.values
        atr14 = talib.ATR(high, low, close)
        atr21 = talib.ATR(high, low, close, 21)
        atr42 = talib.ATR(high, low, close, 42)
        close_up = close - atr21
        close_down = close + atr21
        close_up = np.insert(close_up[:-1], 0, np.nan)
        close_down = np.insert(close_down[:-1], 0, np.nan)
        self._add_technique('ATR14', atr14, 2)
        self._add_technique('ATR', atr42, 2)
        self._add_technique('UP', close_up, style='b--', width=1, alpha=0.5)
        self._add_technique('DOWN', close_down, style='b--', width=1, alpha=0.5)
        return super().run(data)


class WaveletTechnique(Technique):
    def run(self, data) -> list:
        close = data.close.values[:-7]
        wavelet = 'db3'
        level = 4
        wp = pywt.WaveletPacket(close, wavelet, maxlevel=level)
        wl_list = []
        for i in range(level):
            new_wp = pywt.WaveletPacket(None, wavelet, maxlevel=level)
            node = 'a' * (i+1)
            new_wp[node] = wp[node]
            wl = new_wp.reconstruct()
            wl_list.append(wl)
        for i, wl in enumerate(wl_list):
            if i < level-1:
                continue
            self._add_technique('WL{0}'.format(i + 1), wl, x_axis=np.arange(wl.size))
        # node_list = ['daa', 'ada', 'aad']
        # wld_list = []
        # for i in range(level):
        #     new_wp = pywt.WaveletPacket(None, wavelet, maxlevel=level)
        #     node = node_list[i]
        #     new_wp[node] = wp[node]
        #     wld = new_wp.reconstruct()
        #     wld_list.append(wld)
        # for i, wl in enumerate(wld_list):
        #     self._add_technique('D{0}'.format(i + 1), wl, 2, x_axis=np.arange(wl.size))
            return super().run(data)


class WaveletHistoryTechnique(Technique):
    def run(self, data) -> list:
        close_all = data.close.values
        wavelet = 'db2'
        level = 4
        wla = [np.nan]
        for i in range(close_all.size):
            close = close_all[:i+1]
            wp = pywt.WaveletPacket(close, wavelet, maxlevel=level)
            new_wp = pywt.WaveletPacket(None, wavelet, maxlevel=level)
            node = 'a'*level
            new_wp[node] = wp[node]
            wl = new_wp.reconstruct()
            wla.append(wl[i])
        self._add_technique('WLA', wla, x_axis=np.arange(len(wla)))
        return super().run(data)


class LineTechnique(Technique):
    def _evaluate(self):
        pass

    def run(self, data) -> list:
        index = np.arange(data.index.size)
        close = data.close.values
        start = 0

        return super().run(data)
