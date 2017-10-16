import numpy as np
import pywt

from stockzie.strategy.TechStrategy import *
from stockzie.strategy.tech_common import *


def calc_wavelet(close, wavelet='db2', level=4):
    wp = pywt.WaveletPacket(close, wavelet, maxlevel=level)
    new_wp = pywt.WaveletPacket(None, wavelet, maxlevel=level)
    node = 'a' * level
    new_wp[node] = wp[node]
    wl = new_wp.reconstruct()
    try:
        wla = wl[close.size]
        wla_0 = wl[close.size - 1]
    except IndexError:
        wla = wl[close.size - 1]
        wla_0 = wl[close.size - 2]

    direct = wla - wla_0
    if direct > 0:
        direct = 1
    elif direct < 0:
        direct = -1
    else:
        direct = 0
    return wla, direct


class WaveHisTechStrategy(TechStrategy):
    def _handle_trading(self, data):
        wavelet = 'db2'
        level = 6
        close = self.data_hist.close.values[:-1]
        high = self.data_hist.high.values[:-1]
        low = self.data_hist.low.values[:-1]
        wl_c = np.nan
        wl_h = np.nan
        wl_l = np.nan
        direct = 0
        if self.iter > 1:
            wl_c, direct = calc_wavelet(close, wavelet, level)
            wl_h = calc_wavelet(high, wavelet, level)[0]
            wl_l = calc_wavelet(low, wavelet, level)[0]
        self._add_technique_iter('wl_c', wl_c)
        self._add_technique_iter('wl_h', wl_h)
        self._add_technique_iter('wl_l', wl_l)
        self._add_technique_iter('direct', direct, 3)

        if self.iter > 1:
            stg_price_cross_trend(self, [wl_c, direct])
            # stg_price_cross_trend(self, wl_c)

    def _end_trading(self, data):
        close = data.close.values
        wla, direct = calc_wavelet(close)
        print('predict: {0}, {1}'.format(wla, direct))