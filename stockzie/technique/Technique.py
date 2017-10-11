import numpy as np


MAX_DRAWDOWN = 0.1


class Technique():
    def __init__(self):
        self.techniques = []

    def run(self, data) -> list:
        pass

    def get(self):
        return self.techniques

    def clear(self):
        self.techniques = []

    def _add_technique(self, name, value, row=0, style='', width=None, alpha=None, x_axis=None, twin=False):
        self.techniques.append(
            {
                'name': name,
                'value': value,
                'row': row,
                'style': style,
                'width': width,
                'alpha': alpha,
                'x_axis': x_axis,
                'twin': twin
            }
        )


def cal_trade_asset(data, signal, only_trading=False, stop=1):
    close = data.close.values
    high = data.high.values
    low = data.low.values
    if close.size != len(signal):
        return
    total = 1
    asset = []
    trade = []
    position = False
    max_price = 0
    stop_price = 0
    is_stop = False
    for i in range(len(signal)):
        if not position:
            if signal[i] == 1:
                position = True
                trade.append(1)
                asset.append(total)
                max_price = close[i]
            else:
                trade.append(0)
                asset.append(np.nan if only_trading else total)
        else:
            if stop == 1:
                is_stop, max_price, stop_price = __cal_stop_1(max_price, high[i-1], low[i])
            if signal[i] == -1 or is_stop:
                position = False
                is_stop = False
                trade.append(-1)
            else:
                trade.append(0)
            total = total*(stop_price if is_stop else close[i])/close[i-1]
            asset.append(total)
    return trade, asset


def __cal_stop_1(max_price, last_high_price, low_price):
    max_price = max(max_price, last_high_price)
    stop_price = max_price*(1-MAX_DRAWDOWN)
    return low_price <= stop_price, max_price, stop_price
