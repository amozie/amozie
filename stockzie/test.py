import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts

import talib as tl

import stockzie as sz

def test1():
    if 'data' not in locals().keys():
        data = None
    data = sz.demo.plotkv('600056', data)

def test2():
    code='600056'
    ktype='D'
    
    if 'data' not in locals().keys():
        data = sz.data.get(code, ktype=ktype)
    fig, ax = sz.plot.init_fig_axes(3, data)
    
    ax1, ax2, ax3 = ax
    
    sz.plot.kline(ax1, data=data)
    sz.plot.volume(ax2, data=data)
    
    macd = tl.MACD(data.close.values)
    macd = np.array(macd)
    
    idx = iter([1,2,3])
    clr = iter(['r','g','b'])
    for i in macd:
        ax3.plot(np.arange(data.index.size), i, color=next(clr), label=next(idx))
    ax3.legend()
    
    px = np.arange(data.index.size)
    
    ax1.plot(px, tl.MA(data.close.values, 5), 'r', label='5')
    ax1.plot(px, tl.MA(data.close.values, 20), 'b', label='20')
    ax1.plot(px, tl.MA(data.close.values, 30), 'm', label='30')
    
    ax1.legend()
    
    plt.show()

def test3():
    data = sz.data.get('600056')
    strategy = sz.strategy.Ma520Strategy(data.tail(200))
    strategy.run()
    strategy.plot_demo()

if __name__ == '__main__':
    test3()
