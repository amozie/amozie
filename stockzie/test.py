import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts

import talib as tl

import stockzie as sz

# data = demo.plotkv('600056')

code='600056'
ktype='D'

data = sz.data.get(code, ktype=ktype)
fig, ax = sz.plot.init_fig_axes(2, data)

ax1, ax2 = ax

sz.plot.kline(ax1, data=data)
sz.plot.volume(ax2, data=data)

#px = np.arange(data.index.size)
#
#ax1.plot(px, tl.MA(data.close.values, 5), 'r', label='5')
#ax1.plot(px, tl.MA(data.close.values, 20), 'b', label='20')
#ax1.plot(px, tl.MA(data.close.values, 30), 'm', label='30')

#ax1.set_xticklabels([])

#ax1.legend()

plt.show()
