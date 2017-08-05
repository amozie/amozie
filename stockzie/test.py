import numpy as np
import matplotlib.pyplot as plt
import plot as szp
import seaborn as sns
import tushare as ts

import talib as tl

import demo
import data as szd
import plot as szp

# data = demo.plotkv('600056')

code='600056'
ktype='D'

data = szd.get(code, ktype=ktype)
fig, ax = szp.init_fig_axes(2, data)

ax1, ax2 = ax

szp.kline(ax1, data=data)
szp.volume(ax2, data=data)

#px = np.arange(data.index.size)
#
#ax1.plot(px, tl.MA(data.close.values, 5), 'r', label='5')
#ax1.plot(px, tl.MA(data.close.values, 20), 'b', label='20')
#ax1.plot(px, tl.MA(data.close.values, 30), 'm', label='30')

#ax1.set_xticklabels([])

#ax1.legend()

plt.show()
