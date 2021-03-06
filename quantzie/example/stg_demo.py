# -*- coding: utf-8 -*-

import six
import timeit
from quantdigger import *
from quantdigger.digger.analyze import AnalyzeFrame
import matplotlib.pyplot as plt

from quantzie.strategy.common import *

SIZE = 10


class Q(object):
    def __init__(self, size):
        self.__q = []
        self.__size = size

    def put(self, x):
        if self.is_full():
            self.__q.pop(0)
        self.__q.append(float(x))

    def max(self):
        return max(self.__q)

    def min(self):
        return min(self.__q)

    def is_full(self):
        return len(self.__q) == self.__size

    def __str__(self):
        return str(self.__q)


class Stg1(Strategy):
    def on_init(self, ctx):
        self.q = Q(SIZE - 1)

    def on_bar(self, ctx):
        if self.q.is_full():
            if ctx.pos() == 0:
                if ctx.close > self.q.max():
                    ctx.buy(ctx.close, 1)
            else:
                if ctx.close < self.q.min():
                    ctx.sell(ctx.close, ctx.pos())
        self.q.put(ctx.close)

    def on_exit(self, ctx):
        pass

class Stg2(Strategy):
    def on_init(self, ctx):
        ctx.test = TEST(ctx.close, 'test')

    def on_bar(self, ctx):
        if ctx.curbar == 1:
            ctx.buy(ctx.close, 1)
        elif ctx.curbar == 50:
            ctx.sell(ctx.close, 1)

    def on_exit(self, ctx):
        pass


if __name__ == '__main__':
    start = timeit.default_timer()
    #ConfigUtil.set(source='tushare')
    ConfigUtil.set(source='cached-tushare',
                   cache_path='E:/_cache_tushare')
    set_symbols(['600096.SH-1.Day'], '2000-1-1', '2016-4-1')
    # set_symbols(['BB.SHFE-1.Minute'])
    profile = add_strategy([Stg2('S1')], {'capital': 500000.0})
    run()
    stop = timeit.default_timer()
    six.print_('using time: %d seconds' % (stop - start))

    from quantdigger.digger import finance, plotting
    s = 0
    curve0 = finance.create_equity_curve(profile.all_holdings(0))
    AnalyzeFrame(profile)
    plotting.plot_strategy(profile.data(0),
                           {1: profile.technicals(0)},
                           profile.deals(0), curve0.equity.values,
                           profile.marks(0))
    plt.show()
