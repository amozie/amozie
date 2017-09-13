# -*- coding: utf-8 -*-

import six
import timeit
from quantdigger import *
from quantdigger.digger.analyze import AnalyzeFrame
from quantzie.strategy.common import TEST

SIZE = 10


class Stg1(Strategy):
    def on_init(self, ctx):
        ctx.test = TEST(ctx.close, 1, 'test', 'y', 2)

    def on_bar(self, ctx):
        if ctx.curbar == 1:
            ctx.buy(ctx.close, 1)
        else:
            ctx.sell(ctx.open, 1)
            ctx.buy(ctx.close, 1)

    def on_exit(self, ctx):
        pass


if __name__ == '__main__':
    start = timeit.default_timer()
    #ConfigUtil.set(source='tushare')
    ConfigUtil.set(source='cached-tushare',
                   cache_path='E:/_cache_tushare')
    set_symbols(['600096.SH-1.Day'], '2015-01-04', '2016-01-08')
    # set_symbols(['BB.SHFE-1.Minute'])
    profile = add_strategy([Stg1('S1')], {'capital': 500000.0})
    run()
    stop = timeit.default_timer()
    six.print_('using time: %d seconds' % (stop - start))

    from quantdigger.digger import finance, plotting
    s = 0
    curve0 = finance.create_equity_curve(profile.all_holdings(0))
    # AnalyzeFrame(profile)
    plotting.plot_strategy(profile.data(0), profile.technicals(0),
                           profile.deals(0), curve0.equity.values,
                           profile.marks(0))
