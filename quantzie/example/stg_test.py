# -*- coding: utf-8 -*-

import six
import timeit
from quantdigger import *
from quantdigger.digger.analyze import AnalyzeFrame
from quantzie.strategy.common import *

SIZE = 10


class Stg1(Strategy):
    def on_init(self, ctx):
        ctx.test = TEST(ctx.close, 'test')

    def on_bar(self, ctx):
        pass

    def on_exit(self, ctx):
        pass


class Stg2(Strategy):
    def on_init(self, ctx):
        ctx.macd = MACD(ctx.close, 5)

    def on_bar(self, ctx):
        # ctx.switch_to_pcontract('600056.SH-1.DAY')
        ctx.buy(ctx.close, 1)

    def on_exit(self, ctx):
        pass


if __name__ == '__main__':
    start = timeit.default_timer()
    # ConfigUtil.set(source='tushare')
    # ConfigUtil.set(source='cached-tushare',
    #                cache_path='E:/_cache_tushare')
    # set_symbols(['600056.SH-1.Day'], '2016-01-04', '2016-01-08')
    ConfigUtil.set(source='csv', data_path='E:/_cache_tushare')
    set_symbols(['600096.SH-1.Day'], '2015-01-04', '2016-01-08')
    profile = add_strategy([Stg2('Stg')], {'capital': 100000.0})
    run()
    stop = timeit.default_timer()
    six.print_('using time: %d seconds' % (stop - start))

    from quantdigger.digger import finance, plotting
    s = 0
    curve0 = finance.create_equity_curve(profile.all_holdings(0))
    # AnalyzeFrame(profile)
    plotting.plot_strategy(profile.data(0),
                           {
                               # 1:[profile.technicals(0)],
                               3:[profile.technicals(0)]
                           },
                           {}, curve0.equity.values,
                           profile.marks(0))
