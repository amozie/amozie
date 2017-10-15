import six
import timeit
from quantdigger import *
from quantdigger.digger import finance, plotting
from quantdigger.digger.analyze import AnalyzeFrame
from quantzie.strategy.common import *
import matplotlib.pyplot as plt


class Stg1(Strategy):
    def on_init(self, ctx):
        ctx.ma = MAZ(ctx.close, 6, 'ma', 'g', 1)

    def on_bar(self, ctx):
        if ctx.pos() == 0 and (ctx.open <= ctx.ma < ctx.close or (ctx.open > ctx.ma > ctx.low and ctx.close > ctx.ma)):
            q = ctx.cash() // float(ctx.ma)
            ctx.buy(ctx.ma, q)
        elif ctx.pos() > 0 and (ctx.open >= ctx.ma > ctx.close or (ctx.open < ctx.ma < ctx.high and ctx.close < ctx.ma)):
            ctx.sell(ctx.ma, ctx.pos())

    def on_exit(self, ctx):
        pass


if __name__ == '__main__':
    start = timeit.default_timer()
    ConfigUtil.set(source='cached-tushare',
                   cache_path='E:/_cache_tushare')
    set_symbols(['600056.SH-1.Day'], '2016-09-01')
    profile = add_strategy([Stg1('Stg1')], {'capital': 100000.0})
    run()
    stop = timeit.default_timer()
    six.print_('using time: %d seconds' % (stop - start))

    curve0 = finance.create_equity_curve(profile.all_holdings(0))
    curve = finance.create_equity_curve(profile.all_holdings())
    AnalyzeFrame(profile)
    plotting.plot_strategy(profile.data(0),
                           {
                               1: [profile.technicals(0)]
                           },
                           profile.deals(0), curve.equity.values,
                           profile.marks(0))
    # plotting.plot_curves([curve.networth])
    six.print_(finance.summary_stats(curve, 252))
    plt.show()
