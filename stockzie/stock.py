import stockzie as sz
import numpy as np


class Stocks:
    def __init__(self, codes, start=None, end=None, ktype='D', source='tushare') -> None:
        self.codes = sz.data.to_list(codes)
        self.datas = sz.data.get(self.codes, start, end, ktype)
        self.stocks = []
        for code, data in zip(self.codes, self.datas):
            self.stocks.append(Stock(code, data))

        self.trading_summary = TradingSummary()

    def __getitem__(self, item):
        return self.stocks[item]

    def plot_kv(self, item=None, rows=2):
        if item is not None:
            stocks = self.stocks[item]
        else:
            stocks = self.stocks
        fig_list, ax_list = [], []
        max_row = rows - 1
        for stock in stocks:
            max_row = max(max_row, stock.get_technique_max_row())
        for stock in stocks:
            fig, axes = sz.plot.init_fig_axes(max_row + 1, stock.data)
            fig.suptitle(stock.code)
            sz.plot.kline(axes[0], data=stock.data)
            sz.plot.volume(axes[1], data=stock.data)
            axes_twin = []
            for i in axes:
                ax_twin = i.twinx()
                axes_twin.append(ax_twin)
            stock.plot_technique(axes, axes_twin)
            for i in axes:
                i.legend()
            for i in axes_twin:
                if len(i.lines) == 0:
                    # i.set_ylim([])
                    i.set_yticks([])
                    i.set_yticklabels([])
            fig_list.append(fig)
            ax_list.append(axes)
        return fig_list, ax_list

    def add_technique(self, Technique):
        for stock in self.stocks:
            tech = Technique()
            stock.add_technique(tech.run(stock.data))

    def add_tech_strategy(self, TechStrategy):
        for stock in self.stocks:
            tech_strategy = TechStrategy(stock.code)
            stock.add_tech_strategy(tech_strategy, stock.data)
        self.calc_stocks_trading()

    def calc_stocks_trading(self):
        trading_summary = self.trading_summary
        for tech_strategy in self.stocks[0].tech_strategies:
            trading_summary.stg_names.append(tech_strategy.get_name())

        profit_list = []
        for stock in self.stocks:
            profit_list.append([])
            for tech_strategy in stock.tech_strategies:
                trading = tech_strategy.trading
                profit_list[-1].append(trading.total - trading.init_total)

        trading_summary.calc_dt['profit'] = np.array(profit_list).T

    def stocks_trading_summary(self):
        for k, v in self.trading_summary.sum_dt.items():
            print('{0}: {1}')


class Stock:
    def __init__(self, code, data) -> None:
        self.code = code
        self.data = data
        self.techniques = []
        self.tech_strategies = []

    def add_technique(self, techniques):
        if isinstance(techniques, list):
            self.techniques.extend(techniques)
        elif isinstance(techniques, dict):
            self.techniques.extend(techniques.values())

    def add_tech_strategy(self, tech_strategy, data):
        self.add_technique(tech_strategy.run(data))
        self.tech_strategies.append(tech_strategy)

    def plot_technique(self, axes, axes_twin):
        x_axis_all = np.arange(self.data.index.size)
        for technique in self.techniques:
            if technique['x_axis'] is None:
                x_axis = x_axis_all
            else:
                x_axis = technique['x_axis']
            ax_sel = axes[technique['row']]
            if technique['twin']:
                ax_twin = axes_twin[technique['row']]
                xlim = ax_sel.get_xlim()
                xticks = ax_sel.get_xticks()
                xticklabels = ax_sel.get_xticklabels()
                ax_twin.plot(
                    x_axis, technique['value'], technique['style'],
                    label=technique['name'], alpha=technique['alpha'], linewidth=technique['width'])
                ax_sel.set_xlim(xlim)
                ax_sel.set_xticks(xticks)
                ax_sel.set_xticklabels(xticklabels)
            else:
                ax_sel.plot(
                    x_axis, technique['value'], technique['style'],
                    label=technique['name'], alpha=technique['alpha'], linewidth=technique['width'])

    def get_technique_max_row(self):
        max_row = 0
        for technique in self.techniques:
            max_row = max(max_row, technique['row'])
        return max_row


class TradingSummary:
    def __init__(self) -> None:
        self.stg_names = []
        self.calc_dt = {}
        self.sum_dt = {}