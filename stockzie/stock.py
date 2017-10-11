import stockzie as sz
import numpy as np


class Stocks:
    def __init__(self, codes, start=None, end=None, ktype='D', source='tushare') -> None:
        self.codes = sz.data.to_list(codes)
        self.datas = sz.data.get(self.codes, start, end, ktype)
        self.stocks = []
        for code, data in zip(self.codes, self.datas):
            self.stocks.append(Stock(code, data))

    def __getitem__(self, item):
        return self.stocks[item]

    def plot_kv(self, item=None, rows=2):
        if item is not None:
            stocks = self.stocks[item]
        else:
            stocks = self.stocks
        figs, axes = [], []
        max_row = rows - 1
        for stock in stocks:
            max_row = max(max_row, stock.get_technique_max_row())
        for stock in stocks:
            fig, ax = sz.plot.init_fig_axes(max_row + 1, stock.data)
            fig.suptitle(stock.code)
            sz.plot.kline(ax[0], data=stock.data)
            sz.plot.volume(ax[1], data=stock.data)
            stock.plot_technique(ax)
            for i in ax:
                i.legend()
            figs.append(fig)
            axes.append(ax)
        return figs, axes

    def add_technique(self, Technique):
        for stock in self.stocks:
            tech = Technique()
            stock.add_technique(tech.run(stock.data))

    def add_tech_strategy(self, TechStrategy):
        for stock in self.stocks:
            strategy = TechStrategy()
            stock.add_technique(strategy.run(stock.data))


class Stock:
    def __init__(self, code, data) -> None:
        self.code = code
        self.data = data
        self.techniques = []

    def add_technique(self, techniques):
        if isinstance(techniques, list):
            self.techniques.extend(techniques)
        elif isinstance(techniques, dict):
            self.techniques.extend(techniques.values())

    def plot_technique(self, ax):
        x_axis_all = np.arange(self.data.index.size)
        for technique in self.techniques:
            if technique['x_axis'] is None:
                x_axis = x_axis_all
            else:
                x_axis = technique['x_axis']
            ax_sel = ax[technique['row']]
            if technique['twin']:
                xlim = ax_sel.get_xlim()
                xticks = ax_sel.get_xticks()
                xticklabels = ax_sel.get_xticklabels()
                ax_twin = ax_sel.twinx()
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

