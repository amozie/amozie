import stockzie as sz


class Stocks():
    def __init__(self, codes, start=None, end=None, ktype='D', source='tushare') -> None:
        super().__init__()
        self.codes = sz.data.to_list(codes)
        self.datas = sz.data.get(self.codes, start, end, ktype)

    def __getitem__(self, item):
        return self.datas[item]

    def plot_kv(self, item=None, row=2):
        if item is not None:
            codes = self.codes[item]
            datas = self.datas[item]
        else:
            codes = self.codes
            datas = self.datas
        figs, axes = [], []
        for code, data in zip(codes, datas):
            fig, ax = sz.plot.init_fig_axes(row, data)
            fig.suptitle(code)
            ax1, ax2 = ax
            sz.plot.kline(ax1, data=data)
            sz.plot.volume(ax2, data=data)
            figs.append(fig)
            axes.append(ax)
        return figs, axes
