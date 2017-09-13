from quantdigger.technicals import TechnicalBase
from quantdigger.technicals.base import tech_init, ndarray
from quantdigger.technicals.techutil import register_tech


@register_tech('TEST')
class TEST(TechnicalBase):
    @tech_init
    def __init__(self, data, n, name='TEST',
                 style='y', lw=1):
        super(TEST, self).__init__(name)
        self._args = [ndarray(data), n]

    def _rolling_algo(self, data, n, i):
        return (data[i], )

    def _vector_algo(self, data, n):
        self.values = data

    def plot(self, widget):
        self.widget = widget
        self.plot_line(self.values, self.style, lw=self.lw)