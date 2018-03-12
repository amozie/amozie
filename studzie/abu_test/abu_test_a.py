import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import abupy

abupy.env.disable_example_env_ipython()

from abupy import ABuSymbolPd

df = ABuSymbolPd.make_kl_df('601398')

from abupy import EMarketDataFetchMode, abu

abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop

read_cash = 100000
stock_pickers = None
buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak}, {'xd': 42, 'class': AbuFactorBuyBreak}]
sell_factors = [{'stop_loss_n': 1.0, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop},
                {'pre_atr_n': 1.5, 'class': AbuFactorPreAtrNStop},
                {'close_atr_n': 1.5, 'class': AbuFactorCloseAtrNStop}]

# from abupy import EMarketTargetType
from abupy import ABuMarket

choice_symbols = ABuMarket.K_SAND_BOX_CN
abupy.env.enable_example_env_ipython()
abu_result_tuple, _ = abu.run_loop_back(
    read_cash, buy_factors, sell_factors, stock_pickers,
    choice_symbols=choice_symbols, n_folds=2)
abupy.env.disable_example_env_ipython()

from abupy import AbuMetricsBase

metrics = AbuMetricsBase(*abu_result_tuple)
metrics.fit_metrics()
metrics.plot_returns_cmp()

from abupy import AbuFactorBuyBase