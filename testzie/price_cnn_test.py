import numpy as np
import tushare as ts

code = '002771'

data_5 = ts.get_k_data(code, ktype='5')
data_15 = ts.get_k_data(code, ktype='15')
data_30 = ts.get_k_data(code, ktype='30')
data_60 = ts.get_k_data(code, ktype='60')
data_d = ts.get_k_data(code, ktype='D')
data_w = ts.get_k_data(code, ktype='W')

