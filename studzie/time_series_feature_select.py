import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.feature_selection import RFE

# 加载数据
series = pd.Series.from_csv('./dataset/monthly-car-sales-in-quebec-1960.csv', header=0)
# 平稳化
diff = series.diff(12)[12:]

# 自相关图
plot_acf(diff)
plot_pacf(diff)

# 创建一系列滞后数据
df = pd.DataFrame()
df['t'] = diff
for i in range(1, 13):
    df['t-{0}'.format(str(i))] = diff.shift(i)
df = df.iloc[12:, :]

# 随机森林计算特征重要性
X = df.values[:, 1:]
y = df.values[:, 0]
model = rfr(500, random_state=1)
model.fit(X, y)
fi = model.feature_importances_
plt.bar(np.arange(1, fi.size+1), fi)

# RFE选择特征
rfe = RFE(rfr(500, random_state=1), 4)
fit = rfe.fit(X, y)
print(df.columns[1:][fit.support_])
plt.bar(np.arange(1, fit.support_.size+1), fit.support_)
plt.bar(np.arange(1, fit.ranking_.size+1), fit.ranking_)
