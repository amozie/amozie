import numpy as np
import pandas as pd

lt = 'f:/lt/'

region = pd.read_csv(lt + 'region.csv',sep='\t', index_col=0)
# 排除内蒙古和西藏
# prvs = ['北京', '天津', '河北', '山东', '辽宁', '江苏', '上海', '浙江', '福建', '广东', '海南', '吉林',
#        '黑龙江', '山西', '河南', '安徽', '江西', '湖北', '湖南', '广西', '重庆', '四川', '贵州', '云南',
#        '陕西', '甘肃', '青海', '宁夏', '新疆']
prvs = ['北京', '天津', '河北', '山东', '辽宁', '江苏', '上海', '浙江', '福建', '广东', '广西', '海南',
       '吉林', '黑龙江', '山西', '河南', '安徽', '江西', '湖北', '湖南', '重庆', '四川', '贵州', '云南',
       '陕西', '甘肃', '青海', '宁夏', '新疆', '内蒙古']
years = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012']

worker = pd.read_csv(lt + 'worker.csv', sep='\t', index_col=0).join(region)
capital = pd.read_csv(lt + 'capital.csv', sep='\t', index_col=0).join(region)
energy = pd.read_csv(lt + 'energy.csv', sep='\t', index_col=0).join(region)
gdp = pd.read_csv(lt + 'gdp.csv', sep='\t', index_col=0).join(region)
co2 = pd.read_csv(lt + 'co2.csv', sep='\t', index_col=0).join(region)

table = {'劳动力': worker, '资本': capital, '能源': energy, 'GDP': gdp, 'CO2': co2}

ll = []
ll_indexs = ['劳动力', '资本', '能源', 'GDP', 'CO2']
# ll_columns = ['整体均值', '整体标准差', '东部均值', '东部标准差', '中部均值', '中部标准差', '西部均值', '西部标准差']
ll_columns = ['均值', '标准差', '最小值', '最大值']

for k, v in table.items():
    print(k)
    df = v.loc[prvs, :]

    # 整体
    val = df.loc[:, years].values.ravel()
    avg = val.mean()
    std = np.std(val, ddof=1)
    mini = val.min()
    maxi = val.max()
    # 东部
    val1 = df[df.rgn==1].loc[:, years].values.ravel()
    avg1 = val1.mean()
    std1 = np.std(val1, ddof=1)
    # 中部
    val2 = df[df.rgn==2].loc[:, years].values.ravel()
    avg2 = val2.mean()
    std2 = np.std(val2, ddof=1)
    # 西部
    val3 = df[df.rgn==3].loc[:, years].values.ravel()
    avg3 = val3.mean()
    std3 = np.std(val3, ddof=1)

    print(f'整体\n平均数{avg:.2f}\n标准差{std:.2f}')
    print(f'东部\n平均数{avg1:.2f}\n标准差{std1:.2f}')
    print(f'中部\n平均数{avg2:.2f}\n标准差{std2:.2f}')
    print(f'西部\n平均数{avg3:.2f}\n标准差{std3:.2f}')

    # ll.append([avg, std, avg1, std1, avg2, std2, avg3, std3])
    ll.append([avg, std, mini, maxi])

arr = np.array(ll)
df = pd.DataFrame(arr, ll_indexs, ll_columns)
df.to_csv(lt + 'table2_300.csv')
df.to_csv(lt + 'table6_290.csv')
df.to_csv(lt + 'table6_300.csv')



# eviews

eviews = pd.read_csv(lt + 'eviews.csv', sep='\t')
# 排除内蒙古
eviews = eviews[eviews.prv_id!=5]
# 整体
eviews.shape
des = eviews.describe()
des.to_csv(lt + 'des.csv')
# 东部
eviews = eviews[eviews.rgn=='东部']
eviews.shape
des = eviews.describe()
des.to_csv(lt + 'des.csv')

pd.Series.rank()