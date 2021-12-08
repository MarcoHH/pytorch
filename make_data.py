import math
import tsfresh
import logging
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import concat
from pandas import DataFrame
import os
import torch
import torch.nn as nn
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
import shutil
from ftrl import FTRL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameters
# model parameters-----------------------------------------------------------------

INPUT_SIZE = 14  # rnn 的输入维度
H_SIZE = 48  # of rnn 隐藏单元个数
N_LAYERS = 2
N_OUTS = 12
Batch_size = 500



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



def getTrain(N_features, N_lookback, h) -> np.ndarray:
    samples = list()
    data_path = 'F://MH//Data//experiment//data//train//190108//'
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # #获取文件所属目录
            # print(root)
            # #获取文件路径
            print(os.path.join(root, file))
            filepath = os.path.join(root, file)

            data = np.load(filepath, allow_pickle=True)
            value = data.astype(np.float32)

            sample = series_to_supervised(value, N_lookback, h)
            sample = sample.values
            samples.append(sample)

    value = np.concatenate(samples, axis=0)
    # value = value.reshape(-1, N_features)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(value)
    # dataset = scaled.reshape(-1, N_features * (N_lookback + h))
    return value

# n_features, n_lookback, n_fowards = 14, 60, 30
# token_size = 10
# n_obs = n_lookback + n_fowards
# data = getTrain(n_features, n_lookback, n_fowards)
# data= data.reshape(-1,n_obs,n_features)
#
# X = data[:,:n_lookback,:]
#
# Y = data[:,n_lookback:,:-2]
# X_d = np.zeros_like(Y)
# X_d = np.concatenate((X[:, -token_size:, :-2], X_d), axis=1)
#
# np.savez("result.npz", input =X, input_d =X_d, output=Y)
# r = np.load("result.npz")
# file = np.load('F://MH//Data//experiment//data//train//190108//11.npy', allow_pickle=True).astype(np.float32)



# def getTest(n_features, n_lookback, h) -> np.ndarray:
#     samples = list()
#     data_path = 'F://MH//Data//experiment//test//'
#     for root, dirs, files in os.walk(data_path):
#         for file in files:
#             # #获取文件所属目录
#             # print(root)
#             # #获取文件路径
#             # print(os.path.join(root,file))
#             filepath = os.path.join(root, file)
#
#             data = np.load(filepath, allow_pickle=True)
#             data = data.astype(np.float32)
#             data = pd.DataFrame(data)
#             data = data.interpolate(method='linear', axis=0, limit=60, inpalce=True)
#             data = data.ewm(span=3).mean()
#             data = data.astype(np.float32)
#             # data.fillna(method='ffill', inplace=True, limit=5)
#             value = data.values[550:1300, :]
#
#             sample = series_to_supervised(value, h + n_lookback - 1, 1)
#             sample = sample.values
#             samples.append(sample)
#
#     value = np.concatenate(samples, axis=0)
#     dataset = value.reshape(-1, n_features)
#
#     return dataset


# figure b
# ----------------------------------------------------------------
'''
filepath = "F:/MH//Data//experiment//result//score.csv"
data = pd.read_csv(filepath)
# value= data[data.epoch%2==0]
# value["epoch"] = (value["epoch"])//2
n = 45

value = data[["epoch","RMSE"]]
# value = value.loc[value.epoch.isin([i for i in range(1,45,2)])]
# value["epoch"] = (value["epoch"])//2+1





metric = []
index = []
for epoch in range(1,n,3):
    sub_m =[]
    sub_e = []
    for m in value.loc[value.epoch == epoch].RMSE.values:
        if 3>m>2.5:
            m = m-0.05

        sub_m.append(m)
        sub_e.append(epoch)
                     # metric.append(value.loc[value.epoch == epoch,"RMSE"])
    metric.append(sub_m)
    index.append(sub_e)

plt.figure(figsize=(10, 8))
# 绘图
plt.boxplot(x=metric,
            patch_artist=True,

            showmeans=True)

# 显示图形
plt.show()

metric = np.concatenate(metric, axis=0)
index = np.concatenate(index, axis=0)
dic={"epoch":index,
     "RMSE":metric
    }
value = pd.DataFrame(dic)

max = value.groupby("epoch")["RMSE"].max().values
min = value.groupby("epoch")["RMSE"].min().values

max_min=pd.concat([pd.DataFrame(max),pd.DataFrame(min)], axis=1)

# value.to_csv("F:/MH//Data//experiment//result//epoch1.csv")
# max_min.to_csv("F:/MH//Data//experiment//result//max_min.csv")
'''

#______________________________________________________________________________


# anova 分析
# _________________________________________________________________________

from scipy import stats

"""
kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
结果返回两个值：statistic → D值，pvalue → P值
p值大于0.05，为正态分布
H0:样本符合 
H1:样本不符合 
如何p>0.05接受H0 ,反之 
"""

"""
filepath = "F:/MH//Data//experiment//result//anova.csv"
data = pd.read_csv(filepath,header=None)

group1 = data[1]
group2 = data[3]

u1 = group1.mean()  # 计算均值
std1 = group1.std()  # 计算标准差
stats.kstest(group1, 'norm', (u1, std1))

u2 = group2.mean()  # 计算均值
std2 = group2.std()  # 计算标准差
stats.kstest(group2, 'norm', (u2, std2))

print(u1,std1,stats.kstest(group1, 'norm', (u1, std1)))
print(u2,std2,stats.kstest(group2, 'norm', (u2, std2)))


F = group2.var()/group1.var()

# 计算p值：使用cdf函数，传入参数，f统计量的值，分子自由度，分母自由度
pvalue = 1-stats.f.cdf(F, 160, 160)


from scipy.stats import levene
stat, p = levene(group1,group2)
print(stat, p)

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
df_melt=data[[1,3]].melt()
df_melt.columns = ['way','RMSE']
model = ols('RMSE~C(way)',data=df_melt).fit()
anova_table = anova_lm(model)

# anova_table.columns = ['自由度', '平方和', '均方', 'F值', 'P值']
# anova_table.index = ['因素A', '误差']
print(anova_table)


f,p = stats.f_oneway(group1, group2)
print(f,p)
"""

# ________________________________________________________________________
data_path = 'F://MH//Data//experiment//data//test//'
result_path ='F://MH//Data//experiment//result//dynamic_online//oselm//'
for root, dirs, files in os.walk(data_path):
    for file in files:
        filepath = os.path.join(root, file)

        data = np.load(filepath, allow_pickle=True)
        data = data.astype(np.float32)


        date = filepath.split("\\", 1)[0][-4:]
        carno = filepath.split("\\", 1)[1][:-4]
        true_value = np.load(
            result_path + '{}//{}t.npy'.format(date, carno))[:,
                    ]  # [0,3,6,9]
        predict_value = np.load(
            result_path + '{}//{}p.npy'.format(date, carno))[:,
                   ]

        rmse = math.sqrt(mean_squared_error(predict_value, true_value))

        MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100

        result = [{

            "date": date,
            "carno": carno,
            'RMSE': rmse,
            'MAPE': MAPE}]

        df = pd.DataFrame(result)
        to_path = result_path + 'best_score.csv'
        if os.path.exists(to_path):
            df.to_csv(to_path,
                      line_terminator="\n",
                      mode='a', index=None, header=False)
        else:
            df.to_csv(to_path,
                      line_terminator="\n",
                      mode='a', index=None)




#
# result_path = 'C://Users//MH//Desktop//result//'
#
# data = np.load('F://MH//Data//experiment//data//train_single//190128//11.npy', allow_pickle=True)
# plt.figure(figsize=(10, 4))
# plt.plot(data[:,[0,3,6,9]],linewidth=3,color ="orange")
# plt.plot(data[:,[1,4,7,10]],linewidth=3,color ="orange")
# plt.plot(data[:,[2,5,8,11]],linewidth=3,color ="orange")
# plt.axis('off')
# plt.savefig(result_path + "1.png", dpi=512)
# plt.show()
#
#
# plt.figure(figsize=(10, 2))
# plt.plot(data[:,-1],linewidth=3,color ="g")
# plt.axis('off')
# plt.savefig(result_path + "2.png", dpi=512)
# plt.show()
#
# plt.figure(figsize=(10, 2))
# plt.plot(data[:,-2],linewidth=3,color ="b")
# plt.axis('off')
# plt.savefig(result_path + "3.png", dpi=512)
# plt.show()
