import math
import tsfresh
# import logging
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
from distribution import get_cdf
import shutil
from ftrl import FTRL
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_size, h_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(  # Sequential组合结构
            nn.Linear(in_features=input_size, out_features=h_size),
            nn.ReLU(),
            nn.Linear(in_features=h_size, out_features=int(h_size * 0.5)),
        )
        # self.out = nn.Linear(h_size, 12)  # torch.nn.Linear（
        # in_features：int，out_features：int，bias：bool = True ）

        self.out = nn.Sequential(  # Sequential组合结构
            nn.ReLU(), nn.Linear(int(h_size * 0.5), 12))

    def forward(self, x):
        # Set initial hidden and cell states

        r_out = self.fc(x)
        outs = self.out(r_out)

        return outs


class Pre():
    def __init__(self, train_path, val_path, n_features, n_lookback, n_fowards, scaler=None):
        self.path_train = train_path
        self.path_val = val_path
        self.path_m = 2
        self.feature = n_features
        self.n_lookback = n_lookback
        self.n_fowards = n_fowards
        self.n_obs = n_lookback + n_fowards
        self.scaler = scaler

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
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

    def get_train(self):

        samples = list()
        data_path = self.path_train
        # 'F://MH//Data//experiment//data//val_s//'

        for root, dirs, files in os.walk(data_path):
            for file in files:
                # #获取文件所属目录
                # print(root)
                # #获取文件路径
                # print(os.path.join(root, file))
                filepath = os.path.join(root, file)

                data = np.load(filepath, allow_pickle=True)
                # data = data.astype(np.float32)
                value = data.astype(np.float32)

                sample = self.series_to_supervised(value, self.n_obs - 1, 1)
                sample = sample.values
                samples.append(sample)

        value = np.concatenate(samples, axis=0)
        value = value.reshape(-1, self.feature)
        scaler = MinMaxScaler(feature_range=(0, 0.99))
        self.scaler = scaler
        scaled = scaler.fit_transform(value)
        dataset = scaled.reshape(-1, self.feature * (self.n_obs))

        return dataset, self.scaler

    def get_val(self):

        samples = list()
        data_path = self.path_val
        # 'F://MH//Data//experiment//data//val_s//'

        for root, dirs, files in os.walk(data_path):
            for file in files:
                # #获取文件所属目录
                # print(root)
                # #获取文件路径
                # print(os.path.join(root, file))
                filepath = os.path.join(root, file)

                data = np.load(filepath, allow_pickle=True)
                # data = data.astype(np.float32)
                value = data.astype(np.float32)

                sample = self.series_to_supervised(value, self.n_obs - 1, 1)
                sample = sample.values
                samples.append(sample)

        value = np.concatenate(samples, axis=0)
        value = value.reshape(-1, self.feature)

        scaled = self.scaler.fit_transform(value)
        val_data = scaled.reshape(-1, self.feature * (self.n_obs))
        return val_data


if __name__ == "__main__":

    val_path = 'F://MH//Data//experiment//data//val_s//'
    train_path = 'F://MH//Data//experiment//data//train_s//'

    data_path = 'F://MH//Data//experiment//data//train_single//'
    result_path = 'C://Users//MH//Desktop//result//'
    model_subpath = 'F://MH//Data//experiment//MLP_all//'

    n_features, n_fowards = 14, 5
    H_SIZE = 128
    INPUT_SIZE = 14  # MLP 的输入维度
    Batch_size = 128
    _, my_scaler = Pre(train_path=train_path,
                            val_path=val_path,
                            n_features=n_features,
                            n_fowards=n_fowards,
                            n_lookback=30).get_train()


    for n_lookback in [10,20,30,40,50,60,70,80,90]:
        for n_fowards in [5,10,15]:
            n_obs = n_lookback + n_fowards

            avg_metric =[]
            for root, dirs, files in os.walk(val_path):
                for dir in dirs:
                    # #获取文件所属目录
                    # print(root)
                    # #获取文件路径
                    # print(os.path.join(root, file))


                    filepath = os.path.join(root, dir)
                    print(filepath)

                    data_generate = Pre(train_path=train_path,
                                        val_path=filepath,
                                        n_features=n_features,
                                        n_fowards=n_fowards,
                                        n_lookback=n_lookback,
                                        scaler = my_scaler)

                    val_data = data_generate.get_val()




                    model = MLP(input_size=INPUT_SIZE * n_lookback,
                                h_size=H_SIZE).to(DEVICE)  # 加载模型到对应设备
                    model_path = model_subpath + str(n_lookback) + "rnn_" + str(n_fowards) + ".pth"
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    train_x = torch.from_numpy(val_data[:, 0:n_features * n_lookback])
                    train_y = torch.from_numpy(val_data[:, -n_features:])


                    train_x = train_x.to(DEVICE)


                    with torch.no_grad():
                        predicted = model(train_x).cpu().data.numpy()

                    predict_value = my_scaler.inverse_transform(np.concatenate((predicted, train_y[:, -2:]), axis=1))[:,:-2]
                    true_value = my_scaler.inverse_transform(train_y)[:,:-2]
                    # plt.plot(true_value[:600,0])
                    # plt.plot(predict_value[:600,0])
                    # plt.show()


                    rmse = math.sqrt(mean_squared_error(predict_value, true_value))
                    MAPE = np.mean(abs(predict_value - true_value) / true_value)  * 100

                    index =[0,3,6,9]
                    rmse1 = math.sqrt(mean_squared_error(predict_value[:,index], true_value[:,index]))
                    MAPE1 = np.mean(abs(predict_value[:,index] - true_value[:,index]) / true_value[:,index])  * 100
                    avg_metric.append([rmse,MAPE,rmse1,MAPE1])

            #
            avg_r, avg_m,avg_r1,avg_m1 = np.mean(avg_metric,axis=0)
            result = {"n_look":n_lookback,"n_fowards":n_fowards,
                      'RMSE': avg_r, 'MAPE': avg_m,
                      'RMSE1': avg_r1, 'MAPE1': avg_m1}

            df = pd.DataFrame(result,index=[0])

            df.to_csv(result_path + 'score_onval.csv',  mode='a', line_terminator="\n", index =None, header=False)
            print(result)



            run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
