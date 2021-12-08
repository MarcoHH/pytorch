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
from pyoselm import OSELMRegressor, OSELMClassifier
from sklearn.datasets import load_digits, make_regression
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_size, h_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(  # Sequential组合结构
            nn.Linear(in_features=input_size,out_features=h_size),
            nn.ReLU(),
            nn.Linear(in_features=h_size, out_features = int(h_size*0.5)),
        )
        # self.out = nn.Linear(h_size, 12)  # torch.nn.Linear（
        # in_features：int，out_features：int，bias：bool = True ）

        self.out = nn.Sequential(  # Sequential组合结构
        nn.ReLU(), nn.Linear(int(h_size*0.5), 12))

    def forward(self, x):
        # Set initial hidden and cell states



        r_out = self.fc(x)

        # r_out = r_out.view(r_out.size(0), -1)
        outs = self.out(r_out)
        # outs = self.out(h_STATE[-1])

        # outs = outs[:, -1, :].view(r_out.size(0), -1)


        return outs













### GENERATE DATA ###

# Parameters
# model parameters-----------------------------------------------------------------

INPUT_SIZE = 14  # rnn 的输入维度
H_SIZE: int = 128  # of rnn 隐藏单元个数
N_LAYERS = 2
N_OUTS = 12
Batch_size = 128


# h_state = torch.zeros(N_LAYERS, 8, H_SIZE).to(DEVICE)
# c_state = torch.zeros(N_LAYERS, 8, H_SIZE).to(DEVICE)













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
    # len=0
    # n=1
    samples = list()
    data_path ='F://MH//Data//experiment//data//train_s//'


    for root, dirs, files in os.walk(data_path):
        for file in files:
            # #获取文件所属目录
            # print(root)
            # #获取文件路径
            # print(os.path.join(root, file))
            filepath = os.path.join(root, file)

            data = np.load(filepath, allow_pickle=True)

            value = data.astype(np.float32)


            sample = series_to_supervised(value, h + N_lookback - 1, 1)
            sample = sample.values
            samples.append(sample)

    value = np.concatenate(samples, axis=0)
    value = value.reshape(-1, N_features)
    scaler = MinMaxScaler(feature_range=(0, 0.99))
    scaled = scaler.fit_transform(value)
    dataset = scaled.reshape(-1, N_features * (N_lookback + h))
    return dataset, scaler



def eval_onTest(test_model, Testdata):



    list_lr=[]
    list_epoch = []
    truevalue = []
    predictedvalue = []


    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards

    _, scaler = getTrain(n_features, n_lookback, n_fowards)


    start = time.time()
    flag = 0
    for t in range(epoch):


        # 训练流
        stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()
        scaled_stream = scaler.transform(stream)
        scaled_stream = scaled_stream.reshape(-1, n_obs, n_features)

        X = scaled_stream[:, 0: n_lookback, :]
        X = X.reshape(-1,n_lookback*n_features)
        Y = scaled_stream[:, -1, :-2]


        if t >= n_fowards:
            test_model.fit(X,Y)

        # 预测流

        stream_p = Testdata[(t + n_fowards) * n_obs:(t + n_fowards + 1) * n_obs].copy()  # 预测流  从t+1到t+n_fowards 时
        stream_py = stream_p[-1, :-2]
        stream_p = stream_p.reshape(-1, n_obs, n_features)

        truevalue.append(stream_py)
        stream_px = stream_p[:, 0: n_lookback, :].reshape(-1 , n_features)

        X = scaler.transform(stream_px)
        X = X.reshape(-1, n_lookback, n_features)
        X = X.reshape(-1, n_lookback * n_features)

        test_p = test_model.predict(X)
        # print(test_p.shape)
        # test_p, hstate_p, cstate_p = test_model(X)

        y = scaler.inverse_transform(np.concatenate((test_p,np.zeros((1,2))), axis=1).reshape(-1,14))
        y = y[:, :-2]
        predictedvalue.append(y[-1])
        end = time.time()
        speed = (end-start)/(t+1)
        print(speed)


    predictedvalue = np.array(predictedvalue)
    truevalue = np.array(truevalue)
    # plt.figure(figsize=(10, 5))
    # plt.plot(predictedvalue[:,9],label='P')
    # plt.plot(truevalue[:,9],label='T')
    # plt.legend()
    #
    # plt.show()
    #




    return predictedvalue, truevalue, list_epoch





#
if __name__ == "__main__":

    val_path = 'F://MH//Data//experiment//data//val_s//'


    data_path = 'F://MH//Data//experiment//data//test//'
    result_path ='F://MH//Data//experiment//result//'
    model_path ='F://MH//Data//experiment//MLP//'


    n_features = 14



    mode = 'dynamic_online'



    for n_fowards, n_lookback in [(5, 60)]:
        cdf = get_cdf(n_lookback, n_fowards)
        n_obs = n_lookback + n_fowards

    print("Regression task")
    # Model

    nums = [700]
    # 5_step best  num =700 ,batch =200
    # 10_step best  num =800 ,batch =300



    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)




            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)

            sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
            date = filepath.split("\\", 1)[0][-4:]
            carno = filepath.split("\\", 1)[1][:-4]

            for num in nums:
                n_batch = num

                oselmr = OSELMRegressor(n_hidden=num, activation_func='sigmoid', random_state=123)

                # 初始化训练模型
                Train, scaler = getTrain(n_features, n_lookback, n_fowards)
                X_train = Train[:, 0:n_features * n_lookback]
                y_train = Train[:, -n_features:-2].reshape(-1, 12)
                print(X_train.shape, y_train.shape)
                # Fit model with chunks of data
                X_batch = X_train[0 * n_batch:(0 + 1) * n_batch]
                y_batch = y_train[0 * n_batch:(0 + 1) * n_batch]
                oselmr.fit(X_batch, y_batch)
                # batch_list = [300,400,500]
                for n_batch in [200]:
                    for i in range(num // n_batch, len(y_train) // n_batch):
                        X_batch = X_train[i * n_batch:(i + 1) * n_batch]
                        y_batch = y_train[i * n_batch:(i + 1) * n_batch]
                        oselmr.fit(X_batch, y_batch)



                        # print("Train score for batch %i: %s" % (i + 1, str(oselmr.score(X_batch, y_batch))))

            # if date != '0226' or carno != '6':
            #     continue



            # 测试模型
            Test = sample.values.reshape(-1, n_features)
            predict_value, true_value, _ = eval_onTest(test_model=oselmr,
                                                       Testdata=Test
                                                       )




            subpath = result_path + '//{}//oselm_10//{}//'.format(mode, date)
            if not os.path.exists(subpath):
                os.makedirs(subpath)

            if date == '0226' and carno == '6':


                plt.figure(figsize=(10, 5))
                plt.plot(predict_value[:, 9], label='P')
                plt.plot(true_value[:, 9], label='T')
                plt.title("n={},batch={}".format(num,n_batch))
                plt.legend()
                plt.savefig(result_path + "//{}//oselm_10//{}_{}.png".format(mode, num,n_batch), dpi=512,
                               bbox_inches='tight')


                plt.close()

            #
            # np.save(subpath+ '{}p.npy'.format(carno), predict_value)
            # np.save(subpath+ '{}t.npy'.format(carno), true_value)
            rmse = math.sqrt(mean_squared_error(predict_value, true_value))

            MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100

            result = [{
                      "nums":num,
                      "n_batch":n_batch,
                      "date":date,
                      "carno":carno,
                      'RMSE': rmse,
                      'MAPE': MAPE}]

            # df = pd.DataFrame(result)
            # to_path = result_path + '//{}//oselm_10//bestscore.csv'.format(mode)
            # if os.path.exists(to_path):
            #     df.to_csv(to_path,
            #               line_terminator="\n",
            #               mode='a', index=None, header=False)
            # else:
            #     df.to_csv(to_path,
            #               line_terminator="\n",
            #               mode='a', index=None)
            # plot_result(predict_value, true_value, n_fowards, title=filepath)
                                # result = {'RMSE': np.array(list_rmse).mean(), 'MAPE': np.array(list_mape).mean}









#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# n_batch = 40

# # Fit model with chunks of data
# for i in range(20):
#     X_batch = X_train[i*n_batch:(i+1)*n_batch]
#     y_batch = y_train[i*n_batch:(i+1)*n_batch]
#     oselmr.fit(X_batch, y_batch)
#     print("Train score for batch %i: %s" % (i+1, str(oselmr.score(X_batch, y_batch))))
#
# # Results
# print("Train score of total: %s" % str(oselmr.score(X_train, y_train)))
# print("Test score of total: %s" % str(oselmr.score(X_test, y_test)))
# print("")
#

