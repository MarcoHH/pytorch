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
from distribution import get_cdf
import shutil
from ftrl import FTRL
import threading
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, h_size, n_layers):
        super(RNN, self).__init__()

        self.num_layers = n_layers
        self.hidden_size = h_size
        self.lstm = nn.LSTM(

            input_size=input_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
        )
        # self.out = nn.Linear(h_size, 12)  # torch.nn.Linear（
        # in_features：int，out_features：int，bias：bool = True ）

        self.out = nn.Sequential(  # Sequential组合结构
        nn.ReLU(), nn.Linear(h_size, 12))

    def forward(self, x, h_state=None, c_state=None):
        # Set initial hidden and cell states

        if h_state is None:
            h_state = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
            c_state = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)


        r_out, (h_STATE, c_STATE) = self.lstm(x, (h_state, c_state))

        # r_out = r_out.view(r_out.size(0), -1)
        # outs = self.out(r_out)

        # outs = self.out(torch.mean(r_out, 1))
        outs = self.out(h_STATE[-1])

        # outs = outs[:, -1, :].view(r_out.size(0), -1)
        # outs = outs[:, -1, :]

        return outs, h_STATE.detach(), c_STATE.detach()


class GRU(nn.Module):
    def __init__(self, input_size, h_size, n_layers):
        super(GRU, self).__init__()

        self.num_layers = n_layers
        self.hidden_size = h_size
        self.gru = nn.GRU(

            input_size=input_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
        )
        # self.out = nn.Linear(h_size, 12)  # torch.nn.Linear（
        # in_features：int，out_features：int，bias：bool = True ）

        self.out = nn.Sequential(  # Sequential组合结构
        nn.ReLU(), nn.Linear(h_size, 12))

    def forward(self, x, h_state=None):
        # Set initial hidden and cell states

        if h_state is None:
            h_state = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)



        r_out, h_STATE = self.gru(x, h_state)

        # r_out = r_out.view(r_out.size(0), -1)
        # outs = self.out(r_out)

        # outs = self.out(torch.mean(r_out, 1))

        outs = self.out(h_STATE[-1])

        # outs = outs[:, -1, :].view(r_out.size(0), -1)
        # outs = outs[:, -1, :]

        return outs, h_STATE.detach()


### GENERATE DATA ###
# optimizer parameters-----------------------------------------------------
ftrl_alpha = 1.0
ftrl_beta = 1.0
ftrl_l1 = 1.0
ftrl_l2 = 1.0

# Parameters
# model parameters-----------------------------------------------------------------

INPUT_SIZE = 14  # rnn 的输入维度
H_SIZE = 64  # of rnn 隐藏单元个数
N_LAYERS = 2
N_OUTS = 12
Batch_size = 512


# h_state = torch.zeros(N_LAYERS, 8, H_SIZE).to(DEVICE)
# c_state = torch.zeros(N_LAYERS, 8, H_SIZE).to(DEVICE)


def binned_entropy(x, max_bins):
    """
    First bins the values of x into max_bins equidistant bins.
    Then calculates the value of

    .. math::

        - \\sum_{k=0}^{min(max\\_bins, len(x))} p_k log(p_k) \\cdot \\mathbf{1}_{(p_k > 0)}

    where :math:`p_k` is the percentage of samples in bin :math:`k`.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param max_bins: the maximal number of bins
    :type max_bins: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    # nan makes no sense here
    if np.isnan(x).any():
        return np.nan

    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    probs[probs == 0] = 1.0
    return - np.sum(probs * np.log(probs))



def information(x, h):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    multi_entropy = np.zeros(x.shape[1]-2)
    range_x = x.max(axis=0) - x.min(axis=0)
    x_max = x.max(axis=0)
    for i in range(x.shape[1]-2):
        e = tsfresh.feature_extraction.feature_calculators.autocorrelation(x[:, i], h)

        multi_entropy[i] = math.log(1/abs(e))  * x_max[i]


    return sum(multi_entropy) * 1/(max(x_max))  #1/Max(range(Fi) *Σ I(Fi)*range(Fi)



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


def scaler_update(X, scaler):

    x = np.asarray(X)
    for k in range(x.shape[1]):
        dim_data = x[:, k]

        new_max = scaler.data_max_[k]
        new_min = scaler.data_min_[k]



        if min(dim_data) < scaler.data_min_[k]:
            new_min = min(dim_data)
            # print("update!")
        if max(dim_data) > scaler.data_max_[k]:
            new_max = max(dim_data)
            # print("update!")

            # weight_all = 1.1
            # scaler.data_max_[k] = new_max
            # scaler.data_min_[k] = new_min
        data_range = new_max - new_min
        x[:, k] = (x[:, k] - new_min)/data_range

            # scaler.data_range_[k] = scaler.data_max_[k] - scaler.data_min_[k]

    # X = (X - scaler.data_min_)/scaler.data_range_

    return x


def inverse(X, X_hat,scaler):


    x = np.asarray(X)

    for k in range(x.shape[1] - 2):
        dim_data = x[:, k]

        new_max = scaler.data_max_[k]
        new_min = scaler.data_min_[k]

        if max(dim_data) > scaler.data_max_[k]:
            new_max = max(dim_data)

        if min(dim_data) < scaler.data_min_[k]:
            new_min = min(dim_data)


        X_hat[k] = X_hat[k]*(new_max - new_min) + new_min


    return X_hat

def getTrain(N_features, N_lookback, h) -> np.ndarray:
    samples = list()
    data_path = 'F://MH//Data//experiment//data//train_s//'
        # 'F://MH//Data//experiment//train//'
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

            # data = pd.DataFrame(data)
            # # plt.plot(data.values[:,0],'b.')
            # data = data.interpolate(method='linear', axis=0, limit=10, inpalce=True)
            # # data = data.ewm(span=3).mean()
            # data = data.astype(np.float32)
            # # data.fillna(method='ffill', inplace=True, limit=100)
            # # plt.plot(data.values[:, 0],'r-')
            #
            # # data.fillna(method='ffill', inplace=True, limit=5)


            sample = series_to_supervised(value, h + N_lookback - 1, 1)
            sample = sample.values
            samples.append(sample)

    value = np.concatenate(samples, axis=0)
    value = value.reshape(-1, N_features)
    scaler = MinMaxScaler(feature_range=(0, 0.99))


    scaled = scaler.fit_transform(value)
    dataset = scaled.reshape(-1, N_features * (N_lookback + h))
    return dataset, scaler



def experiment(N_lookback, h, N_features=INPUT_SIZE):
    # define model
    rnn = GRU(input_size=INPUT_SIZE,
              h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  # 加载模型到对应设备
    # lr = 0.00001
    optimizer = torch.optim.Adam(rnn.parameters(),  weight_decay=0.0001)  # adam优化，
    # optimizer = torch.optim.SGD(rnn.parameters(),lr=lr, momentum = 0.8, weight_decay=0.0001)
    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差

    Train, scaler = getTrain(N_features, N_lookback, h)
    train_x = torch.from_numpy(Train[:, 0:N_features * N_lookback].reshape(-1, N_lookback, N_features))
    train_y = torch.from_numpy(Train[:, -N_features:-2].reshape(-1, 12))
    print(train_x.size(), train_y.size())
    Train_set = TensorDataset(train_x, train_y)
    Train_loader = DataLoader(Train_set,
                              batch_size=Batch_size,
                              shuffle=True)
    rnn.train()
    for epoch in range(200):
        # if (epoch+1) % 400 ==0:
        #     lr *= 0.1
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        for i, data in enumerate(Train_loader):
            inputs, targets = data
            y_hat,_ = rnn(inputs)  # rnn output

            loss = criterion(y_hat.cpu(), targets)
            # 这三行写在一起就可以
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:  # 每训练100个批次可视化一下效果，并打印一下loss
            print("EPOCHS: {},Loss:{:4f}".format(epoch, loss))
            # print(epoch, i,'inputs',inputs.data.size(),
            #       'targets',targets.data.size())

        if (epoch + 1) % 400 == 0:
            run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
            torch.save({
                'epoch': epoch,
                'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss

            }, 'D://lab//Data//experiment//checkpoint//{}_{}.pth'.format(run_id,h))


    # 打印最终的损失值
    # output = rnn(inputs)
    # loss = criterion(output, targets)
    # print(loss.item())

    return rnn


def load_offline(model_path):
    model = RNN(input_size=INPUT_SIZE,
                h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  # 加载模型到对应设备
    model.load_state_dict(torch.load(model_path))

    return model




def information(x, h):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    multi_entropy = np.zeros(x.shape[1]-2)
    range_x = x.max(axis=0) - x.min(axis=0)
    x_max = x.max(axis=0)
    for i in range(x.shape[1]-2):
        e = tsfresh.feature_extraction.feature_calculators.autocorrelation(x[:, i], h)

        multi_entropy[i] = math.log(1/abs(e))  * x_max[i]


    return sum(multi_entropy) * 1/(max(x_max))  #1/Max(range(Fi) *Σ I(Fi)*range(Fi)





def get_weight(model):
    '''
    获得模型的权重列表
    :param model:
    :return:
    '''
    weight = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight[name] = param.clone().detach()

    return weight


def eval_onTest(test_model, Testdata, mode='offline', is_resample=False, mylr=0.0001, rate=0.2):


    optimizer = torch.optim.Adam(test_model.parameters(),lr=mylr)
    # optimizer = torch.optim.SGD(test_model.parameters(), lr=mylr, momentum=rate)

    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差

    list_info = []
    list_lr=[]
    list_epoch = []
    truevalue = []
    predictedvalue = []
    error_buffer = []


    start = time.time()
    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards
    k_count = 0
    _, scaler = getTrain(n_features, n_lookback, n_fowards)

    hstate = torch.randn(2, 1, H_SIZE).to(DEVICE)
    cstate = torch.randn(2, 1, H_SIZE).to(DEVICE)

    hstate_p = torch.rand(2, 1, H_SIZE).to(DEVICE)
    cstate_p = torch.rand(2, 1, H_SIZE).to(DEVICE)

    flag = 0
    for t in range(epoch):
        list_lr.append(mylr)
        k_count += 1

        if flag == 0:
            weight_buffer = get_weight(test_model)
            flag = 1
        else:
            for name, param in test_model.named_parameters():
                if 'weight' in name:
                    weight_buffer[name].mul_(0.05).add_(param.clone().detach(), alpha=0.95)

        if mode == 'online':
            stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()

            score = information(stream, n_fowards)
            delta_peresist = max(abs(stream[-(n_fowards + 1), :-2] - stream[-1, :-2]))

            scaled_stream = scaler.transform(stream)
            scaled_stream = scaled_stream.reshape(-1, n_obs, n_features)

            X = scaled_stream[:, 0: n_lookback, :]
            X = torch.from_numpy(X)
            X = X.to(DEVICE)

            Y = scaled_stream[:, -1, :-2]
            Y = torch.from_numpy(Y)
            Y = Y.to(DEVICE)

            test_model.train()  # 模型训练模式

            e_list = []
            if t > n_fowards and is_resample:

                for i in range(n_fowards):
                    error = abs((error_buffer[i][n_fowards - 1 - i, :] - truevalue[t - n_fowards]))
                    e_list.append(error)
                delta_window = np.array(e_list)
                delta_mean = np.mean(delta_window[:, [0, 3, 6, 9]], axis=1)
                delta_max = np.max(delta_window, axis=1)

                error1 = delta_mean[:int(n_fowards * 0.5)+1].mean()
                error2 = delta_mean[int(n_fowards * 0.5)-1:].mean()

                # delta = max(e_list[0])

                epoch_of_targetsample = n * cdf(score) * (error2 / (error1 + 0.001)) * error2 / (delta_peresist + 0.001)
                if epoch_of_targetsample > 20:
                    epoch_of_targetsample = 20
                # list_epoch.append(delta_max)



                #  训练
                step = 0
                while step < epoch_of_targetsample:
                    step += 1

                    lr_local = mylr / (math.sqrt(step))  # 动态调整一个样本的学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_local

                    train_p, hstate,_ = test_model(X)  #
                    # train_p, hstate = test_model(X, hstate)
                    loss = criterion(train_p.cpu(), Y)  # torch.size [1,3]
                    # 这三行写在一起就可以


                    regret = 0
                    for name, param in test_model.named_parameters():
                        if 'weight' in name:
                            regret += torch.norm(param - weight_buffer[name], p=2)

                    loss = loss + regret.mul(rate)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 这三行写在一起就可以


            # 不采样训练
            else:
                epoch_of_persample = 0
                step = 0
                while step < epoch_of_persample:
                    step += 1
                    train_p, hstate,_ = test_model(X)  # rnn output
                    # train_p, hstate = test_model(X, hstate)
                    # hstate = hstate.detach()
                    # cstate = cstate.detach()
                    loss = criterion(train_p.cpu(), Y)
                    # 这三行写在一起就可以
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # if (t + 1) % 20 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        #     print("EPOCHS: {},Loss:{:4f}".format(t, loss))



        # stream_p = Testdata[(t + n_fowards) * n_obs:(t + n_fowards + 1) * n_obs]  # 预测流 8车 电机1
        # stream_py = stream_p[-1, :-2]
        # truevalue.append(stream_py)
        #
        stream_p = Testdata[(t + 1) * n_obs:(t + n_fowards + 1) * n_obs].copy()  # 预测流  从t+1到t+n_fowards 时
        stream_py = stream_p[-1, :-2]
        stream_p = stream_p.reshape(-1, n_obs, n_features)

        truevalue.append(stream_py)


        stream_px = stream_p[:,0: n_lookback, :].reshape(-1 , n_features)
        inv_x = stream_px.copy()
        if mode == 'offline':
            X = scaler.transform(stream_px)
        else:
            X = scaler.transform(stream_px)


        X = X.reshape(-1, n_lookback, n_features)
        X = torch.from_numpy(X).to(DEVICE)

        # Test the model
        test_model.eval()
        with torch.no_grad():
            test_p, _,_ = test_model(X)
            # test_p, hstate_p, cstate_p = test_model(X)
            test_p = test_p.cpu().data.numpy()



            y = scaler.inverse_transform(np.concatenate((test_p, np.zeros((n_fowards,2))), axis=1).reshape(-1, 14))


            y = y[:, :-2]

            predictedvalue.append(y[-1])
            end=time.time()
            average =(end-start)/(t+1)
            print('avg_cost:{}-------{}'.format(average,t))

            if len(error_buffer) == n_fowards:
                error_buffer.pop(0)
                error_buffer.append(y)

            else:
                error_buffer.append(y)

    predictedvalue = np.array(predictedvalue)
    truevalue = np.array(truevalue)
    # list_epoch.sort()

    # plt.figure(figsize=(20, 6))
    # plt.plot(list_epoch, label='epoch')
    # plt.legend()
    # plt.title('rate{}:lr:{}'.format(rate, mylr))
    # plt.show()
    #
    # plt.figure(figsize=(20, 6))
    # plt.plot(list_lr, label='lr')
    # plt.legend()
    # plt.show()
    #
    # plt.figure(figsize=(20, 6))
    # plt.plot(list_info, label='infoinfomation')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(20, 6))
    # plt.plot(truevalue[:800, 0])
    # plt.plot(predictedvalue[:800, 0])
    # plt.title('rate{}:lr:{}'.format(rate, mylr))
    # plt.show()
    return predictedvalue, truevalue, list_epoch


class myThread (threading.Thread):
    def __init__(self, threadID, list_v, list_p, Test_):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.list_v = list_v
        self.list_p = list_p
        self.Test_ = Test_
    def run(self):
        print ("开始线程：" + str(self.list_p))
        # threadLock.acquire()
        train(self.list_v, self.list_p, self.Test_)
        # threadLock.release()
        print ("退出线程：" + str(self.list_p))

def train(list_v, list_p, Test_):
    time.sleep(1)
    for v in list_v:
        for p in list_p:
            lr = pow(10, -p)

            offline_model = load_offline(model_path)

            subpath = result_path + '//{}//{}_GRU//{}_{}//{}//'.format(mode, n_fowards, v, p, date)
            if not os.path.exists(subpath):
                os.makedirs(subpath)

            if mode == 'dynamic_online':
                predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                           Testdata=Test_,
                                                           mode='online',
                                                           is_resample=True,
                                                           rate=v,
                                                           mylr=lr
                                                           )

            if mode == 'offline':
                predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                           Testdata=Test)

            np.save(subpath + '{}p.npy'.format(carno), predict_value)
            np.save(subpath + '{}t.npy'.format(carno), true_value)

            value_p = predict_value[:, :]
            value_t = true_value[:, :]
            rmse = math.sqrt(mean_squared_error(value_p, value_t))

            MAPE = np.sum(abs(value_p - value_t) / value_t) / value_t.size * 100

            result = {'RMSE': rmse, 'MAPE': MAPE,'p':p,'rate':v}
            print(result)
            df = pd.DataFrame(result, index=['lr:{}-rate:{}'.format(lr, v)])
            df.to_csv(result_path + '//{}//{}_GRU//test.csv'.format(mode, n_fowards),
                      mode='a', header=False)
            print('活跃数:{}'.format(threading.activeCount()))


#
if __name__ == "__main__":
    exitFlag = 0
    val_path = 'F://MH//Data//experiment//data//val_s//'
    data_path = 'F://MH//Data//experiment//data//test//'
    result_path ='F://MH//Data//experiment//result//'
    # model_path ='D://lab//Data//experiment//model8//2021-01-03_08.14.4230rnn_10.pth'
    n_features = 14

    n = 12
    # mode = 'dynamic_online'
    # mode = 'offline'
    H_SIZE = 64
    # for n_fowards in [10,15]:
    #     for n_lookback in [60]:
    #         n_obs = n_lookback + n_fowards
    #         model1 = experiment(N_lookback=n_lookback, h=n_fowards, N_features=14)
    #         run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    #         torch.save(model1.state_dict(),
    #                    "D://lab//Data//experiment//GRU//" + str(n_lookback) + "gru_" + str(n_fowards) + ".pth")

    # for mode in ['dynamic_online']:
    #     for n_fowards in [5]:
    #         for n_lookback in [60]:
    #             cdf = get_cdf(n_lookback, n_fowards)
    #
    #             model_path = 'F://MH//Data//experiment//GRU//{}gru_{}.pth'.format(n_lookback,n_fowards)
    #             n_obs = n_lookback + n_fowards
    #
    #             for root, dirs, files in os.walk(data_path):
    #                 for file in files:
    #                     filepath = os.path.join(root, file)
    #
    #                     date = filepath.split("\\", 1)[0][-4:]
    #                     carno = filepath.split("\\", 1)[1][:-4]
    #
    #                     if date == '0207':
    #                         print(filepath)
    #                         data = np.load(filepath, allow_pickle=True)
    #                         data = data.astype(np.float32)
    #
    #                         sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
    #                         Test = sample.values.reshape(-1, n_features)
    #                         # Tests.append(Test)
    #                         threadLock = threading.Lock()
    #                         threads = []
    #
    #                         # list_rmse = []
    #                         thread1 = myThread(1, [0, 0.2, 0.4, 0.6, 0.8], [3,4], Test.copy())
    #                         thread2 = myThread(2, [0, 0.2, 0.4, 0.6, 0.8], [5,6], Test.copy())
    #                         # thread3 = myThread(3, [0, 0.2, 0.4, 0.6, 0.8], [5], Test.copy())
    #                         # thread4 = myThread(4, [0, 0.2, 0.4, 0.6, 0.8], [6], Test.copy())
    #
    #
    #                         thread1.start()
    #                         thread2.start()
    #                         # thread3.start()
    #                         # thread4.start()
    #                         # 添加线程到线程列表
    #                         threads.append(thread1)
    #                         threads.append(thread2)
    #                         # threads.append(thread3)
    #                         # threads.append(thread4)
    #
    #                         # 等待所有线程完成
    #                         for t in threads:
    #                             t.join()
    #                         print("退出主线程")


    for mode in ['offline']:
        for n_fowards in [10]:
            for n_lookback in [60]:
                cdf = get_cdf(n_lookback, n_fowards)

                model_path = 'F://MH//Data//experiment//LSTM//{}lstm_{}.pth'.format(n_lookback,n_fowards)
                n_obs = n_lookback + n_fowards
                Tests = list()


                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        filepath = os.path.join(root, file)

                        date = filepath.split("\\", 1)[0][-4:]
                        carno = filepath.split("\\", 1)[1][:-4]

                        # if carno == '6':
                        # print(filepath)
                        data = np.load(filepath, allow_pickle=True)
                        data = data.astype(np.float32)


                        sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
                        Test = sample.values.reshape(-1, n_features)
                        # Tests.append(Test)



                        offline_model = load_offline(model_path)

                        subpath = result_path + '//{}//lstm_{}//{}//'.format(mode, n_fowards, date)
                        if not os.path.exists(subpath):
                            os.makedirs(subpath)

                        if mode == 'dynamic_online':
                            predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                       Testdata=Test,
                                                                       # Testdata=np.concatenate(Tests, axis=0),
                                                                       mode='online',
                                                                       is_resample=True,
                                                                       rate=0.2,
                                                                       mylr=0.0001
                                                                       )


                        if mode == 'offline':
                            predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                       Testdata=Test,
                                                                       # Testdata=np.concatenate(Tests, axis=0)
                                                                       )


                        np.save(subpath + '{}p.npy'.format(carno), predict_value)
                        np.save(subpath+ '{}t.npy'.format(carno), true_value)

                        value_p = predict_value[:, :]
                        value_t = true_value[:, :]
                        rmse = math.sqrt(mean_squared_error(value_p, value_t))

                        MAPE = np.sum(abs(value_p - value_t) / value_t) / value_t.size * 100

                        result = [{

                        "date": date,
                        "carno": carno,
                        'RMSE': rmse,
                        'MAPE': MAPE}]
                        print(result)
                        df = pd.DataFrame(result)
                        df.to_csv(result_path + '//{}//lstm_{}//test.csv'.format(mode, n_fowards),
                                  mode='a', line_terminator="\n", header=False,index =None)


        #     if mode == 'offline':
        #         predict_value, true_value, _ = eval_onTest(test_model=offline_model,
        #                                                    Testdata=np.concatenate(Tests, axis=0))
        #     value_p = predict_value[:, :]
        #     value_t = true_value[:, :]
        #     rmse = math.sqrt(mean_squared_error(value_p, value_t))
        #
        #     MAPE = np.sum(abs(value_p - value_t) / value_t) / value_t.size * 100
        #
        #     result = {'RMSE': rmse, 'MAPE': MAPE}
        #     print(result)
        #     df = pd.DataFrame(result, index=['n_forwards:{}-n_lookback:{}'.format(n_fowards, n_lookback)])
        #     df.to_csv(result_path + '//lstm_test.csv', mode='a', header=False)
        #
        # list_rmse.append(rmse)


    # model1 = experiment(N_lookback=30, h=10, N_features=14)
    # run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    # torch.save(model1.state_dict(),
    #            "F://MH//Data//experiment//model8//" + str(run_id) + str(n_lookback) + "rnn_" + str(n_fowards) + ".pth")

    # model1 = experiment(N_lookback=30, h=10, N_features=14)
    # run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    # torch.save(model1.state_dict(),
    #            "F://MH//Data//experiment//model7//" + str(run_id) + str(n_lookback) + "rnn_" + str(n_fowards) + ".pth")

    # H_SIZE = 64
    # model2 = experiment(N_lookback=30, h=5, N_features=14)
    # torch.save(model2.state_dict(),
    #            "F://MH//Data//experiment//model6//" + str(n_lookback) + "rnn_" + str(n_fowards) + ".pth")

    # eval_onTrain()

    # logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO)
#
# for v in [0, 0.2, 0.4, 0.6, 0.8]:
#     for lr in [0.001, 0.01, 0.1, 0.2, 0.4, 0.6,0.8]:
# for v in [0.2]:
#     for lr in [0.2]:
#
#         Tests = list()
#         list_rmse = []
#         list_mape = []
#         for root, dirs, files in os.walk(data_path):
#             for file in files:
#                 filepath = os.path.join(root, file)
#                 print(filepath)
#                 print('rate:{} lr:{}'.format(v, lr))
#                 data = np.load(filepath, allow_pickle=True)
#                 data = data.astype(np.float32)
#                 # data = pd.DataFrame(data)
#                 # data = data.interpolate(method='linear', axis=0, limit=10, inpalce=True)
#                 # data = data.ewm(span=2).mean()
#                 # data = data.astype(np.float32)
#                 # # data.fillna(method='ffill', inplace=True, limit=5)
#                 # value = data.values[650:1300, :]
#
#                 sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
#                 Test = sample.values.reshape(-1, n_features)
#                 Tests.append(Test)
#
#                 date = filepath.split("\\", 1)[0][-4:]
#                 carno = filepath.split("\\", 1)[1][:-4]
#
#
#
#                 offline_model = load_offline(model_path)
#                 # offline_model = RNN(input_size=INPUT_SIZE,
#                 #           h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  # 加载模型到对应设备
#
#                 subpath = result_path + '//{}//1-3//{}//'.format(mode, date)
#                 if not os.path.exists(subpath):
#                     os.makedirs(subpath)
#
#                 if mode == 'baseline':
#                     predict_value, true_value, _ = eval_onTest(test_model=offline_model,
#                                                                Testdata=Test)
#
#                 if mode == 'dynamic_online':
#                     predict_value, true_value, _ = eval_onTest(test_model=offline_model,
#                                                                Testdata=Test,
#                                                                mode='online',
#                                                                is_resample=True,
#                                                                # rate=v,
#                                                                # mylr=lr
#                                                                )
#                 if mode == 'static_online':
#                     predict_value, true_value, _ = eval_onTest(test_model=offline_model,
#                                                                Testdata=Test,
#                                                                mode='online')
#
#                 np.save(subpath+ '{}p.npy'.format(carno), predict_value)
#                 np.save(subpath+ '{}t.npy'.format(carno), true_value)
#                 rmse = math.sqrt(mean_squared_error(predict_value, true_value))
#                 list_rmse.append(rmse)
#                 MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100
#                 list_mape.append(MAPE)
#                 result = {'RMSE': rmse, 'MAPE': MAPE}
#                 df = pd.DataFrame(result, index=['rate:{}-lr:{}-date:{}-no:{}'.format(v, lr, date, carno)])
#                 df.to_csv(result_path + '//{}//1-3//score.csv'.format(mode), mode='a', header=False)
#
#

                # plot_result(predict_value, true_value, n_fowards, title=filepath)




            # df.to_csv(result_path + 'score.csv', mode='a', header=False)
            # np.save(result_path + '{}-{}p.npy'.format(date, carno), predict_value)
            # np.save(result_path + '{}-{}t.npy'.format(date, carno), true_value)

            # logging.info(filepath)
            # logging.info('RMSE--------{0}--MAPE------{1}'.format(rmse, MAPE))



            # plot_result(predict_value, true_value, n_fowards, title=filepath)




      # 离线模式
    # offline_model = load_offline(model_path)
    #
    # predict_value, true_value, _ = eval_onTest(test_model=offline_model,
    #                                            Testdata=np.concatenate(Tests, axis=0))
    # np.save(result_path + '//baseline//p.npy', predict_value)
    # np.save(result_path + '//baseline//t.npy', true_value)
    #
    # #   在线模式
    # offline_model = load_offline(model_path)
    #
    # predict_value, true_value, _ = eval_onTest(test_model=offline_model,
    #                                            Testdata=np.concatenate(Tests, axis=0),
    #                                            mode='online')
    # np.save(result_path + '//static_online//p.npy', predict_value)
    # np.save(result_path + '//static_online//t.npy', true_value)


    # #  在线模式
    # for lr in [0.02,0.01,0.005,0.001]:
    #
    #     for rate in [0.2,0.4,0.6,0.8]:
    #
    #
    #         offline_model = load_offline(model_path)
    #         predict_value, true_value, _ = eval_onTest(test_model=offline_model,
    #                                                    Testdata=np.concatenate(Tests, axis=0),
    #                                                    mode='online',
    #                                                    is_resample=True,
    #                                                    mylr= lr,
    #                                                    rate= rate)
    #         np.save(result_path + '//dynamic_online//{}p{}.npy'.format(rate, lr), predict_value)
    #         # np.save(result_path + '//dynamic_online//t.npy', true_value)
    #         rmse = math.sqrt(mean_squared_error(predict_value, true_value))
    #         #
    #
    #         MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100
    #         result = {'RMSE': rmse, 'MAPE': MAPE}
    #         df = pd.DataFrame(result, index=['rate:{}-lr:{}'.format(rate, lr)])
    #         df.to_csv(result_path + '//dynamic_online//score.csv', mode='a', header=False)


