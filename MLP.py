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





# class myLoss(nn.Module):
#     def __init__(self,parameters):
#         self.params = self.parameters
#
#     def forward(self):
#         loss = cal_loss(self.params)
#         return loss







### GENERATE DATA ###
# optimizer parameters-----------------------------------------------------
ftrl_alpha = 1
ftrl_beta = 1
ftrl_l1 = 0
ftrl_l2 = 0

# Parameters
# model parameters-----------------------------------------------------------------

INPUT_SIZE = 14  # rnn 的输入维度
H_SIZE: int = 128  # of rnn 隐藏单元个数
N_LAYERS = 2
N_OUTS = 12
Batch_size = 128


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
    # len=0
    # n=1
    samples = list()
    data_path ='F://MH//Data//experiment//data//train_s//'
               # 'F://MH//Data//experiment//data//val_s//'

    for root, dirs, files in os.walk(data_path):
        for file in files:
            # #获取文件所属目录
            # print(root)
            # #获取文件路径
            # print(os.path.join(root, file))
            filepath = os.path.join(root, file)

            data = np.load(filepath, allow_pickle=True)
            # len+=data.shape[0]
            # print(len/n)
            # n+=1
            # data = data.astype(np.float32)
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


def experiment(N_lookback, h, N_features=INPUT_SIZE):
    # define model

    liner = MLP(input_size= N_features*N_lookback,
              h_size=H_SIZE).to(DEVICE)  # 加载模型到对应设备
    lr = 0.00001
    optimizer = torch.optim.Adam(liner.parameters(), lr = lr, weight_decay=0.0001)  # adam优化，
    # optimizer = torch.optim.SGD(liner.parameters(),lr=lr, momentum = 0.9, weight_decay=0.0001)
    # StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差

    Train, scaler = getTrain(N_features, N_lookback, h)
    train_x = torch.from_numpy(Train[:, 0:N_features * N_lookback])
    train_y = torch.from_numpy(Train[:, -N_features:-2].reshape(-1, 12))
    print(train_x.size(), train_y.size())
    Train_set = TensorDataset(train_x, train_y)
    Train_loader = DataLoader(Train_set,
                              batch_size=Batch_size,
                              shuffle=True)
    liner.train()
    for epoch in range(1000):
        if (epoch+1) % 400 ==0:
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(Train_loader):
            inputs, targets = data
            y_hat = liner(inputs)  # rnn output

            loss = criterion(y_hat.cpu(), targets)
            # 这三行写在一起就可以
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:  # 每训练100个批次可视化一下效果，并打印一下loss
            print("EPOCHS: {},Loss:{:4f}".format(epoch, loss))
            # print(epoch, i,'inputs',inputs.data.size(),
            #       'targets',targets.data.size())

        if (epoch + 1) % 400 == 0:
            run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
            torch.save({
                'epoch': epoch,
                'model_state_dict': liner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss

            }, 'D://lab//Data//experiment//checkpoint//{}.pth'.format(run_id))


    # 打印最终的损失值
    # output = rnn(inputs)
    # loss = criterion(output, targets)
    # print(loss.item())

    return liner


def load_offline(model_path):
    model = MLP(input_size=INPUT_SIZE * n_lookback,
                h_size=H_SIZE).to(DEVICE)  # 加载模型到对应设备
    model.load_state_dict(torch.load(model_path))

    return model


def eval_onTrain() -> None:
    h = 5
    n_back = 30
    Train, scaler = getTrain(14, n_back, h)
    train_x = torch.from_numpy(Train[:, 0:n_back * 14].reshape(-1, n_back, 14))
    train_y = torch.from_numpy(Train[:, -14:].reshape(-1, 14))

    train_x = train_x.to(DEVICE)
    offline_model = load_offline('F://MH//Data//experiment//model5//30rnn_5.pth')
    offline_model.eval()
    with torch.no_grad():
        predicted, _, _ = offline_model(train_x[0:1600, :, :])

    value = np.concatenate((predicted, train_y[0:1600, -2:]), axis=1)

    inv_y = scaler.inverse_transform(value)
    train_y = scaler.inverse_transform(train_y)

    predicted_value = inv_y[h:, 0:3]
    target_value = train_y[h:1600, 0:3]
    history_value = train_y[0:1600, 0:3]
    plt.figure(figsize=(20, 6))
    plt.plot(predicted_value, '.', label='predict')
    plt.plot(target_value, '-', label='target')
    plt.plot(history_value, ':', label='history')
    plt.legend()
    plt.ylabel('value', fontsize=15)
    plt.xlabel('Time Step(1 min)', fontsize=15)
    plt.show()


# optimizer = FTRL(offline_model.parameters(), alpha=ftrl_alpha, beta=ftrl_beta, l1=ftrl_l1, l2=ftrl_l2)
# for name, p in offline_model.named_parameters():
#     if name.startswith('lstm'):
#         p.requires_grad = False
#
# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad is not False ,offline_model.parameters()),
# lr=0.1) # SGD优化，
# optimizer = torch.optim.Adam(offline_model.parameters())  # adam优化，

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


def eval_onTest(test_model, Testdata, mode='offline', is_resample=False, mylr=0.000001, rate=0.9):

    # mode = 'offline'
    # optimizer = torch.optim.SGD(test_model.parameters(), lr=mylr, momentum=rate, dampening=0)

    optimizer = torch.optim.Adam(test_model.parameters(),lr=mylr)
    # optimizer = FTRL(test_model.parameters(), alpha=ftrl_alpha, beta=ftrl_beta, l1=ftrl_l1, l2=ftrl_l2)
    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差

    list_info = []
    list_lr=[]
    list_epoch = []
    truevalue = []
    predictedvalue = []


    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards
    k_count = 0
    _, scaler = getTrain(n_features, n_lookback, n_fowards)
    error_buffer = []

    start = time.time()
    flag = 0
    for t in range(epoch):
        list_lr.append(mylr)
        k_count += 1



        if  flag == 0:
            weight_buffer = get_weight(test_model)
            flag = 1
        else:
            for name, param in test_model.named_parameters():
                if 'weight' in name:
                     weight_buffer[name].mul_(0.05).add_(param.clone().detach(), alpha=0.95)


        if mode == 'online':
            stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()



            # 计算时间序列信息熵
            score = information(stream, n_fowards)
            delta_peresist = max(abs(stream[-(n_fowards+1),:-2]-stream[-1,:-2]))
            # 记录所有信息熵
            # list_info.append(entropy)

            # scaled_stream = scaler_update(stream, scaler)
            scaled_stream = scaler.transform(stream)
            scaled_stream = scaled_stream.reshape(-1, n_obs, n_features)

            X = scaled_stream[:, 0: n_lookback, :]
            X = X.reshape(-1,n_lookback*n_features)
            X = torch.from_numpy(X)
            X = X.to(DEVICE)

            Y = scaled_stream[:, -1, :-2]
            Y = torch.from_numpy(Y)
            Y = Y.to(DEVICE)

            test_model.train()  # 模型训练模式




            # if t % 60 ==0:
            #     lr = lr*0.9
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr

            #动态调整全局学习率
            # if t % 10 ==0:
            #     lr = lr/math.sqrt(k_count+1)
            #     lr = lr * 0.99
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr

            # 采样训练

            e_list = []
            if t >= n_fowards and is_resample:

                for i in range(n_fowards):
                    error = abs((error_buffer[i][n_fowards-1-i, :] - truevalue[t - n_fowards]) )
                    e_list.append(error)
                delta_window = np.array(e_list)
                delta_mean = np.mean(delta_window[:, [0, 3, 6, 9]], axis=1)
                delta_max = np.max(delta_window,axis=1)

                error1 = delta_mean[:int(n_fowards*0.5)].mean()
                error2 = delta_mean[int(n_fowards*0.5):].mean()

                delta = max(e_list[0])
                # delta = max(abs(predictedvalue[t - n_fowards] - truevalue[t - n_fowards]) )
                # print(delta,delta_peresist)

                # if trend < 0:
                #     epoch_of_targetsample = 6*cdf(score)*max(e_list[-1])*max(e_list[-1])/(delta*delta_peresist)
                #
                # else:
                #     epoch_of_targetsample = 6*cdf(score)*max(delta/delta_peresist, 1)

                epoch_of_targetsample = n*cdf(score)*(error2/(error1+0.001))*error2/(delta_peresist+0.001)
                if epoch_of_targetsample >20:
                    epoch_of_targetsample =20
                list_epoch.append(delta_max)
                # print('trend:{}---delta:{} --de_p:{}--epoch:{}'.format(trend,delta,delta_peresist,epoch_of_targetsample))
                # print(epoch_of_targetsample)
                # if delta >= eplison:
                #
                #     epoch_of_targetsample = int((1 - np.exp(-2 * (delta - eplison) / eplison)) * 4) + 1

                    # max([epoch_of_targetsample1,epoch_of_targetsample2])





                #  动态调整全局学习率
                # if epoch_of_targetsample > 1 and k_count>10:
                #     # nonlocal lr
                #     k_count = 0
                #     # mylr = 0.2
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = mylr

                #  训练
                step = 0
                while step < epoch_of_targetsample:
                    step += 1

                    lr_local = mylr / (math.sqrt(step))  # 动态调整一个样本的学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_local


                    train_p = test_model(X)  #

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

                        # .mul(0.5)

                    # 这三行写在一起就可以


            # 不采样训练
            else:
                epoch_of_persample = n
                step = 0
                while step < epoch_of_persample:
                    step += 1
                    train_p = test_model(X)  # rnn output

                    loss = criterion(train_p.cpu(), Y)
                    # 这三行写在一起就可以
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # if (t + 1) % 20 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        #     print("EPOCHS: {},Loss:{:4f}".format(t, loss))



        stream_p = Testdata[(t + 1) * n_obs:(t + n_fowards + 1) * n_obs].copy()  # 预测流  从t+1到t+n_fowards 时
        stream_py = stream_p[-1, :-2]
        stream_p = stream_p.reshape(-1, n_obs, n_features)





        truevalue.append(stream_py)
        stream_px = stream_p[:, 0: n_lookback, :].reshape(-1 , n_features)
        # inv_x = stream_px.copy()
        if mode == 'offline':
            X = scaler.transform(stream_px)
        else:
            # X = scaler_update(stream_px, scaler)
            X = scaler.transform(stream_px)

        X = X.reshape(-1, n_lookback, n_features)
        X = X.reshape(-1, n_lookback * n_features)
        X = torch.from_numpy(X).to(DEVICE)

        # Test the model
        test_model.eval()
        with torch.no_grad():
            test_p = test_model(X)
            # test_p, hstate_p, cstate_p = test_model(X)
            test_p = test_p.cpu().data.numpy()
            y = scaler.inverse_transform(np.concatenate((test_p,np.zeros((n_fowards,2))), axis=1).reshape(-1,14))
            y = y[:, :-2]
            predictedvalue.append(y[-1])
            # end = time.time()
            # average = (end - start) / (t + 1)
            # print('avg_cost:{}-------{}'.format(average, t))

            if len(error_buffer) == n_fowards:
                error_buffer.pop(0)
                error_buffer.append(y)

            else:
                error_buffer.append(y)
            # scaler.sc
            # y = test_p * scaler.data_range_[:-2] + scaler.data_min_[:-2]


    predictedvalue = np.array(predictedvalue)
    truevalue = np.array(truevalue)
    # list_epoch.sort()

    # plt.figure(figsize=(20, 6))
    # plt.plot(list_epoch, label='epoch')
    # plt.legend()
    # plt.title('rate{}:lr:{}'.format(rate, mylr))
    # plt.show()

    # plt.figure(figsize=(20, 6))
    # plt.plot(list_lr, label='lr')
    # plt.legend()
    # plt.show()
    # #
    # plt.figure(figsize=(20, 6))
    #     #     # plt.plot(list_info, label='infoinfomation')
    #     #     # plt.legend()
    #     #     # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(predictedvalue[:,9],label='P')
    # plt.plot(truevalue[:,9],label='T')
    # plt.title('rate{}:lr:{}'.format(rate, mylr))
    # plt.legend()
    # #
    # plt.show()
    # baseline = np.load(
    #     result_path + 'offline//1-30-1e-05//{}//{}_{}p.npy'.format('0226', 6, 5))[:,
    #            ]
    # plt.figure(figsize=(10, 8))
    #
    # marke_pos = [i * 4 for i in range(int(truevalue.shape[0] / 4))]
    # for i in range(4):
    #     plt.subplot(4, 1, i + 1)
    #     # pyplot.title(r"$\bf{" + 'No.{}'.format(i+1) + "}$",fontsize=12)
    #     plt.plot(truevalue[:, i * 3], markevery=marke_pos, label='No.{}:truevalue'.format(i + 1))
    #     # pyplot.plot(truevalue[:,(i * 3)+1],marker='.' ,markevery=marke_pos, label='baseline', color='green')
    #     # pyplot.plot(truevalue[:,(i * 3)+2],marker='.' ,markevery=marke_pos, label='online_dynamic', color='r')
    #
    #     plt.plot(baseline[:, i * 3], '--',label='offline', color='r')
    #
    #     # pyplot.plot(baseline[:, (i * 3) + 1], label='baseline', color='orange')
    #     # pyplot.plot(baseline[:, (i * 3) + 2], label='online_dynamic', color='orange')
    #     plt.plot(predictedvalue[:, i * 3], marker='.', markevery=marke_pos,label='online', color='grey')
    #
    #     plt.tick_params(labelsize=12)
    #     plt.ylabel('temperature (℃)', fontsize=12)
    # plt.legend(loc=8)
    # plt.xlabel('Time (1 min)', fontsize=12)
    # plt.savefig(result_path + "figure//date//{}-{}.png".format(trick, rate), dpi=512,
    #                bbox_inches='tight')
    # # plt.show()
    return predictedvalue, truevalue, list_epoch





#
if __name__ == "__main__":

    val_path = 'F://MH//Data//experiment//data//val_s//'


    data_path = 'F://MH//Data//experiment//data//test//'
    result_path ='F://MH//Data//experiment//result//'
    model_path ='F://MH//Data//experiment//MLP//'
    # model_path ='F://MH//Data//experiment//model6//30rnn_5.pth'


    # mode = 'offline'
    # mode = 'dynamic_online'
    # mode = 'static_online'

    n_features = 14

    # n_features, n_fowards = 14, 5
    # H_SIZE = 128
    # for n_lookback in [60]:
    #     n_obs = n_lookback + n_fowards
    #     model1 = experiment(N_lookback=n_lookback, h=5, N_features=14)
    #     run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    #     torch.save(model1.state_dict(),
    #                "D://lab//Data//experiment//MLP//" + str(n_lookback) + "rnn_" + str(n_fowards) + ".pth")

    # n_features, n_fowards = 14, 10
    # for n_lookback in [60,70,80]:
    #     n_obs = n_lookback + n_fowards
    #     model1 = experiment(N_lookback=n_lookback, h=10, N_features=14)
    #     run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    #     torch.save(model1.state_dict(),
    #                "D://lab//Data//experiment//MLP//" + str(n_lookback) + "rnn_" + str(
    #                    n_fowards) + ".pth")
    #
    # n_features, n_fowards = 14, 15
    # for n_lookback in [60,70,80]:
    #     n_obs = n_lookback + n_fowards
    #     model1 = experiment(N_lookback=n_lookback, h=15, N_features=14)
    #     run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    #     torch.save(model1.state_dict(),
    #                "D://lab//Data//experiment//MLP//" + str(n_lookback) + "rnn_" + str(
    #                    n_fowards) + ".pth")

    # list_rmse = []
    # for n_fowards in [5, 10, 15]:
    #     for n_lookback in [110,120,130]:
    #         model_path = 'D://lab//Data//experiment//MLP//{}rnn_{}.pth'.format(n_lookback, n_fowards)
    #         n_obs = n_lookback + n_fowards
    #         Tests = list()
    #
    #         for root, dirs, files in os.walk(val_path):
    #             for file in files:
    #                 filepath = os.path.join(root, file)
    #                 # print(filepath)
    #
    #                 data = np.load(filepath, allow_pickle=True)
    #                 data = data.astype(np.float32)
    #
    #                 sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
    #                 Test = sample.values.reshape(-1, n_features)
    #                 Tests.append(Test)
    #
    #         offline_model = load_offline(model_path)
    #
    #         if mode == 'baseline':
    #             predict_value, true_value, _ = eval_onTest(test_model=offline_model,
    #                                                        Testdata=np.concatenate(Tests, axis=0))
    #
    #
    #         value_p = predict_value[:,[0,3,6,9]]
    #         value_t = true_value[:,[0,3,6,9]]
    #         rmse = math.sqrt(mean_squared_error(value_p, value_t))
    #
    #         MAPE = np.sum(abs(value_p - value_t) / value_t) / value_t.size * 100
    #
    #         result = {'RMSE': rmse, 'MAPE': MAPE}
    #         print(result)
    #         df = pd.DataFrame(result, index=['n_forwards:{}-n_lookback:{}'.format(n_fowards, n_lookback)])
    #         df.to_csv(result_path + '//score_val.csv', mode='a', header=False)
    #
    #         list_rmse.append(rmse)


    mode = 'offline'

    '''
        for mode in ['offline','dynamic_online']:
            for p in [4]:
                lr = pow(10,-p)
                for rate in [0.6]:
                    Tests = list()



                    for root, dirs, files in os.walk(data_path):
                        for file in files:
                            filepath = os.path.join(root, file)
                            # print(filepath)
                            date = filepath.split("\\", 1)[0][-4:]
                            carno = filepath.split("\\", 1)[1][:-4]
                            if carno =='6':
                            # if date == '0301':
                                data = np.load(filepath, allow_pickle=True)
                                data = data.astype(np.float32)
                                # print(filepath)
                                n_obs = n_lookback + n_fowards
                                sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
                                Test = sample.values.reshape(-1, n_features)
                                Tests.append(Test)


                    offline_model = load_offline(model_path+'{}rnn_{}.pth'.format(n_lookback,n_fowards))


                    # offline_model = MLP(input_size=n_features*n_lookback,
                    #                                   h_size=H_SIZE).to(DEVICE)  #
                    subpath = result_path + '//{}//base_{}////{}//'.format(mode,n_fowards,date)
                    if not os.path.exists(subpath):
                        os.makedirs(subpath)

                    if mode == 'dynamic_online':
                        predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                                   Testdata = np.concatenate(Tests, axis=0),
                                                                                   mode = 'online',
                                                                                   is_resample=True,
                                                                                   rate=rate,
                                                                                   mylr=lr
                                                                                   )

                    if mode == 'offline':
                        predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                   Testdata=np.concatenate(Tests, axis=0))
                    if mode == 'static_online':
                        predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                    Testdata=np.concatenate(Tests, axis=0),
                                                                    mode='online')
                    # run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
            #         np.save(subpath + '{}p.npy'.format(carno), predict_value)
            #         np.save(subpath + '{}t.npy'.format(carno), true_value)
            #         rmse = math.sqrt(mean_squared_error(predict_value, true_value))
            #
            #         MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100
            #
            #         result = {'RMSE': rmse, 'MAPE': MAPE}
            #
            # # df = pd.DataFrame(result, index=['car:{}-date:{}-n:{}-lr:{}'.format(carno, date, n, lr)])
            #         df = pd.DataFrame(result, index=['car:{}-date:{}-rate:{}-trick:{}'.format(carno, date, rate, trick)])
            #         df.to_csv(result_path + '//{}//base_{}//score.csv'.format(mode,n_fowards), mode='a', header=False)

                    # error = abs(predict_value - true_value) / true_value
                    # plt.plot(pd.DataFrame(error.mean(axis=1)).ewm(span=1000).mean())
                    # plt.show()
        #
        #
        #
    '''

    for n_fowards, n_lookback in [(10, 60)]:
        cdf = get_cdf(n_lookback, n_fowards)
        n_obs = n_lookback + n_fowards


    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)




            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)

            sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)


            date = filepath.split("\\", 1)[0][-4:]
            carno = filepath.split("\\", 1)[1][:-4]
            # if date == '0226' and carno == '6':
            #     continue


            for n in range(10, 11):
                for v in [0.6]:
                    for lr in [0.00001]:
                        Test = sample.values.reshape(-1, n_features)

                        offline_model = load_offline(model_path+'{}rnn_{}.pth'.format(n_lookback,n_fowards))
                        # offline_model = RNN(input_size=INPUT_SIZE,
                        #           h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  # 加载模型到对应设备

                        subpath = result_path + '//{}//base_10//{}//'.format(mode, date)
                        if not os.path.exists(subpath):
                            os.makedirs(subpath)


                        if mode == 'offline':
                            predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                       Testdata=Test)

                        if mode == 'dynamic_online':
                            predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                   Testdata = Test,
                                                                   mode = 'online',
                                                                   is_resample=True,
                                                                   rate=v,
                                                                   mylr=lr
                                                                   )
                        if mode == 'static_online':
                            predict_value, true_value, _ = eval_onTest(test_model=offline_model,
                                                                       Testdata=Test,
                                                                       mode='online')

                        # np.save(subpath+ '{}p.npy'.format(carno), predict_value)
                        # np.save(subpath+ '{}t.npy'.format(carno), true_value)
                        rmse = math.sqrt(mean_squared_error(predict_value, true_value))

                        MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100

                        result = {"rate":v,
                                  "lr":lr,
                                  "date":date,
                                  "carno":carno,
                                  "epoch": n-1,
                                  'RMSE': rmse,
                                  'MAPE': MAPE}

                        df = pd.DataFrame(result, index=['rate:{}-lr:{}-date:{}-no:{}'.format(v, lr, date, carno)])
                        to_path = result_path + '//{}//base_10//score.csv'.format(mode)
                        if os.path.exists(to_path):
                            df.to_csv(to_path,
                                      line_terminator="\n",
                                      mode='a', index=None, header=False)
                        else:
                            df.to_csv(to_path,
                                      line_terminator="\n",
                                      mode='a', index=None)
                        # plot_result(predict_value, true_value, n_fowards, title=filepath)
                        # result = {'RMSE': np.array(list_rmse).mean(), 'MAPE': np.array(list_mape).mean}

