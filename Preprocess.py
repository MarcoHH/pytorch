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
            h_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
            c_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)


        r_out, (h_STATE, c_STATE) = self.lstm(x, (h_state, c_state))

        # r_out = r_out.view(r_out.size(0), -1)
        # outs = self.out(r_out)
        outs = self.out(h_STATE[-1])

        # outs = outs[:, -1, :].view(r_out.size(0), -1)
        # outs = outs[:, -1, :]

        return outs, h_STATE, c_STATE


### GENERATE DATA ###
# optimizer parameters-----------------------------------------------------
ftrl_alpha = 1.0
ftrl_beta = 1.0
ftrl_l1 = 1.0
ftrl_l2 = 1.0

# Parameters
# model parameters-----------------------------------------------------------------

INPUT_SIZE = 14  # rnn 的输入维度
H_SIZE = 48  # of rnn 隐藏单元个数
N_LAYERS = 2
N_OUTS = 12
Batch_size = 500


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


    for k in range(X.shape[1]):
        dim_data = X[:, k]

        new_max = max(dim_data)
        new_min = min(dim_data)
        if new_max > scaler.data_max_[k]:
            # weight_all = 1.1
            scaler.data_max_[k] = new_max
            print("update!")

        if new_min < scaler.data_min_[k]:
            scaler.data_min_[k] = new_min
            print("update!")
        scaler.data_range_[k] = scaler.data_max_[k] - scaler.data_min_[k]
    X = (X - scaler.data_min_)/scaler.data_range_

    return X, scaler


def getTrain(N_features, N_lookback, h) -> np.ndarray:
    samples = list()
    data_path = 'F://MH//Data//experiment//train//'
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # #获取文件所属目录
            # print(root)
            # #获取文件路径
            # print(os.path.join(root, file))
            filepath = os.path.join(root, file)

            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)
            data = pd.DataFrame(data)
            # plt.plot(data.values[:,0],'b.')
            data = data.interpolate(method='linear', axis=0, limit=10, inpalce=True)
            # data = data.ewm(span=3).mean()
            data = data.astype(np.float32)
            # data.fillna(method='ffill', inplace=True, limit=100)
            # plt.plot(data.values[:, 0],'r-')

            # data.fillna(method='ffill', inplace=True, limit=5)

            if root[-4:] == '0101':
                value = data.values[750:1400, :]
            elif root[-4:] == '0102':
                value = data.values[550:1200, :]
            else:
                value = data.values[550:1300, :]
            # plt.plot(value, 'r-', label=str(root[-4:]))
            # plt.legend()
            # plt.show()
            sample = series_to_supervised(value, h + N_lookback - 1, 1)
            sample = sample.values
            samples.append(sample)

    value = np.concatenate(samples, axis=0)
    value = value.reshape(-1, N_features)
    scaler = MinMaxScaler(feature_range=(0, 1))
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


def experiment(N_lookback, h, N_features=INPUT_SIZE) -> None:
    # define model
    rnn = RNN(input_size=INPUT_SIZE,
              h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  # 加载模型到对应设备
    lr = 0.1
    # optimizer = torch.optim.Adam(rnn.parameters())  # adam优化，
    optimizer = torch.optim.SGD(rnn.parameters(),lr=lr, momentum = 0.9, weight_decay=0.0001)
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
    for epoch in range(6000):
        if epoch % 100 ==0:
            lr = lr * 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(Train_loader):
            inputs, targets = data
            y_hat, _, _ = rnn(inputs)  # rnn output

            loss = criterion(y_hat.cpu(), targets)
            # 这三行写在一起就可以
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:  # 每训练10个批次可视化一下效果，并打印一下loss
            print("EPOCHS: {},Loss:{:4f}".format(epoch, loss))
            # print(epoch, i,'inputs',inputs.data.size(),
            #       'targets',targets.data.size())

    torch.save(rnn.state_dict(), "F://MH//Data//experiment//model2//" + str(N_lookback) + "rnn_" + str(h) + ".pth")

    # 打印最终的损失值
    # output = rnn(inputs)
    # loss = criterion(output, targets)
    # print(loss.item())

    return None


def load_offline(model_path):
    model = RNN(input_size=INPUT_SIZE,
                h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  # 加载模型到对应设备
    model.load_state_dict(torch.load(model_path))

    return model


def eval_onTrain() -> None:
    h = 10
    n_back = 30
    Train, scaler = getTrain(14, n_back, h)
    train_x = torch.from_numpy(Train[:, 0:n_back * 14].reshape(-1, n_back, 14))
    train_y = torch.from_numpy(Train[:, -14:].reshape(-1, 14))

    train_x = train_x.to(DEVICE)
    offline_model = load_offline('F://MH//Data//experiment//model2//30rnn_10.pth')
    offline_model.eval()
    with torch.no_grad():
        predicted, _, _ = offline_model(train_x[0:800, :, :])

    value = np.concatenate((predicted, train_y[0:800, -2:]), axis=1)

    inv_y = scaler.inverse_transform(value)
    train_y = scaler.inverse_transform(train_y)

    predicted_value = inv_y[h:, 0:3]
    target_value = train_y[h:800, 0:3]
    history_value = train_y[0:800, 0:3]
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


def eval_onTest(test_model, Testdata, is_resample=False):



    optimizer = torch.optim.SGD(offline_model.parameters(), lr=0.1, momentum=0.5)

    criterion = nn.MSELoss()  # 因为最终的结果是一个数值，所以损失函数用均方误差
    lr =0.1
    ae = []
    ae1 =[]
    ae2 = []
    list_lr=[]
    list_epoch = []
    truevalue = []
    predictedvalue = []
    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards
    k_count =0
    _, scaler = getTrain(n_features, n_lookback, n_fowards)

    hstate = torch.rand(2, 1, 48).to(DEVICE)
    cstate = torch.rand(2,1,48).to(DEVICE)

    hstate_p = torch.rand(2, 1, 48).to(DEVICE)
    cstate_p = torch.rand(2, 1, 48).to(DEVICE)

    for t in range(epoch):
        list_lr.append(lr)
        k_count = k_count+1
        stream = Testdata[t * n_obs:(t + 1) * n_obs]

        # 计算时间序列信息熵
        series_shape = stream.shape[1] - 2
        multi_entropy=np.zeros(series_shape)
        for i in range(series_shape):
            multi_entropy[i] = binned_entropy(stream[:,i], 4)


        entropy = max(multi_entropy[[0,3,6,9]])-min(multi_entropy[[0,3,6,9]])
        ae.append(entropy)
        ae1.append(max(multi_entropy[[0,3,6,9]]))
        ae2.append(min(multi_entropy[[0,3,6,9]]))
        # 综合所有信息熵

        # scaled_stream,scaler = scaler_update(stream, scaler)

        scaled_stream = scaler.transform(stream)
        scaled_stream = scaled_stream.reshape(-1, n_obs, n_features)

        X = scaled_stream[:, 0: n_lookback, :]

        X = torch.from_numpy(X)
        X = X.to(DEVICE)

        Y = scaled_stream[:, -1, :-2]
        Y = torch.from_numpy(Y)
        Y = Y.to(DEVICE)

        test_model.train()  # 模型训练模式


        epoch_of_persample = 0
        step = 0
        while step < epoch_of_persample:
            step += 1
            train_p, hstate, cstate = test_model(X, hstate, cstate )  # rnn output
            hstate = hstate.detach()
            cstate = cstate.detach()
            loss = criterion(train_p.cpu(), Y)
            # 这三行写在一起就可以
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if t % 60 ==0:
        #     lr = lr*0.9
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        lr = lr*0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if t >= n_fowards and is_resample:
            delta = max(abs((predictedvalue[t - n_fowards] - truevalue[t - n_fowards]) / truevalue[t - n_fowards]))
            if delta >= eplison:
                # if delta>2*eplison:
                #     epoch=5
                # else:
                epoch_of_targetsample = int((1 - np.exp(-2 * (delta - eplison) / eplison)) * 4) + 1
                # epoch=int(4*(delta-eplison)/eplison)+1   # 乘一个分布， 分析不做样本选取最大误差出现的地方   或者根据差分看最大误差

                # print(epoch_of_targetsample)
                # if entropy > 0.8:
                #     epoch_of_targetsample = int((1 - np.exp(-2 * (entropy - 0.8) / 0.8)) * 4) + 1
                list_epoch.append(epoch_of_targetsample)

                if epoch_of_targetsample > 3 and k_count>10:
                    # nonlocal lr
                    k_count = 0
                    lr = 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                step = 0
                while step < epoch_of_targetsample:
                    step += 1
                    lr_local = lr / (math.sqrt(step))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_local
                    train_p, hstate, cstate = test_model(X, hstate,cstate)  #
                    hstate = hstate.detach()
                    cstate = cstate.detach()
                    loss = criterion(train_p.cpu(), Y)  # torch.size [1,3]
                    # 这三行写在一起就可以
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 这三行写在一起就可以

        # if (t + 1) % 20 == 0:  # 每训练20个批次可视化一下效果，并打印一下loss
        #     print("EPOCHS: {},Loss:{:4f}".format(t, loss))

        stream_p = Testdata[(t + n_fowards) * n_obs:(t + n_fowards + 1) * n_obs]  # 预测流 8车 电机1
        stream_py = stream_p[-1, :-2]
        truevalue.append(stream_py)

        stream_px = stream_p[0: n_lookback, :]



        # X, scaler = scaler_update(stream_px, scaler)

        X = scaler.transform(stream_px)

        X = X.reshape(-1, n_lookback, n_features)

        X = torch.from_numpy(X).to(DEVICE)

        # Test the model
        test_model.eval()
        with torch.no_grad():
            test_p, hstate_p, cstate_p = test_model(X, hstate_p, cstate_p)
            # test_p, hstate_p, cstate_p = test_model(X)
            test_p = test_p.cpu().data.numpy().flatten()
            # y = []
            #
            # for i in range(test_p.shape[0]):
            #
            #     ymax = dict_scaler_p[i][0]
            #     ymin = dict_scaler_p[i][1]
            #     y.append(test_p[i]*(ymax-ymin)+ymin)
            y = test_p * scaler.data_range_[:-2] + scaler.data_min_[:-2]
            predictedvalue.append(y)

    predictedvalue = np.array(predictedvalue)
    truevalue = np.array(truevalue)
    # list_epoch.sort()
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ae)
    plt.plot(ae1)
    plt.plot(ae2)
    plt.subplot(2, 1, 2)
    plt.plot(truevalue[:,[0,3,6,9]])
    plt.show()
    return predictedvalue, truevalue, ae, list_epoch


def plot_result(predictedvalue, truevalue, h, title):
    plt.figure(figsize=(20, 24))

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(predictedvalue[h:, i * 3:(i + 1) * 3], marker='.', label='retrain')
        plt.plot(truevalue[h:, i * 3:(i + 1) * 3], '-', label='target')
        plt.plot(truevalue[0: - h, i * 3:(i + 1) * 3], ':', label='history')

        signal1 = predictedvalue[h:, i * 3]
        signal2 = truevalue[h:, i * 3]
        num_peak1 = signal.find_peaks(signal1, distance=60, width=10, height=80)
        num_peak2 = signal.find_peaks(signal2, distance=60, width=10, height=80)

        for ii in range(len(num_peak1[0])):
            plt.plot(num_peak1[0][ii], signal1[num_peak1[0][ii]], 'r*', markersize=10)

        for ii in range(len(num_peak2[0])):
            plt.plot(num_peak2[0][ii], signal2[num_peak2[0][ii]], 'b.', markersize=10)

        plt.legend()
        plt.ylabel('value', fontsize=15)
        plt.xlabel('Time Step(1 min)', fontsize=15)
        plt.title(title)
    plt.savefig(result_path + "{}-{}.png".format(date, carno))
    plt.show()


    # print('MSE--------{}', 'MAPE------{}'.format(mse, MAPE))
    # plt.plot(err)
    # plt.show()
    # Test_dataset = Test.reshape(-1,  n_features * (n_lookback + n_fowards))
    # target = Test_dataset[:,-n_features:-2]

    # signal = predictedvalue[0]
    # plt.plot(ae)
    # plt.show()
    return None


#
if __name__ == "__main__":


    # offline_model = load_offline('F://MH//Data//experiment//model//30rnn_5.pth')
    # run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
    eplison = 0.08
    data_path = 'F://MH//Data//experiment//test1//'
    result_path ='F://MH//Data//experiment//result//'
    n_features, n_lookback, n_fowards = 14, 30, 5
    n_obs = n_lookback + n_fowards

    # experiment(N_lookback=30, h=10 ,N_features=14)



    logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO)
    Tests = list()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)

            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)
            data = pd.DataFrame(data)
            data = data.interpolate(method='linear', axis=0, limit=10, inpalce=True)
            # data = data.ewm(span=2).mean()
            data = data.astype(np.float32)
            # data.fillna(method='ffill', inplace=True, limit=5)
            value = data.values[600:1300, :]

            sample = series_to_supervised(value, n_fowards + n_lookback - 1, 1)
            Test = sample.values.reshape(-1, n_features)
            Tests.append(Test)

            # offline_model = RNN(input_size=INPUT_SIZE,
            #                     h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)
            # optimizer = torch.optim.Adam([{'params': offline_model.lstm.parameters(), 'lr': 0.2},
            #                              {'params': offline_model.out.parameters(), 'lr': 0.1}
            #                              ])  # adam优化，
            # for m in offline_model.modules():
            #     if isinstance(m, nn.LSTM):
            #         nn.init.normal(m.weight.data)
            #
            #         m.bias.data.fill_(0)
            #     elif isinstance(m, nn.Linear):
            #         m.weight.data.normal_()  # 全连接层参数初始化


            # predict_value, true_value, _, _ = eval_onTest(test_model=offline_model, Testdata=Test)
            # rmse = math.sqrt(mean_squared_error(predict_value, true_value))
            #
            # error = abs(predict_value - true_value) / true_value
            # err = error.mean(axis=1)
            # MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100
            # result = {'RMSE': rmse, 'MAPE': MAPE}
            # df = pd.DataFrame(result, index=[filepath])


            date = filepath.split("\\",1)[0][-4:]
            carno = filepath.split("\\",1)[1][:-4]

            # df.to_csv(result_path + 'score.csv', mode='a', header=False)
            # np.save(result_path + '{}-{}p.npy'.format(date, carno), predict_value)
            # np.save(result_path + '{}-{}t.npy'.format(date, carno), true_value)

            # logging.info(filepath)
            # logging.info('RMSE--------{0}--MAPE------{1}'.format(rmse, MAPE))



            # plot_result(predict_value, true_value, n_fowards, title=filepath)





    offline_model = load_offline('F://MH//Data//experiment//model2//30rnn_5.pth')

    predict_value, true_value, _, _ = eval_onTest(test_model=offline_model, Testdata=np.concatenate(Tests, axis=0))
    rmse = math.sqrt(mean_squared_error(predict_value, true_value))
    #

    MAPE = np.sum(abs(predict_value - true_value) / true_value) / true_value.size * 100
    result = {'RMSE': rmse, 'MAPE': MAPE}
    df = pd.DataFrame(result, index=[filepath])


    df.to_csv(result_path + 'score.csv', mode='a', header=False)
    np.save(result_path + 'p.npy', predict_value)
    np.save(result_path + 't.npy', true_value)



    plot_result(predict_value, true_value, n_fowards, title=filepath)

# test_y = Test.reshape(-1, n_features * (n_lookback + n_fowards))[:,-n_features:-2]  # unscaled target
# scaled = scaler.transform(Test)
# Test_dataset = scaled.reshape(-1,  n_features * (n_lookback + n_fowards))  #(n_samples ,n_features* timestep)
# test_x = torch.from_numpy(Test_dataset[:, 0:n_lookback * n_features].reshape(-1, n_lookback, n_features))
# concat_y = Test_dataset[:, -n_features:].reshape(-1, n_features)
#
# test_x = test_x.to(DEVICE)
#
# with torch.no_grad():
#     predicted, _, _ = offline_model(test_x[0:800, :, :])
#
# value = np.concatenate((predicted, concat_y[0:800, -2:]), axis=1)
#
# inv_y = scaler.inverse_transform(value)
#
#
#
#
#
# predicted_value = inv_y[5:, 0:3]
# target_value = test_y[5:800, 0:3]
# history_value = test_y[0:800, 0:3]
# plt.figure(figsize=(20, 6))
# plt.plot(predicted_value, '.', label='predict')
# plt.plot(target_value, '-', label='target')
# plt.plot(history_value, ':', label='history')
# plt.legend()
# plt.ylabel('value', fontsize=15)
# plt.xlabel('Time Step(1 min)', fontsize=15)
# plt.show()
