import torch
import torch.nn as nn

import numpy as np
import model
from matplotlib import pyplot as plt
from ftrl import FTRL
import matplotlib.animation
import math, random
from pandas import concat


# Parameters
# model parameters-----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 6 # rnn 的输入维度
H_SIZE = 48 # of rnn 隐藏单元个数
N_LAYERS = 2

# input parameters------------------------------------------------------------------
# batchsize = 8
TIME_STEP = 30 # rnn 时序步长数
h = 5
n_feature = 6
n_look_back = 30
n_obs = n_feature * n_look_back

# optimizer parameters-----------------------------------------------------
ftrl_alpha = 1.0
ftrl_beta = 1.0
ftrl_l1 = 1.0
ftrl_l2 = 1.0

# other parameters-----------------------------------------------------
car45 = [8,9,16,17]
eplison = 0.08
resample = True

list_epoch = []

np.seterr(divide='ignore',invalid='ignore')




# h_state = torch.randn(N_LAYERS, batchsize, H_SIZE) #初始化隐藏层状态
# c_state = torch.randn(N_LAYERS, batchsize, H_SIZE)

# define model
rnn = model.RNN(input_size=INPUT_SIZE,
                h_size=H_SIZE, n_layers=N_LAYERS).to(DEVICE)  #加载模型到对应设备
# optimizer = FTRL(rnn.parameters(), alpha=ftrl_alpha, beta=ftrl_beta, l1=ftrl_l1, l2=ftrl_l2)
# optimizer = torch.optim.SGD(rnn.parameters(),lr=0.1) # SGD优化，
optimizer = torch.optim.SGD(rnn.parameters(),lr=0.01,momentum=0.5) # adam优化，
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差
h_state = torch.zeros(N_LAYERS, 8, H_SIZE).to(DEVICE)
c_state = torch.zeros(N_LAYERS, 8, H_SIZE).to(DEVICE)


def myscaler(data, old_dict=None):
    list_data = []
    new_dict = {}
    weight_old = 0.3
    weight_all = 1
    if old_dict is not None:
        for k in range(data.shape[1]):
            dim_data = data[:, k]
            old_dim = old_dict[k]
            old_max = old_dim[0]
            old_min = old_dim[1]
            dim_max = old_max
            dim_min = old_min
            new_max = max(dim_data)
            new_min = min(dim_data)
            if new_max > old_max:
               # weight_all = 1.1
               dim_max = new_max
            if new_min < old_min:
               dim_min = new_min
            # dim_max = weight_old*old_max + (weight_all-weight_old)*new_max
            # dim_min = weight_old*old_min + (weight_all-weight_old)*new_min


            dim_data = (dim_data - dim_min)/(dim_max - dim_min)
            list_data.append(dim_data)
            new_dict[k] = (dim_max, dim_min)
            np_data = np.array(list_data).T
    else:
        for k in range(data.shape[1]):
            dim_data = data[:, k]
            dim_max = max(dim_data)
            dim_min = min(dim_data)
            dim_data = (dim_data - dim_min) / (dim_max - dim_min)
            list_data.append(dim_data)
            new_dict[k] = (dim_max, dim_min)
            np_data = np.array(list_data).T
    return np_data, new_dict



def mystand_scaler(data, old_dict=None):
    list_data = []
    new_dict = {}
    weight_old = 0.3
    weight_all = 1
    if old_dict is not None:
        for k in range(data.shape[1]):
            dim_data = data[:, k]
            old_dim = old_dict[k]
            old_mean = old_dim[0]
            old_min = old_dim[1]

            new_mean = np.mean(dim_data)
            new_std = np.std(dim_data)

            dim_mean = new_mean
            dim_std = new_std

            # dim_max = weight_old*old_max + (weight_all-weight_old)*new_max
            # dim_min = weight_old*old_min + (weight_all-weight_old)*new_min


            dim_data = (dim_data - dim_mean)/dim_std
            list_data.append(dim_data)
            new_dict[k] = (dim_mean, dim_std)
            np_data = np.array(list_data).T
    else:

        for k in range(data.shape[1]):
            dim_data = data[:, k]
            dim_mean = np.mean(dim_data)
            dim_std = np.std(dim_data)
            if  dim_std == 0:
                dim_std = 0.1



            dim_data = (dim_data - dim_mean) / dim_std
            list_data.append(dim_data)
            new_dict[k] = (dim_mean, dim_std)
            np_data = np.array(list_data).T


    return np_data, new_dict




dataset1 = []
dataset2 = []
dataset3 = []
dataset4 = []
for index in range(4):
    data1 = np.load('F:/MH/Data/temperature/0109_temp/multi-8/'+ str(index)+'.npy')
    data3 = np.load('F:/MH/Data/temperature/0109_temp/multi-16/' + str(index) + '.npy')  #8、9车并列
    data3 = data3[70:,:]
    value1 = np.concatenate((data1, data3), axis=0)
    data2 = np.load('F:/MH/Data/temperature/0109_temp/multi-9/'+ str(index)+'.npy')
    data4 = np.load('F:/MH/Data/temperature/0109_temp/multi-17/' + str(index) + '.npy')
    data4 = data3[70:, :]
    value2 = np.concatenate((data2, data4), axis=0)
    dataset1.append(value1)
    dataset2.append(value2)

truevalue = []
predictedvalue = []
dict_scaler = None
for t in range(4):
    stream = np.zeros((1, 6))  # 初始化训练批次
    stream_p = np.zeros((1, 6))  # 初始化预测批次
    for i in range(4):  # 积攒 训练流
        stream1 = dataset1[i]
        stream1 = stream1[t:t+n_look_back+h, :]
        stream2 = dataset2[i]
        stream2 = stream2[t:t+n_look_back+h, :]
        stream = np.concatenate((stream, stream1), axis=0)
        stream = np.concatenate((stream, stream2), axis=0)
    stream = np.delete(stream, 0, axis=0)
    # scaled_stream, dict_scaler = myscaler(stream, dict_scaler)
    scaled_stream, dict_scaler = mystand_scaler(stream, None)
    # scaled_stream, dict_scaler = myscaler(stream)

    scaled = scaled_stream.reshape(8, n_look_back+h, n_feature)

    train_X = scaled[:, 0: n_look_back, :]
    train_X = torch.from_numpy(train_X).float()
    train_X = train_X.to(DEVICE)

    train_Y = scaled[:, -1, 0:3]
    train_Y = torch.from_numpy(train_Y).float()
    train_Y = train_Y.to(DEVICE)

    rnn.train()  #模型训练模式

    # train_p, h_state, c_state = rnn(train_X, h_state, c_state)  # rnn output
    # loss = criterion(train_p.cpu(), train_Y)
    # # 这三行写在一起就可以
    # optimizer.zero_grad()
    # loss.backward(retain_graph = True)
    # optimizer.step()


    epoch_of_persample = 6
    step = 0
    while step < epoch_of_persample:
        step += 1
        train_p, h_state, c_state = rnn(train_X, h_state, c_state)  # rnn output
        h_state = h_state.data
        c_state = c_state.data
        loss = criterion(train_p.cpu(), train_Y)
        # 这三行写在一起就可以
        optimizer.zero_grad()
        loss.backward()
        a = optimizer.param_groups[0]['params']
        optimizer.step()


    if t >= h and resample:
        delta = max(abs((predictedvalue[t-h] - truevalue[t-h])/truevalue[t-h]))
        if delta >= eplison:
            # if delta>2*eplison:
            #     epoch=5
            # else:
            epoch_of_targetsample = int((1-np.exp(-2*(delta-eplison)/eplison)) * 5)+1
                # epoch=int(4*(delta-eplison)/eplison)+1   # 乘一个分布， 分析不做样本选取最大误差出现的地方   或者根据差分看最大误差
            list_epoch.append(epoch_of_targetsample)
            # print(epoch_of_targetsample)
            step = 0
            while step < epoch_of_targetsample:
                step += 1
                train_p,_, _ = rnn(train_X[[0]],h_state,c_state)  # train_X[0]为额外训练的样本
                # h_state = h_state.data
                # c_state = c_state.data
                loss = criterion(train_p.cpu(), train_Y[[0]])  # torch.size [1,3]
                # 这三行写在一起就可以
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 这三行写在一起就可以





    if (t+1) % 20 == 0: #每训练20个批次可视化一下效果，并打印一下loss
        print("EPOCHS: {},Loss:{:4f}".format(t, loss))


    stream_p = dataset1[0]  #预测流 8车 电机1
    stream_px = stream_p[t+h:t+h+n_look_back, :]
    stream_py = stream_p[t+h+n_look_back+h, 0:3]
    truevalue.append(stream_py)
    # scaled_stream_p, dict_scaler_p = myscaler(stream_px, dict_scaler)
    scaled_stream_p, dict_scaler_p = mystand_scaler(stream_px, None)


    test_X = torch.from_numpy(scaled_stream_p[np.newaxis, :, :]).float()
    test_X = test_X.to(DEVICE)
    # Test the model
    rnn.eval()
    with torch.no_grad():
        test_p, _ , _ = rnn(test_X, h_state, c_state)
        test_p = test_p.cpu().data.numpy().flatten()
        y = []
        for i in range(test_p.shape[0]):

            # ymax = dict_scaler_p[i][0]
            # ymin = dict_scaler_p[i][1]
            # y.append(test_p[i]*(ymax-ymin)+ymin)
            ymean = dict_scaler_p[i][0]
            ystd = dict_scaler_p[i][1]
            y.append(test_p[i] * ystd + ymean)
        predictedvalue.append(y)



plt.figure(figsize=(20, 6))



targetPlot = plt.plot(truevalue[h:600], label='target', color='red', marker='.', linestyle='-')
trainplot = plt.plot(predictedvalue[h:600], label='retrain', color='blue', marker='.', linestyle=':')
plt.plot(truevalue[0:600-h], label='history', color='green', marker='.', linestyle=':')


plt.legend()
plt.ylabel('value', fontsize=15)
plt.xlabel('Time Step(1 min)', fontsize=15)
# pyplot.grid()
plt.show()



list_epoch.sort()
plt.plot(list_epoch)
plt.show()























