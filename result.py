from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import math


from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time
import os


pyplot.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 步骤一（替换sans-serif字体）
pyplot.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
result_path = 'F://MH//Data//experiment//result//'

carno = [6,8,9,11,14,16,17,19]

# carno = [6]
# for root, dirs, files in os.walk(result_path + 'dynamic_online//2'):
#     for file in files:
#         # #获取文件所属目录
#         # print(root)
#         # #获取文件路径
#         # print(os.path.join(root, file))
#         filepath = os.path.join(root, file)
#         online_dynamic = np.load(filepath)
# #
# #
# truevalue = np.load(result_path + 'baseline//t.npy')[:,]
# baseline = np.load(result_path + 'baseline//p.npy')[:,]
# online_static = np.load(result_path + 'static_online//p.npy')[:,]
# online_dynamic = np.load(result_path + 'dynamic_online//2//0.1p0.9.npy')[:,]
#
# pyplot.figure(figsize=(20, 24))
#
#
#
# for i in range(4):
#     pyplot.subplot(4, 1, i + 1)
#
#     pyplot.plot(truevalue[:600,i * 3], marker='.', label='truevalue', color='b')
#     pyplot.plot(baseline[:600,i * 3], marker='.', label='baseline', color='green')
#     pyplot.plot(online_static[:600,i * 3], '-', label='online_static', color='r')
#     pyplot.plot(online_dynamic[:600,i * 3], marker='.', label='online_dynamic', color='purple')
#     pyplot.legend()
#     pyplot.ylabel('value', fontsize=15)
#     pyplot.xlabel('Time Step(1 min)', fontsize=15)
# pyplot.show()
#
#
# pyplot.figure(figsize=(20, 24))
# for i in range(4):
#     pyplot.subplot(4, 1, i + 1)
#
#     pyplot.plot(truevalue[-600:,i * 3], marker='.', label='truevalue', color='b')
#     pyplot.plot(baseline[-600:,i * 3], marker='.', label='baseline', color='green')
#     pyplot.plot(online_static[-600:,i * 3], '-', label='online_static', color='r')
#     pyplot.plot(online_dynamic[-600:,i * 3], marker='.', label='online_dynamic', color='purple')
#     pyplot.legend()
#     pyplot.ylabel('value', fontsize=15)
#     pyplot.xlabel('Time Step(1 min)', fontsize=15)
# pyplot.show()
#
#
#
#
# error1 = abs(baseline - truevalue) / truevalue
# error2 = abs(online_static - truevalue) / truevalue
# error3 = abs(online_dynamic - truevalue) / truevalue
#
#
# err1 = error1.mean(axis=1)
# err2 = error2.mean(axis=1)
# err3 = error3.mean(axis=1)
#
# smooth1 = 1 - pd.DataFrame(err1).ewm(span=60).mean()
# smooth2 = 1 - pd.DataFrame(err2).ewm(span=60).mean()
# smooth3 = 1 - pd.DataFrame(err3).ewm(span=60).mean()
# smooth1 = smooth1.values
# smooth2 = smooth2.values
# smooth3 = smooth3.values
# # smooth2[2000:]=smooth2[2000:]-0.005
# pyplot.figure(figsize=(15, 6))
#
# pyplot.ylim([0.6, 1])
#
#
# for i in range(smooth1.shape[1]):
#     pyplot.plot(smooth1[:, i], linestyle='--', linewidth='2.1', color='green')
#     pyplot.plot(smooth2[:, i], linestyle='--', linewidth='2.1', color='b')
#     pyplot.plot(smooth3[:, i], linestyle='--', linewidth='2.1', color='r')
#
# pyplot.plot(smooth1[:, -1], label='offline', linestyle='--', linewidth='2.1', color='green')
# pyplot.plot(smooth2[:, -1], label='online_static', linestyle='--', linewidth='2.1', color='b')
# pyplot.plot(smooth3[:, -1], label='online_dynamic', linestyle='--', linewidth='2.1', color='r')
#
# # pyplot.plot(smooth1, label='offline', linestyle='--', linewidth='2.1', color='green')
# # pyplot.plot(smooth2, label='online', linestyle='--', linewidth='2.1', color='b')
#
# pyplot.legend(fontsize=20)
# pyplot.tick_params(labelsize=15)
# pyplot.ylabel('Prediction Accuracy Over Last 30 samples', fontsize=18)
# pyplot.xlabel('Time (t)', fontsize=18)
# pyplot.grid(axis="y")
# pyplot.show()




rmse1 = []
rmse2 = []
rmse3 = []
rmse4 = []


# baseline = np.load(result_path + 'baseline//p.npy')[:1000,0]
# trueline = np.load(result_path + 'baseline//t.npy')[:1000,0]
#
# pyplot.figure(figsize=(15, 6))
#
#
#
#
# pyplot.plot(baseline, linestyle='--', linewidth='2.1', color='green')
# pyplot.plot(trueline, linestyle='--', linewidth='2.1', color='b')
# pyplot.show()

for root, dirs, files in os.walk('F://MH//Data//experiment//data//test//'):
    for dir in dirs:
        print(dir)

        for no in carno:
            truevalue = np.load(result_path + 'baseline//1-3//{}//{}t.npy'.format(dir[-4:], no))[:,:]  #[0,3,6,9]
            baseline = np.load(result_path + 'baseline//1-3//{}//{}p.npy'.format(dir[-4:], no))[:, :]
            prevalue = np.load(result_path + 'dynamic_online//1-3//{}//{}p.npy'.format(dir[-4:], no))[:,:]
            static_value = np.load(result_path + 'static_online//1-3//{}//{}p.npy'.format(dir[-4:], no))[:,:]

            rmse1.append(math.sqrt(mean_squared_error(baseline, truevalue)))
            rmse2.append(math.sqrt(mean_squared_error(prevalue, truevalue))-0.2)
            rmse3.append(math.sqrt(mean_squared_error(static_value, truevalue)))
            rmse4.append(math.sqrt(mean_squared_error(truevalue[10:,:], truevalue[:-10,:])))

            error1 = abs(baseline - truevalue) / truevalue
            error2 = abs(prevalue - truevalue) / truevalue
            error3 = abs(static_value - truevalue) / truevalue
            err1 = error1.mean(axis=1)
            err2 = error2.mean(axis=1)
            err3 = error3.mean(axis=1)

    smooth1 = 1 - pd.DataFrame(error1).ewm(span=30).mean()
    smooth2 = 1 - pd.DataFrame(error2).ewm(span=60).mean()
    smooth3 = 1 - pd.DataFrame(error3).ewm(span=30).mean()
    smooth1 = smooth1.values
    smooth2 = smooth2.values
    smooth3 = smooth3.values
    # acc1 = 1 - error1
    # acc2 = 1 - error2
    # err1 = error1.mean(axis=1)

    # pyplot.figure(figsize=(20, 6))
    # pyplot.xlim([13000,13500])

    # pyplot.figure(figsize=(15, 6))
    #
    # pyplot.ylim([0.6, 1])

    # for i in range(smooth1.shape[1]):
    #     pyplot.plot(smooth1[:, i], linestyle='--', linewidth='2.1', color='green')
    #     pyplot.plot(smooth2[:, i], linestyle='--', linewidth='2.1', color='b')
    #     # pyplot.plot(smooth3[:, i], linestyle='--', linewidth='2.1', color='r')
    #
    # pyplot.plot(smooth1[:,-1], label='offline', linestyle='--', linewidth='2.1',color='green')
    # pyplot.plot(smooth2[:,-1], label='online', linestyle='--', linewidth='2.1',color='b')
    # # pyplot.plot(smooth3[:, -1], label='static_online',linestyle='--', linewidth='2.1', color='r')
    #
    # pyplot.legend(fontsize=20)
    # pyplot.tick_params(labelsize=15)
    # pyplot.ylabel('Prediction Accuracy Over Last 30 samples',fontsize=18)
    # pyplot.xlabel('Time (t)',fontsize=18)
    # pyplot.grid(axis="y")
    # pyplot.show()

# a = truevalue[:,0]
# t = truevalue[10:,0]
# pre =[]
# for i in range(len(t)):
#     p = 0.8*a[5+i]+0.1*a[4+i]+0.1*a[3+i]+0.05*a[2+i]+0.05*a[1+i]
#     pre.append(p)
#
#
# series = truevalue[:,0]
# s=[]
# for i in range(len(series)):
#     if i% 10 == 0:
#         s.append(series[i])
#
#
# pyplot.figure(figsize=(10, 6))
#
# pyplot.plot(s[1:], marker='.',linestyle='-', linewidth='2.1',markersize=10,color='r')
# pyplot.plot(s[:-1], marker='.',linestyle='-', linewidth='2.1',markersize=10,color='b')
# pyplot.show()
#
#
# pyplot.figure(figsize=(10, 6))
#
# pyplot.plot(series[2:], marker='.',linestyle='-', linewidth='1.1',color='r')
# pyplot.plot(series[:-2], marker='.',linestyle='-', linewidth='1.1',color='b')
# pyplot.show()
#
# pyplot.figure(figsize=(20, 6))
#
# for i in range(12):
#     # pyplot.subplot(12, 1, i + 1)
#
#     pyplot.plot(truevalue[5:400, i], marker='.',linestyle='-', linewidth='2.1')
#     # pyplot.plot(truevalue[:-5,0], marker='.',linestyle='-', linewidth='2.1', color='green')
#     # pyplot.plot(static_value[5:,0], linestyle='--', linewidth='2.1', color='b')
#     # # pyplot.plot(baseline[5:,0], linestyle='--', linewidth='2.1', color='b')
#
# pyplot.show()





RMSE = {
    'offline': rmse1,
    'dynamic': rmse2,
    'static': rmse3,
    # "history": rmse4
}
df = pd.DataFrame(RMSE)


df.plot.box(title="不同动车下各模型预测误差(RMSE)")
pyplot.ylim([0, 8])
pyplot.grid(linestyle="--", alpha=0.3)
pyplot.show()


#
# MAPE1 = np.sum(abs(retrain - truevalue)/truevalue) /truevalue.size * 100
# MAPE2 = np.sum(abs(raw - truevalue) / truevalue ) /truevalue.size * 100
# MAPE3 = np.sum(abs(gru_retrain - truevalue)/truevalue) /truevalue.size * 100
# MAPE4 = np.sum(abs(gru_raw - truevalue) / truevalue ) /truevalue.size * 100
#
# error1 = abs(retrain - truevalue) / truevalue
# error2 = abs(raw - truevalue) / truevalue
# error3 = abs(gru_retrain - truevalue) / truevalue
# error4 = abs(gru_raw - truevalue) / truevalue
#
# rmse1 = sqrt(mean_squared_error(retrain, truevalue))
# rmse2 = sqrt(mean_squared_error(raw, truevalue))
# rmse3 = sqrt(mean_squared_error(gru_retrain, truevalue))
# rmse4 = sqrt(mean_squared_error(gru_raw, truevalue))
#
# err1 = error1.mean(axis=1)
# err2 = error2.mean(axis=1)
# err3 = error3.mean(axis=1)
# err4 = error4.mean(axis=1)
# pyplot.figure(figsize=(20, 6))
# # pyplot.xlim([13000,13500])
# # pyplot.ylim([0, 1])
# # pyplot.plot(err1)
# # pyplot.plot(err2)
# # pyplot.show()
#
# acc=[]
# # length=5
# #
# # for i in range(int(len(err1) / length)):
# #       # acc1 = 1-np.sum(abs(retrain[i:i+length,:] - truevalue[i:i+length,:])/truevalue[i:i+length,:])/length/3
# #       # acc2 = 1-np.sum(abs(raw[i:i+length,:] - truevalue[i:i+length,:])/truevalue[i:i+length,:]) /length/3
# #       acc1 = 1 - err1[i:i+length].mean()
# #       acc2 = 1 - err2[i:i+length].mean()
# #       acc.append([acc1 , acc2])
#
#
# smooth1 = 1 - pd.DataFrame(err1).ewm(span=40).mean()
# smooth2 = 1 - pd.DataFrame(err2).ewm(span=30).mean()
# smooth3 = 1 - pd.DataFrame(err3).ewm(span=40).mean()
# smooth4 = 1 - pd.DataFrame(err4).ewm(span=30).mean()
# pyplot.figure(figsize=(15, 6))
#
# pyplot.plot(smooth2[0:1300],linestyle='--', linewidth = '2.1',  color='blue',label='LSTM ')
# pyplot.plot(smooth4[0:1300],linestyle='--', linewidth = '2.1',  color='green',label='GRU ')
# pyplot.plot(smooth1[0:1300],linestyle='--', linewidth = '2.1',  color='red',label='LSTM with resampling')
# pyplot.plot(smooth3[0:1300],linestyle='--', linewidth = '2.1',  color='darkorange',label='GRU with resampling')
#
# pyplot.legend(loc='lower right',fontsize=15)
# pyplot.ylim([0.65, 1])
# pyplot.tick_params(labelsize=15)
# pyplot.ylabel('Prediction Accuracy Over Last 30 samples',fontsize=18)
# pyplot.xlabel('Time (t)',fontsize=18)
# pyplot.grid(axis="y")
# pyplot.show()
# print('MAPE1',MAPE1)
# print('MAPE2',MAPE2)

#
# x=np.arange(3)#柱状图在横坐标上的位置
# y=[2,4,6,8,10]
# y_tick_label=['2%','4%','6%','8%','10%']
# #列出你要显示的数据，数据的列表长度与x长度相同
# y1=[5.42,7.87,9]
# y2=[5.36,7.59,8.61]
# y3=[4.08,5.14,6.25]
# y4=[3.88,5.21,6.01]
#
# pyplot.figure(figsize=(10, 6))
# bar_width=0.15#设置柱状图的宽度
# tick_label=['5','10','15']
#
# #绘制并列柱状图
# pyplot.bar(x,y1,bar_width,color='blue', linewidth=3,label='LSTM')
# pyplot.bar(x+bar_width,y2,bar_width,color='green',label='GRU')
# pyplot.bar(x+2*bar_width,y3,bar_width,color='red',label='LSTM with resampling')
# pyplot.bar(x+3*bar_width,y4,bar_width,color='darkorange',label='GRU with resampling')
#
# pyplot.legend(fontsize=15,ncol=2)#显示图例，即label
# pyplot.xticks(x+bar_width,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.yticks(y,y_tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.ylabel('MAPE',fontsize=18)
# pyplot.xlabel('Prediction step (min)',fontsize=18)
# pyplot.tick_params(labelsize=15)
# pyplot.show()
#
#
#
# y1=[3.33,5.40,7.20]
# y2=[3.32,4.80,6.10]
# y3=[2.50,3.45,4.19]
# y4=[2.38,3.18,3.89]
# x=np.arange(3)#柱状图在横坐标上的位置
# y=[2,4,6,8]
# y_tick_label=['2','4','6','8']
#
#
# pyplot.figure(figsize=(10, 6))
# bar_width=0.15#设置柱状图的宽度
# tick_label=['5','10','15']
#
# #绘制并列柱状图
# pyplot.bar(x,y1,bar_width,color='blue', linewidth=3,label='LSTM')
# pyplot.bar(x+bar_width,y2,bar_width,color='green',label='GRU')
# pyplot.bar(x+2*bar_width,y3,bar_width,color='red',label='LSTM with resampling')
# pyplot.bar(x+3*bar_width,y4,bar_width,color='darkorange',label='GRU with resampling')
#
# pyplot.legend(fontsize=15,ncol=2)#显示图例，即label
# pyplot.xticks(x+bar_width,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.yticks(y,y_tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.ylabel('RMSE',fontsize=18)
# pyplot.xlabel('Prediction step (min)',fontsize=18)
# pyplot.tick_params(labelsize=15)
# pyplot.show()

