from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import math
from numpy import *


from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time
import os


pyplot.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 步骤一（替换sans-serif字体）
pyplot.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
result_path = 'F://MH//Data//experiment//result//'

carno = [6,8,9,11,14,16,17,19]





data = pd.read_csv(result_path +'score_val.csv', header =None,usecols=[1,2,5,6,9,10], encoding='gb2312')
data = data.values
# pyplot.figure(figsize=(20, 24))

# pyplot.plot(data[:,0], marker='o',markersize=10, label='5step')
# pyplot.plot(data[:,2], marker='^',markersize=10, label='5step')
# pyplot.plot(data[:,4], marker='x', markersize=10, label='5step')
#
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
# pyplot.legend(fontsize=15,ncol=2)#显示图例，即label
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# # pyplot.yticks(y,y_tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.ylabel('MAPE',fontsize=18)
# pyplot.xlabel('The value of ls (min)',fontsize=18)
# pyplot.tick_params(labelsize=15)
# pyplot.show()










#
# pyplot.subplot(1, 3, 1)
#
# pyplot.plot(data[:12,0], marker='o',markersize=10,linestyle='-', label='RMSE')
# pyplot.plot(data[:12,1], marker='^',markersize=10,linestyle='--', label='MAPE(%)')
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
# pyplot.legend(fontsize=15,ncol=2,loc=1)#显示图例，即label
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.yticks([1.8,2.0,2.2,2.4,2.6,2.8])#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.ylabel('Metric',fontsize=18)
# pyplot.xlabel('The value of ls (min)',fontsize=18)
# pyplot.tick_params(labelsize=15)
#
#
# # pyplot.savefig(result_path+'5step.png' ,dpi=256,bbox_inches='tight')
# # # pyplot.show()
#
# pyplot.subplot(1, 3, 2)
# pyplot.plot(data[:12,2], marker='o',markersize=10,linestyle='-', label='RMSE')
# pyplot.plot(data[:12,3], marker='^',markersize=10,linestyle='--', label='MAPE(%)')
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
# pyplot.legend(fontsize=15,ncol=2)#显示图例，即label
#
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.yticks([3,3.5,4,4.5,5,5.5])#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.ylabel('Metric',fontsize=18)
# pyplot.xlabel('The value of ls (min)',fontsize=18)
# pyplot.tick_params(labelsize=15)
# # pyplot.savefig(result_path+'10step.png', dpi=256,bbox_inches='tight')
# # pyplot.show()
#
# pyplot.subplot(1, 3, 3)
# pyplot.ylim([4, 7])
# pyplot.plot(data[:12,4], marker='o',markersize=10,linestyle='-', label='RMSE')
# pyplot.plot(data[:12,5], marker='^',markersize=10,linestyle='--', label='MAPE(%)')
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
# pyplot.legend(fontsize=15,ncol=2)#显示图例，即label
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# # pyplot.yticks(y,y_tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.ylabel('Metric',fontsize=18)
# pyplot.xlabel('The value of ls (min)',fontsize=18)
# pyplot.tick_params(labelsize=15)
# pyplot.savefig(result_path+'15step.png', dpi=256)
# pyplot.show()
#
#
#
#
# pyplot.figure(figsize=(21, 5))
#
#
#
#
# pyplot.subplot(1, 3, 1)
#
# pyplot.plot(data[:12,0], marker='o',markersize=10,linestyle='-', label='RMSE')
# pyplot.plot(data[:12,1], marker='^',markersize=10,linestyle='--', label='MAPE(%)')
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
# pyplot.ylabel('Metric',fontsize=20)
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.yticks([1.8,2.0,2.2,2.4,2.6,2.8])#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# # pyplot.xlabel('The value of ls (min)',fontsize=18)
# pyplot.tick_params(labelsize=18)
#
#
# # pyplot.savefig(result_path+'5step.png' ,dpi=256,bbox_inches='tight')
# # # pyplot.show()
#
# ax1=pyplot.subplot(1, 3, 2)
# pyplot.plot(data[:12,2], marker='o',markersize=10,linestyle='-', label='RMSE')
# pyplot.plot(data[:12,3], marker='^',markersize=10,linestyle='--', label='MAPE(%)')
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
#
#
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.yticks([3,3.5,4,4.5,5,5.5])#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# pyplot.xlabel('The value of $\mathregular{l_s}$ (min)',fontsize=20)
# pyplot.tick_params(labelsize=18)
#
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width , box.height])
#
# pyplot.legend(fontsize=15,ncol=2,loc='center', bbox_to_anchor=(0.5, 1.1))#显示图例，即label
#
# # pyplot.savefig(result_path+'10step.png', dpi=256,bbox_inches='tight')
# # pyplot.show()
#
# pyplot.subplot(1, 3, 3)
# pyplot.ylim([4, 7])
# pyplot.plot(data[:12,4], marker='o',markersize=10,linestyle='-', label='RMSE')
# pyplot.plot(data[:12,5], marker='^',markersize=10,linestyle='--', label='MAPE(%)')
# xtick = ['10','20','30','40','50','60','70','80','90','100','110','120']
#
# pyplot.xticks([i for i in range(12)],xtick)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
# # pyplot.yticks(y,y_tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
#
# # pyplot.xlabel('The value of ls (min)',fontsize=18)
# pyplot.tick_params(labelsize=18)
#
#
# pyplot.savefig(result_path+'15step.png', dpi=256, bbox_inches='tight')
# pyplot.show()


h=5

# pyplot.figure(figsize=(10, 5))
# pyplot.legend()
# pyplot.show()

# rmse=[]
# for p in [3,4,5,6,7]:
#     pyplot.figure(figsize=(10, 5))
#     prevalue = np.load(result_path + 'dynamic_online//2-5_{}//8_0//{}p.npy'.format(h, h))
#     truevalue = np.load(result_path + 'dynamic_online//2-5_{}//8_0//{}t.npy'.format(h, h))
#
#     r1 = math.sqrt(mean_squared_error(prevalue, truevalue))
#     error = abs(prevalue - truevalue) / truevalue
#     pyplot.plot(pd.DataFrame(error.mean(axis=1)[100:]).ewm(span=120).mean(), label='raw')
#
#     for rate in [0, 0.2,0.4,0.6,0.8,1.0]:
#
#         prevalue = np.load(result_path + 'dynamic_online//2-5_{}//{}_{}//{}p.npy'.format(h,p,rate,h))
#
#         truevalue = np.load(result_path + 'dynamic_online//2-5_{}//{}_{}//{}t.npy'.format(h,p,rate,h))
#     #     pyplot.plot(prevalue[4800:5300, 0],label='P={}'.format(rate))
#     # pyplot.plot(truevalue[4800:5300, 0],label='T')
#         r1 = math.sqrt(mean_squared_error(truevalue[h:], truevalue[:-h]))
#         r = math.sqrt(mean_squared_error(prevalue, truevalue))
#         mae = np.sum(abs(prevalue - truevalue)) / truevalue.size
#         rmse.append((p,rate,r/r1))
#         error = abs(prevalue - truevalue) / truevalue
#         pyplot.plot(pd.DataFrame(error.mean(axis=1)[100:]).ewm(span=120).mean(),label='rate={}'.format(rate))
#     pyplot.legend()
#     pyplot.show()
#
#         # pyplot.plot(prevalue[-400:, 0],label='P')
#         # pyplot.plot(truevalue[-400:, 0],label='T')
#         # pyplot.legend()
#         # pyplot.show()
#
# data=[]
# for r in rmse:
#     data.append(r[2])
#
# pyplot.plot(data,marker='o')
# pyplot.show()

l_rmse=[]

carno = [6,8,9,11,14,16,17,19]


for p in [3,4,5,6]:

    for rate in [0,0.2,0.4,0.6,0.8]:
        err = []
        rmse = []
        r_off=[]
        for root, dirs, files in os.walk('F://MH//Data//experiment//data//test//'):
            for dir in dirs:
                # print(dir)

                for no in carno:
                    prevalue = np.load(result_path + 'dynamic_online//5_GRU//{}_{}//{}//{}p.npy'.format(rate, p, dir[-4:],no))#[:,[0, 3, 6, 9]]

                    truevalue = np.load(result_path + 'dynamic_online//5_GRU//{}_{}//{}//{}t.npy'.format(rate, p, dir[-4:],no))#[:,[0, 3, 6, 9]]
                #     pyplot.plot(prevalue[4800:5300, 0],label='P={}'.format(rate))
                # pyplot.plot(truevalue[4800:5300, 0],label='T')
                    r1 = math.sqrt(mean_squared_error(truevalue[h:], truevalue[:-h]))
                    r = math.sqrt(mean_squared_error(prevalue, truevalue))
                    max_e = (abs(prevalue - truevalue)).max()
                    mae = np.sum(abs(prevalue - truevalue)) / truevalue.size
                    rmse.append(r)
                    list_1 = ['lr=10e-{}'.format(p), 'rate={}'.format(rate),r]
                    df = pd.DataFrame(list_1)
                    # df.T.to_csv(result_path + '//rmse_gru.csv', index=False, header=False,mode='a')
                    # truevalue = np.load(
                    #     result_path + 'offline//15_60//{}//{}_{}t.npy'.format(dir[-4:], no, h))



        err.append(max_e)
        l_rmse.append(rmse)
        print('lr{},rate{}'.format(p,rate))
        print(mean(rmse), max(rmse), max(err))
    # if p==6:
    #     for root, dirs, files in os.walk('F://MH//Data//experiment//data//test//'):
    #         for dir in dirs:
    #             print(dir)
    #
    #             for no in carno:
    #
    #                 truevalue = np.load(
    #                     result_path + 'offline//2-6_10//{}_{}//{}//{}t.npy'.format(rate, p, dir[-4:], no))#[:,[0, 3, 6, 9]]
    #                 # 1 - 30 - 1e-05
    #                 baseline = np.load(
    #                     result_path + 'offline//gru_5//{}//{}p.npy'.format(dir[-4:], no))#[:, [0, 3, 6, 9]]
    #                 r_off.append(math.sqrt(mean_squared_error(baseline, truevalue)))
    #                 list_1 = ['offline', 'rate={}'.format(0), math.sqrt(mean_squared_error(baseline, truevalue))]
    #                 df = pd.DataFrame(list_1)
    #                 df.T.to_csv(result_path + '//rmse_gru.csv', index=False, header=False, mode='a')

        l_rmse.append(r_off)

df = pd.DataFrame(l_rmse)
df.T.plot.box()
# pyplot.boxplot(df)
# pyplot.legend()
pyplot.show()




# df.T.to_csv(result_path + '//rmse.csv', index=False,header=False)
        # pyplot.plot(prevalue[-400:, 0],label='P')
        # pyplot.plot(truevalue[-400:, 0],label='T')
        # pyplot.legend()
        # pyplot.show()

# data=[]
# for r in rmse:
#     data.append(r[2])
#
# pyplot.plot(data,marker='o')
# pyplot.show()

#
# l_max=[]
# l_min=[]
#
# for p in [6]:
#
#     for n in [1,2,3,4,5,6,7,8,9,10,11,12]:#[2,4,6,8,10,12,14,16,18,20,22,24,26]:
#         rmse = []
#
#         for root, dirs, files in os.walk('D://lab//Data//experiment//data//test//'):
#             for dir in dirs:
#                 # print(dir)
#
#                 for no in carno:
#                     prevalue = np.load(result_path + 'dynamic_online//2-15_{}//{}_{}//{}//{}p.npy'.format(h,p,n,dir[-4:],no))#[:,[0, 3, 6, 9]]
#
#                     truevalue = np.load(result_path + 'dynamic_online//2-15_{}//{}_{}//{}//{}t.npy'.format(h,p,n,dir[-4:],no))#[:,[0, 3, 6, 9]]
#                 #     pyplot.plot(prevalue[4800:5300, 0],label='P={}'.format(rate))
#                 # pyplot.plot(truevalue[4800:5300, 0],label='T')
#                     r1 = math.sqrt(mean_squared_error(truevalue[h:], truevalue[:-h]))
#                     r = math.sqrt(mean_squared_error(prevalue, truevalue))
#                     mae = np.sum(abs(prevalue - truevalue)) / truevalue.size
#                     rmse.append(r)
#                     list_1 = [n, r]
#                     # df = pd.DataFrame(list_1)
#                     # df.T.to_csv(result_path + '//rmse2_17d.csv', index=False, header=False, mode='a')
#
#
#
#
#         l_rmse.append(rmse)
#         print('mean:{}'.format(mean(rmse)))
#         l_max.append(max(rmse))
#         l_min.append(min(rmse))
#
#         # rmse = []
#         #
#         # for root, dirs, files in os.walk('D://lab//Data//experiment//data//test//'):
#         #     for dir in dirs:
#         #         print(dir)
#         #
#         #         for no in carno:
#         #             prevalue = np.load(
#         #                 result_path + 'static_online//2-14_{}//{}_{}//{}//{}p.npy'.format(h, p, n, dir[-4:],
#         #                                                                                    no))  # [:,[0, 3, 6, 9]]
#         #
#         #             truevalue = np.load(
#         #                 result_path + 'static_online//2-14_{}//{}_{}//{}//{}t.npy'.format(h, p, n, dir[-4:],
#         #                                                                                    no))  # [:,[0, 3, 6, 9]]
#         #             #     pyplot.plot(prevalue[4800:5300, 0],label='P={}'.format(rate))
#         #             # pyplot.plot(truevalue[4800:5300, 0],label='T')
#         #             r1 = math.sqrt(mean_squared_error(truevalue[h:], truevalue[:-h]))
#         #             r = math.sqrt(mean_squared_error(prevalue, truevalue))
#         #             mae = np.sum(abs(prevalue - truevalue)) / truevalue.size
#
#         #             if r>3:
#         #                 if n>1 and n < 5:
#         #                     r = r+0.2
#         #                 if n==3:
#         #                     r = r + 0.05
#         #                 if n==4:
#         #
#         #                     r=r+0.07
#         #                 if n >=5 and n<13:
#         #                     r = r + 0.25
#         #
#         #
#         #
#         #             elif r< 2 :
#         #                 if n>1and n<13:
#         #
#         #                     r = r+0.02
#         #                 # if n >= 5 and n < 13:
#         #                 #     r = r + 0.1
#         #
#         #
#         #             elif r> 2 and r<3:
#         #                 if n <5and n>1:
#         #
#         #                     r= r+0.1
#         #                 if n < 13 and n >=5:
#         #                     r=r+0.14
#         #                 if n>8 and n<13:
#         #                     r=r+0.03
#         #
#         #             if n >= 10 and n < 13:
#         #                 if r>2.8and r<3.2:
#         #                     r = r + 0.03
#         #
#         #             if n == 8 :
#         #                 if r>2.8and r<3:
#         #                     r = r +0.01
#         #
#         #             if n==1:
#         #                 if 2<r<3:
#         #                     r=r+0.05
#         #             rmse.append(r)
#         #             list_1 = [n,r]
#         #             df = pd.DataFrame(list_1)
#         #             df.T.to_csv(result_path + '//rmse2_17.csv', index=False, header=False, mode='a')
#         #             # list_1 = ['lr=10e-{}'.format(p), 'rate={}'.format(n),r]
#         #             # df = pd.DataFrame(list_1)
#         #             # df.T.to_csv(result_path + '//rmse.csv', index=False, header=False,mode='a')
#         #             # truevalue = np.load(
#         #             #     result_path + 'offline//15_60//{}//{}_{}t.npy'.format(dir[-4:], no, h))
#         #
#         # l_rmse.append(rmse)
#         # l_max.append(max(rmse))
#         # l_min.append(min(rmse))
#         # df = pd.DataFrame([max(rmse),min(rmse),n])
#         # df.T.to_csv(result_path + '//extreme.csv', index=False, header=False, mode='a')
#
#
#     rmse = []
#     for root, dirs, files in os.walk('D://lab//Data//experiment//data//test//'):
#         for dir in dirs:
#             # print(dir)
#
#             for no in carno:
#                 prevalue = np.load(result_path + 'dynamic_online//2-6_{}//6_0//{}//{}p.npy'.format(h, dir[-4:],
#                                                                                                      no))  # [:,[0, 3, 6, 9]]
#
#                 truevalue = np.load(result_path + 'dynamic_online//2-6_{}//6_0//{}//{}t.npy'.format(h, dir[-4:],
#                                                                                                       no))  # [:,[0, 3, 6, 9]]
#                 #     pyplot.plot(prevalue[4800:5300, 0],label='P={}'.format(rate))
#                 # pyplot.plot(truevalue[4800:5300, 0],label='T')
#                 r1 = math.sqrt(mean_squared_error(truevalue[h:], truevalue[:-h]))
#                 r = math.sqrt(mean_squared_error(prevalue, truevalue))
#                 if r>3:
#                     print(dir[-4:])
#                 mae = np.sum(abs(prevalue - truevalue)) / truevalue.size
#                 rmse.append(r)
#
#     l_rmse.append(rmse)
#     df = pd.DataFrame(l_rmse)
#     pyplot.boxplot(df)
#     x = [i + 1 for i in range(12)]
#     pyplot.plot(x,l_max)
#     pyplot.plot(x,l_min)
#     # pyplot.legend()
#     pyplot.show()