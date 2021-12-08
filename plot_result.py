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










# n_forwards = 5
# for root, dirs, files in os.walk('D://lab//Data//experiment//data//test//'):
#     for dir in dirs:
#         print(dir)
#         if dir[-4:]== '0203':
#             for no in carno:
#                 truevalue = np.load(result_path + 'offline//1-30-1e-05//{}//{}_{}t.npy'.format(dir[-4:], no, n_forwards))[:,
#                             ]  # [0,3,6,9]
#                 baseline = np.load(result_path + 'offline//1-30-1e-05//{}//{}_{}p.npy'.format(dir[-4:], no, n_forwards))[:,
#                           ]
#
#
#                 prevalue = np.load(
#                     result_path + 'dynamic_online//2-6_5//4_0.6/{}//{}p.npy'.format(dir[-4:], no))[:,
#                           ]
#                 # static_value = np.load(
#                 #     result_path + 'dynamic_online//1-29-1e-05//{}//{}_{}t.npy'.format(dir[-4:], no, n_forwards))[:,
#                 #               ]
#                 error1 = abs(baseline - truevalue) / truevalue
#                 error2 = abs(prevalue - truevalue) / truevalue
#
#                 err1 = error1.mean(axis=1)
#                 err2 = error2.mean(axis=1)
#
#                 smooth1 = 1 - pd.DataFrame(err1).ewm(span=30).mean()
#                 smooth2 = 1 - pd.DataFrame(err2).ewm(span=60).mean()
#
#                 smooth1 = smooth1.values
#                 smooth2 = smooth2.values
#
#                 pyplot.figure(figsize=(15, 6))
#
#                 for i in range(3):
#                     pyplot.plot(truevalue[:, i+3], marker='.', label='T', color='b')
#                     pyplot.plot(baseline[:, i +1+3], marker='.', label='off', color='green')
#                     pyplot.plot(prevalue[:, i +2+3], '-', label='online', color='r')
#                 pyplot.legend()
#                 pyplot.show()

                # pyplot.figure(figsize=(20, 20))
                #
                # for i in range(4):
                #     pyplot.subplot(5, 1, i + 1)
                #
                #     pyplot.plot(truevalue[:,i * 3], marker='.', label='truevalue', color='b')
                #     pyplot.plot(baseline[:,i * 3], marker='.', label='baseline', color='green')
                #     pyplot.plot(prevalue[:,i * 3], '-', label='online_dynamic', color='r')
                #
                #     pyplot.legend()
                #     pyplot.ylabel('value', fontsize=15)
                #     pyplot.xlabel('Time Step(1 min)', fontsize=15)
                #
                # pyplot.subplot(5, 1, 5)
                #
                # pyplot.ylim([0.6, 1])
                #
                #
                # for i in range(smooth1.shape[1]):
                #     pyplot.plot(smooth1[:, i], linestyle='--', linewidth='2.1', color='green',label='offline')
                #     pyplot.plot(smooth2[:, i], linestyle='--', linewidth='2.1', color='b',label='online')
                # pyplot.legend(fontsize=20)
                # pyplot.tick_params(labelsize=15)
                # pyplot.ylabel('Prediction Accuracy Over Last 30 samples', fontsize=18)
                # pyplot.xlabel('Time (t)', fontsize=18)
                # pyplot.grid(axis="y")
                #
                # # pyplot.savefig(result_path +  "figure//{}-{}.png".format(dir[-4:],no))
                # pyplot.show()

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

#
# truevalue = np.load(result_path + 'dynamic_online//2-18_5//4_0.8//0301//9t.npy')[:,
#                            ]  # [0,3,6,9]
# baseline = np.load(result_path +'offline//2-18_5//4_0//0301//9p.npy')[:,
#                            ]
# prevalue =  np.load(result_path +'dynamic_online//2-18_5//4_0.6//0301//9p.npy')[:,
#                            ]
# error1 = abs(baseline - truevalue) / truevalue
# error2 = abs(prevalue - truevalue) / truevalue
#
# err1 = error1.mean(axis=1)
# err2 = error2.mean(axis=1)
#
# # smooth1 = 1 - pd.DataFrame(err1).ewm(span=300).mean()
# # smooth2 = 1 - pd.DataFrame(err2).ewm(span=500).mean()
#
# smooth1 = 1 - pd.DataFrame(err1).rolling(window=300).mean()
# smooth2 = 1 - pd.DataFrame(err2).rolling(window=300).mean()
#
#
# smooth3 = 1 - pd.DataFrame(err1).rolling(window=1000).mean()
# smooth4 = 1 - pd.DataFrame(err2).rolling(window=1000).mean()
#
# smooth1 = smooth1.values
# smooth2 = smooth2.values
# smooth3 = smooth3.values
# smooth4 = smooth4.values
#
# # pyplot.figure(figsize=(15, 6))
# #
# # for i in range(3):
# #     pyplot.plot(truevalue[4000:4500, i+3], '-', label='T', color='b')
# #     pyplot.plot(baseline[4000:4500, i +1 +3], '-', label='off', color='green')
# #     pyplot.plot(prevalue[4000:4500, i +2 +3], '-', label='online', color='r')
# # pyplot.legend()
# # pyplot.show()
#
# # pyplot.ylim([0.6, 1])
#
# pyplot.figure(figsize=(14, 6))
# for i in range(smooth1.shape[1]):
#     pyplot.plot(smooth1[1000:7000, i], linestyle='-', linewidth='2', color='green',label='MA300(offline)')
#     pyplot.plot(smooth2[1000:7000, i], linestyle='-', linewidth='2', color='b',label='MA300(online)')
#     pyplot.plot(smooth3[1000:7000, i], linestyle='--', linewidth='1', color='r', label='MA1000(offline)')
#     pyplot.plot(smooth4[1000:7000, i], linestyle='--', linewidth='1', color='orange', label='MA1000(online)')
# # pyplot.legend(fontsize=20)
# pyplot.grid(linestyle=":",linewidth='1.2',)
# pyplot.ylabel('Prediction Accuracy ', fontsize=20)
# pyplot.tick_params(labelsize=18)
# pyplot.xlabel('Time (t)', fontsize=20)
# pyplot.legend(fontsize=18)
#
# pyplot.axvspan(3700, 4500, facecolor='#45FFAB', alpha=0.3)
# pyplot.savefig(result_path + "Accuracy.png", dpi=400,
#                                bbox_inches='tight')
# pyplot.show()
#


'''

n=1
color = ['b','green','orange','purple']
marker =['.','x','>','<']
pyplot.figure(figsize=(10, 8))
n_forwards = 5
for root, dirs, files in os.walk('F://MH//Data//experiment//data//test//'):
    for dir in dirs:
        print(dir)
        # if dir[-4:]== '0203':

        for no in carno:
            if  no == 6 and dir[-4:]== '0226' :
                truevalue = np.load(
                    result_path + 'dynamic_online//5_GRU//0.8_4//{}//{}t.npy'.format(dir[-4:], no))[:,
                            ]  # [0,3,6,9]
                baseline = np.load(
                    result_path + 'offline//gru_5//{}//{}p.npy'.format(dir[-4:], no))[:,
                           ]

                prevalue = np.load(
                    result_path + 'dynamic_online//5_GRU//0.8_4//{}//{}p.npy'.format(dir[-4:], no))[:,
                           ]
                # df = pd.DataFrame(prevalue)
                # df.to_csv(result_path + '//plot.csv', index=False, header=False, mode='a')
                # prevalue = pd.read_csv(
                #     result_path + '//plot.csv', header=None).values
                #
                # marke_pos = [i * 4 for i in range(int(baseline.shape[0] / 4))]
                # pyplot.subplot(4, 1, n)
                #
                # pyplot.plot(truevalue[:, 0],  markevery=marke_pos, label='{}'.format(dir[-6:]), color='b')
                # pyplot.plot(baseline[:, 0], '--',label='truevalue', color='r')
                # pyplot.plot(prevalue[:, 0],marker=marker[0],markevery=marke_pos, label='online', color='grey')
                # # pyplot.title(r"$\bf{" + 'Date:{}'.format(dir[-4:]) + "}$",fontsize=12)
                # pyplot.legend(loc=1)
                # pyplot.ylabel('temperature (℃)', fontsize=12)
                # pyplot.tick_params(labelsize=12)
                # if n == 4:
                #
                #
                #     # pyplot.subplots_adjust(top=0.96, bottom=0.065, right=1, left=0.065, hspace=0.3, wspace=0.4)
                #     # pyplot.margins(0.2, 0.2)
                #
                #     pyplot.xlabel('Time (min)', fontsize=12)
                #
                #     pyplot.savefig(result_path + "figure//date//{}-{}.png".format(dir[-4:], no),dpi=512,bbox_inches='tight')
                #     pyplot.show()
                #     pyplot.figure(figsize=(10, 8))
                #     n = 0
                # n += 1


                pyplot.figure(figsize=(10, 8))

                marke_pos = [i * 4 for i in range(int(baseline.shape[0] / 4))]
                for i in range(4):
                    pyplot.subplot(4, 1, i + 1)
                    # pyplot.title(r"$\bf{" + 'No.{}'.format(i+1) + "}$",fontsize=12)

                    # pyplot.plot(truevalue[:,(i * 3)+1],marker='.' ,markevery=marke_pos, label='baseline', color='green')
                    # pyplot.plot(truevalue[:,(i * 3)+2],marker='.' ,markevery=marke_pos, label='online_dynamic', color='r')



                    # pyplot.plot(baseline[:, (i * 3) + 1], label='baseline', color='orange')
                    # pyplot.plot(baseline[:, (i * 3) + 2], label='online_dynamic', color='orange')
                    # pyplot.plot(prevalue[:, i * 3], marker='.',markevery=marke_pos,label='online', color='grey')
                    pyplot.plot(truevalue[:, i * 3], color=color[i], marker='.' ,markevery=marke_pos, label='{}-{}-{}'.format(dir[-4:], no, 1+i))
                    # pyplot.plot(baseline[:, i * 3], '--', label='offline', color='r')
                    # pd.DataFrame(prevalue[:, i * 3]).ewm(span=3).mean()
                    pyplot.tick_params(labelsize=12)
                    pyplot.ylabel('temperature (℃)', fontsize=12)
                    # if i==0:
                    pyplot.legend(loc=1)
                    #     pyplot.legend(bbox_to_anchor=(0.5, 1.15),ncol=3, loc=10,fontsize=15)
                pyplot.xlabel('Time (1 min)', fontsize=12)
                pyplot.savefig(result_path + "figure//5//pre.png".format(dir[-4:], no), dpi=512,
                               bbox_inches='tight')
                pyplot.close()
                # pyplot.show()



#
#                 # static_value = np.load(
#                 #     result_path + 'dynamic_online//1-29-1e-05//{}//{}_{}t.npy'.format(dir[-4:], no, n_forwards))[:,
#                 #               ]
#                 #
#                 # pyplot.figure(figsize=(15, 6))
#                 #
#                 # for i in range(3):
#                 #     pyplot.plot(truevalue[:, i*3], marker='.', color='b')
#                 #     pyplot.plot(truevalue[:, (i*3)+1], marker='.', color='green')
#                 #     pyplot.plot(truevalue[:, (i*3)+2], '-', color='r')
#                 #
#                 # pyplot.plot(truevalue[:, 3 * 3], marker='.', label='T', color='b')
#                 # pyplot.plot(truevalue[:, (3 * 3) + 1], marker='.', label='off', color='green')
#                 # pyplot.plot(truevalue[:, (3 * 3) + 2], '-', label='online', color='r')
#                 # # pyplot.plot(truevalue, marker='.', label='T', color='b')
#                 # pyplot.legend()
#                 # pyplot.title(dir[-4:])
#                 # pyplot.show()
#
# n = 1
# color = ['b', 'green', 'orange', 'purple']
# marker = ['.', 'x', '>', '<']
# pyplot.figure(figsize=(10, 8))
# n_forwards = 5
# for date, carno, no in [('0129',11,1),('0203',8,1),('0207',16,3),('0210',17,1)]:
#
#         truevalue = np.load(
#             result_path + 'offline//base_5//{}//{}t.npy'.format(date, carno))[:,
#                     ]  # [0,3,6,9]
#         baseline = np.load(
#             result_path + 'offline//base_5//{}//{}p.npy'.format(date, carno))[:,
#                    ]
#
#         prevalue = np.load(
#             result_path + 'dynamic_online//2-6_5//4_0.6/{}//{}p.npy'.format(date, carno))[:,
#                    ]
#
#
#
#         marke_pos = [i * 4 for i in range(int(baseline.shape[0] / 4))]
#
#         pyplot.subplot(4, 1, n)
#         # pyplot.title(r"$\bf{" + 'No.{}'.format(i+1) + "}$",fontsize=12)
#
#         # pyplot.plot(truevalue[:,(i * 3)+1],marker='.' ,markevery=marke_pos, label='baseline', color='green')
#         # pyplot.plot(truevalue[:,(i * 3)+2],marker='.' ,markevery=marke_pos, label='online_dynamic', color='r')
#
#         # pyplot.plot(baseline[:, (i * 3) + 1], label='baseline', color='orange')
#         # pyplot.plot(baseline[:, (i * 3) + 2], label='online_dynamic', color='orange')
#         i = no-1
#         pyplot.plot(prevalue[:,  i * 3], marker='.', markevery=marke_pos, label='online', color='grey')
#         pyplot.plot(truevalue[:, i * 3], markevery=marke_pos, marker=marker[n-1],color='b', label='{}-{}-{}'.format(date, carno, 1))
#         pyplot.plot(baseline[:, i * 3], '--', label='offline', color='r')
#         # pd.DataFrame(prevalue[:, i * 3]).ewm(span=3).mean()
#         pyplot.tick_params(labelsize=12)
#         pyplot.ylabel('temperature (℃)', fontsize=12)
#         # if n == 1:
#         #     pyplot.legend(bbox_to_anchor=(0.5, 1.15), ncol=3, loc=10, fontsize=15)
#         pyplot.legend(loc=4)
#         n+=1
# pyplot.xlabel('Time (1 min)', fontsize=12)
# pyplot.savefig(result_path + "figure//date//result.png", dpi=512,
#                bbox_inches='tight' )
# pyplot.close()
# # pyplot.show()
'''

n = 1
color = ['b', 'green', 'orange', 'purple']
marker = ['.', 'x', '>', '<']
pyplot.figure(figsize=(10, 8))
n_forwards = 5
for date, carno, no in [('0129',11,1),('0203',8,1),('0207',16,3),('0210',17,1)]:

        truevalue = np.load(
            result_path + 'offline//base_5//{}//{}t.npy'.format(date, carno))[:,
                    ]  # [0,3,6,9]
        baseline = np.load(
            result_path + 'offline//base_5//{}//{}p.npy'.format(date, carno))[:,
                   ]

        prevalue = np.load(
            result_path + 'dynamic_online//2-6_5//4_0.6/{}//{}p.npy'.format(date, carno))[:,
                   ]
        elm = np.load(
            result_path + 'dynamic_online//oselm//{}//{}p.npy'.format(date, carno))[:,
              ]



        marke_pos = [i * 5 for i in range(int(baseline.shape[0] / 5))]

        pyplot.subplot(4, 1, n)
        # pyplot.title(r"$\bf{" + 'No.{}'.format(i+1) + "}$",fontsize=12)

        # pyplot.plot(truevalue[:,(i * 3)+1],marker='.' ,markevery=marke_pos, label='baseline', color='green')
        # pyplot.plot(truevalue[:,(i * 3)+2],marker='.' ,markevery=marke_pos, label='online_dynamic', color='r')

        # pyplot.plot(baseline[:, (i * 3) + 1], label='baseline', color='orange')
        # pyplot.plot(baseline[:, (i * 3) + 2], label='online_dynamic', color='orange')
        i = no-1
        pyplot.plot(truevalue[:, i * 3], markevery=marke_pos, label='True Value')
        # pyplot.plot(baseline[:, i * 3], '--', label='MLP', color='r')
        pyplot.plot(elm[:, i * 3], '-.', label='OS-ELM', color='green')
        pyplot.plot(prevalue[:,  i * 3], marker='.', markevery=marke_pos, label='MLP with Online Learning', color='grey')

        # pyplot.plot(truevalue[:, i * 3], markevery=marke_pos, label='True Value')
        # pyplot.plot(baseline[:, i * 3], '--', label='MLP', color='r')
        # pyplot.plot(elm[:, i * 3], '.-', markevery=marke_pos, label='OS-ELM',
        #             color='green')
        # pyplot.plot(prevalue[:, i * 3], marker='.', markevery=marke_pos, label='MLP with Online Learning',
        #             color='grey')


        # pd.DataFrame(prevalue[:, i * 3]).ewm(span=3).mean()
        pyplot.tick_params(labelsize=12)
        pyplot.ylabel('temperature (℃)', fontsize=12)
        if n == 1:
            pyplot.legend(bbox_to_anchor=(0.5, 1.15), ncol=3, loc=10, fontsize=15)
        # pyplot.legend(loc=4)
        n+=1
pyplot.xlabel('Time (1 min)', fontsize=12)
# pyplot.savefig(result_path + "figure//date//result1.png", dpi=512,
#                bbox_inches='tight' )
# pyplot.close()
pyplot.show()


n = 1
color = ['b', 'green', 'orange', 'purple']
marker = ['.', 'x', '>', '<']
pyplot.figure(figsize=(10, 8))
n_forwards = 5
for date, carno, no in [('0129',11,1),('0203',8,1),('0207',16,3),('0210',17,1)]:

        truevalue = np.load(
            result_path + 'offline//base_5//{}//{}t.npy'.format(date, carno))[:,
                    ]  # [0,3,6,9]
        baseline = np.load(
            result_path + 'offline//base_5//{}//{}p.npy'.format(date, carno))[:,
                   ]

        prevalue = np.load(
            result_path + 'dynamic_online//2-6_5//4_0.6/{}//{}p.npy'.format(date, carno))[:,
                   ]
        elm = np.load(
            result_path + 'dynamic_online//oselm//{}//{}p.npy'.format(date, carno))[:,
              ]



        marke_pos = [i * 4 for i in range(int(baseline.shape[0] / 4))]

        pyplot.subplot(4, 1, n)
        # pyplot.title(r"$\bf{" + 'No.{}'.format(i+1) + "}$",fontsize=12)

        # pyplot.plot(truevalue[:,(i * 3)+1],marker='.' ,markevery=marke_pos, label='baseline', color='green')
        # pyplot.plot(truevalue[:,(i * 3)+2],marker='.' ,markevery=marke_pos, label='online_dynamic', color='r')

        # pyplot.plot(baseline[:, (i * 3) + 1], label='baseline', color='orange')
        # pyplot.plot(baseline[:, (i * 3) + 2], label='online_dynamic', color='orange')
        i = no-1
        pyplot.plot(truevalue[:, i * 3], marker = marker[n-1], markevery=marke_pos, label='{}-{}-{}'.format(date,carno,1),color='b')
        # pyplot.plot(baseline[:, i * 3], '--', label='MLP', color='r')
        # pyplot.plot(prevalue[:,  i * 3], marker='.', markevery=marke_pos, label='MLP with Online Learning', color='grey')

        # pd.DataFrame(prevalue[:, i * 3]).ewm(span=3).mean()
        pyplot.tick_params(labelsize=12)
        pyplot.ylabel('temperature (℃)', fontsize=12)
        # if n == 1:
        #     pyplot.legend(bbox_to_anchor=(0.5, 1.15), ncol=3, loc=10, fontsize=15)
        pyplot.legend(loc=4)
        n+=1
pyplot.xlabel('Time (1 min)', fontsize=12)
# pyplot.savefig(result_path + "figure//date//result3.png", dpi=512,
#                bbox_inches='tight' )
pyplot.show()
pyplot.close()




carno = [6,8,9,11,14,16,17,19]
n=1
color = ['b','green','orange','purple']
marker =['.','x','>','<']
pyplot.figure(figsize=(10, 8))
n_forwards = 5
for root, dirs, files in os.walk('F://MH//Data//experiment//data//test//'):
    for dir in dirs:
        print(dir)
        # if dir[-4:]== '0203':

        for no in carno:
            if  no == 6 and dir[-4:]== '0226' :
                truevalue = np.load(
                    result_path + 'dynamic_online//5_GRU//0.6_4//{}//{}t.npy'.format(dir[-4:], no))[:,
                            ]  # [0,3,6,9]
                baseline = np.load(
                    result_path + 'offline//base_5//{}//{}p.npy'.format(dir[-4:], no))[:,
                           ]

                prevalue = np.load(
                    result_path + 'dynamic_online//2-6_5//4_0/{}//{}p.npy'.format(dir[-4:], no))[:,
                           ]

                prevalue = np.load(
                    result_path + 'dynamic_online//4-12//{}//{}p.npy'.format(dir[-4:], no))[:,
                           ]
                # df = pd.DataFrame(prevalue)
                # df.to_csv(result_path + '//plot.csv', index=False, header=False, mode='a')
                prevalue = pd.read_csv(
                    result_path + '//plot.csv', header=None).values

                elm = np.load(
                    result_path + 'dynamic_online//oselm//{}//{}p.npy'.format(dir[-4:], no))[:,
                           ]

                # marke_pos = [i * 4 for i in range(int(baseline.shape[0] / 4))]
                # pyplot.subplot(4, 1, n)
                #
                # pyplot.plot(truevalue[:, 0],  markevery=marke_pos, label='{}'.format(dir[-6:]), color='b')
                # pyplot.plot(baseline[:, 0], '--',label='truevalue', color='r')
                # pyplot.plot(prevalue[:, 0],marker=marker[0],markevery=marke_pos, label='online', color='grey')
                # # pyplot.title(r"$\bf{" + 'Date:{}'.format(dir[-4:]) + "}$",fontsize=12)
                # pyplot.legend(loc=1)
                # pyplot.ylabel('temperature (℃)', fontsize=12)
                # pyplot.tick_params(labelsize=12)
                # if n == 4:
                #
                #
                #     # pyplot.subplots_adjust(top=0.96, bottom=0.065, right=1, left=0.065, hspace=0.3, wspace=0.4)
                #     # pyplot.margins(0.2, 0.2)
                #
                #     pyplot.xlabel('Time (min)', fontsize=12)
                #
                #     pyplot.savefig(result_path + "figure//date//{}-{}.png".format(dir[-4:], no),dpi=512,bbox_inches='tight')
                #     pyplot.show()
                #     pyplot.figure(figsize=(10, 8))
                #     n = 0
                # n += 1


                pyplot.figure(figsize=(10, 8))

                marke_pos = [i * 4 for i in range(int(baseline.shape[0] / 4))]
                for i in range(4):
                    pyplot.subplot(4, 1, i + 1)
                    # pyplot.title(r"$\bf{" + 'No.{}'.format(i+1) + "}$",fontsize=12)

                    # pyplot.plot(truevalue[:,(i * 3)+1],marker='.' ,markevery=marke_pos, label='baseline', color='green')
                    # pyplot.plot(truevalue[:,(i * 3)+2],marker='.' ,markevery=marke_pos, label='online_dynamic', color='r')



                    # pyplot.plot(baseline[:, (i * 3) + 1], label='baseline', color='orange')
                    # pyplot.plot(baseline[:, (i * 3) + 2], label='online_dynamic', color='orange')

                    pyplot.plot(truevalue[:, i * 3],markevery=marke_pos, label='True Value')
                    # pyplot.plot(baseline[:, i * 3], '--', label='MLP', color='r')
                    pyplot.plot(elm[:, i * 3], '-.', markevery=marke_pos, label='OS-ELM',
                                color='green')
                    pyplot.plot(prevalue[:, i * 3], marker='.', markevery=marke_pos, label='MLP with Online Learning',
                                color='grey')


                    # pd.DataFrame(prevalue[:, i * 3]).ewm(span=3).mean()
                    pyplot.tick_params(labelsize=12)
                    pyplot.ylabel('temperature (℃)', fontsize=12)
                    if i==0:
                    # pyplot.legend(loc=1)
                        pyplot.legend(bbox_to_anchor=(0.5, 1.15),ncol=3, loc=10,fontsize=15)
                pyplot.xlabel('Time (1 min)', fontsize=12)
                pyplot.show()
                # pyplot.savefig(result_path + "figure//date//result2.png".format(dir[-4:], no), dpi=512,
                #                bbox_inches='tight')
                pyplot.close()
                # pyplot.show()



score = pd.read_csv(result_path + '//dynamic_online//oselm_10//score.csv')
a = score["RMSE"].groupby(["nums","n_batch"])

a.mean()