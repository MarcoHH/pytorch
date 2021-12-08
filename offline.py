import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import concat
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import os

car_no = [6, 8, 9, 11, 14, 16, 17, 19]
# car_no =[9]
raw_path = 'F://MH//Data//experiment//data//rawtest//'
train_path = 'F://MH//Data//experiment//data//detect//'

for root, dirs, files in os.walk(raw_path):
    for dir in dirs:
        # #获取文件所属目录
        # print(root)
        # #获取文件路径
        print(os.path.join(root, dir))
        for no in car_no:
            # filepath = os.path.join(root, dir) +'//{}.npy'.format(no)
            filepath = os.path.join(root, dir) + '//{}.csv'.format(no)

            if os.path.exists(filepath):
                # data = np.load(filepath, allow_pickle=True)
                data = pd.read_csv(filepath, encoding='utf-8')
                # pyplot.plot(data[:,0])
                # pyplot.show()

                features = data.dropna(axis=1, how="all")  # 删除 空的列
                values = np.delete(features.values, [0, 15, 16, 17, 18, 19, 20], axis=1)  # 去除空行
                values[:, -2] = values[:, -3] * 0.5 + values[:, -2]  # 合并冷却风机特征
                values = np.delete(values, -3, axis=1)

                data = values.astype(np.float32)
                data = pd.DataFrame(data)

                # data = data.interpolate(method='linear', axis=0, limit=5, inpalce=True)
                # data = data.ewm(span=2).mean()
                values = data.values

                # index = np.argwhere(values[:, 0] > 30)
                # start_index = np.argwhere(index[:, 0] > 400)[0, 0]
                # start = index[start_index, 0]
                # end = 1300
                # if dir == '190120':
                #     start += 200
                # # if dir == '190215':
                # #     end -= 200
                #
                # if dir == '190215' or dir == '190217' or dir == '190224':
                #     end -= 150
                #
                # if dir == '190214':
                #     start += 350
                #
                # if dir == '190205' or dir == '190211':
                #     start += 200
                #
                # if dir == '190221':
                #     start += 150
                # if no == 6:
                #     pyplot.figure(figsize=(15, 6))
                #     pyplot.plot(values[start:end, 0])
                #     #
                #     pyplot.title(dir)
                #     # # pyplot.savefig(raw_path + "{}.png".format(dir))
                #     pyplot.show()

                this_path = train_path + dir + '//'
                try:
                    os.mkdir(this_path)
                except Exception as e:
                    print('文件夹已存在')

                this_file = this_path + '{}.npy'.format(no)
                np.save(this_file, values[:, :])

        # features = data.dropna(axis=1, how="all")
        # values = np.delete(features.values, [0, 15, 16, 17, 18, 19, 20], axis=1)  # 去除空行
        # values[:, -2] = values[:, -3] * 0.5 + values[:, -2]  # 合并特征
        # values = np.delete(values, -3, axis=1)
        #
        #
        # this_path = train_path + root[-4:] + '//'
        # try:
        #     os.mkdir(this_path)
        # except Exception as e:
        #     print('文件夹已存在')
        #
        # this_file = this_path + file[0:-4]
        # np.save(this_file, values)
