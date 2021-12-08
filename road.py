import pandas as pd
from pandas import concat
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import os
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import networkx as nx

import Moran
from networkx.algorithms import community
import xlrd
import pyproj

from  pyproj  import  CRS

colors = ['#fe5858', '#034b61', '#5d353e', '#b2d6ca','#b725ca','#a525ca','#b2d6ca','#b725ca','#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca','#b725ca','#a525ca','#b2d6ca','#b725ca','#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca'
          ,'#fe5858', '#034b61', '#5d353e', '#b2d6ca','#b725ca','#a525ca','#b2d6ca','#b725ca','#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca','#b725ca','#a525ca','#b2d6ca','#b725ca','#a525ca']
options = {'font_family': 'serif', 'font_weight': 'semibold', 'font_size': '12', 'font_color': '#ffffff'}
savefig_path = 'F:/Python/NetworkSci/'


def com_postion(size, scale=1, center=(0, 0), dim=2):
    # generat the postion for each nodes in a community
    num = size
    center = np.asarray(center)
    theta = np.linspace(0, 1, num + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack([np.cos(theta), np.sin(theta), np.zeros((num, 0))])
    pos = scale * pos + center
    return pos


def node_postion(one_com, scale=1, center=(0, 0), dim=2):
    # generat the postion for each nodes in a community
    num = len(one_com)
    node = list(one_com)
    center = np.asarray(center)
    theta = np.linspace(0, 1, num + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack([np.cos(theta), np.sin(theta), np.zeros((num, 0))])
    pos = scale * pos + center
    pos = dict(zip(node, pos))
    return pos







speed =[]







out_path='C://Users//MH//Desktop//anomoly//'
figure_path='C://Users//MH//Desktop//figure//'
#
# for no in range(250):
#     speed = []
#     days= []
#     for root, dirs, files in os.walk('C://Users//MH//Desktop//data//'):  #数据文件夹地址
#         for file in files:
#
#             # print(os.path.join(root, file))
#             # if file =='2019-12-07.csv':
#
#             filepath = os.path.join(root, file)
#
#
#
#
#             data = pd.read_csv(filepath, encoding='gb2312')
#
#             data.groupby('路段名称')
#             list_value = []
#             list_name = []
#             for name, group in data.groupby('路段名称'):
#                 list_name.append(name)
#                 list_value.append(group)
#
#
#             road = list_value[no].values
#             value = pd.DataFrame(road[:, -2]).values
#
#
#             # value = value.ewm(span=30).mean().values
#             speed.append(value)
#             days.append(file[:-4])
#     all_days = pd.DataFrame(np.concatenate(speed, axis=1), columns=days)
#
#     all_days.to_csv(out_path+'{}.csv'.format(list_name[no]),mode='a', line_terminator='\n')
#     print(list_name[no])





# #
# speed = []
# days= []
# for root, dirs, files in os.walk('C://Users//MH//Desktop//data//'):  #数据文件夹地址
#     for file in files:
#
#         # print(os.path.join(root, file))
#        if file =='2019-12-09.csv':
#
#             filepath = os.path.join(root, file)
#
#             # for no in range(250):
#
#             data = pd.read_csv(filepath,encoding='gb2312')
#             print(filepath)
#             data.groupby('路段名称')
#             list_value = []
#             list_name = []
#             for name, group in data.groupby('路段名称'):
#
#                 road = group
#                 subpath = figure_path + '{}//'.format(file[:-4])
#                 if not os.path.exists(subpath):
#                     os.makedirs(subpath)
#                 plt.plot(road['平均速度'].values)
#                 plt.savefig(subpath + '{}raw.png'.format(name),dpi=200)
#                 plt.close()
#                 # plt.show()
#                 value = road[['平均速度', '拥堵指数']]
#                 forward = value.diff().fillna(0)
#                 back = value.diff(periods=-1).fillna(0)
#                 plt.plot(forward.values[:,0],color='orange')
#                 plt.savefig(subpath + '{}f.png'.format(name),dpi=200)
#                 plt.close()
#                 plt.plot(back.values[:,0],color='orange')
#                 plt.savefig(subpath + '{}b.png'.format(name),dpi=200)
#                 plt.close()
#                 # plt.show()
#                 mean_f = forward.describe().loc['mean']
#                 std_f = forward.describe().loc['std']
#                 index1 = forward[forward['平均速度'] > mean_f['平均速度'] + 1.5 * std_f['平均速度']].index
#                 road.loc[index1, ['平均速度', '拥堵指数']] = nan
#                 mean_b = back.describe().loc['mean']
#                 std_b = back.describe().loc['std']
#                 index2 = back[back['平均速度'] > mean_b['平均速度'] + 1.5 * std_b['平均速度']].index
#
#                 road.loc[index2, ['平均速度', '拥堵指数']] = nan
#                 road.fillna(method='ffill',inplace=True)
#                 road.fillna(method='bfill',inplace=True)
#                 plt.plot(road['平均速度'].values,color='b')
#
#
#                 plt.savefig(subpath + '{}.png'.format(name),dpi=200)
#                 plt.close()

            # road.to_csv(out_path + '{}.csv'.format(file[:-4]),  index=False,
            #             mode='a', header=None, line_terminator='\n',encoding='utf-8')

            # list_name.append(name)
                # list_value.append(group)



            # road = list_value[no]


            # value = pd.DataFrame(road)


                # value = value.ewm(span=30).mean().values
                # speed.append(value)
                # days.append(file[:-4])

    # all_days = pd.DataFrame(np.concatenate(speed, axis=1), columns=days)
    #
    # all_days.to_csv(out_path+'{}.csv'.format(list_name[no]),mode='a', line_terminator='\n')
    # print(list_name[no])






#
# data = pd.read_csv('C://Users//MH//Desktop//sort_by_road//安慧桥-健翔桥.csv', usecols=[3,4,5,6,7,
#                                       10,11,12,13,14,
#                                       17,18,19,20,21,
#                                       24,25,27,28,
#                                       31], encoding='gb2312')
# # df.drop('column_name', axis=1, inplace=True)
#
#
#
# start = '1/1/2020'
# x = pd.date_range(start, periods=288, freq='5T')
# data.index = x
# speed_hour = data.resample('H').mean()
#
# pyplot.plot(speed_hour.values)
# pyplot.show()



attr = pd.read_csv('C://Users//MH//Desktop//anomoly//2019-12-08.csv', header=None,
                   names = ['路段名称','路段长度','拥堵指数','平均速度'] ,
                   encoding='utf-8',index_col=0)
timeindex = np.unique(attr.index.values).tolist()
attr_s = attr.loc[timeindex[96]]
# data_fecha.loc[fecha_1: fecha_2]
dict_attr={}
for i in (range(attr_s.shape[0])):
    dict_attr[attr_s['路段名称'][i]] = [attr_s['路段长度'][i]
        ,attr_s['拥堵指数'][i], attr_s['平均速度'][i]]



data = pd.read_csv('C://Users//MH//Desktop//anomoly//2019-12-08.csv', header=None,
                   names = ['路段名称','路段长度','拥堵指数','平均速度'] ,
                   encoding='utf-8',index_col=0)


data.groupby('路段名称')
list_value = []
list_name = []
for name, group in data.groupby('路段名称'):
    list_name.append(name)
    list_value.append(group)




unique_list = []   # 重复路段名中取一个
for name in list_name:  # name:路段名    b_name:反向路段名

    start, end = name.split('-', 1)[0], name.split('-', 1)[1]
    b_name ='{}-{}'.format(end, start)
    if not name in unique_list  :
        if not b_name in unique_list :
            unique_list.append(name)


import pyproj

from  pyproj  import  CRS

# -------------坐标转换
crs=CRS.from_epsg(4326)
crs_cs=CRS.from_epsg(3857)

transformer = pyproj.Transformer.from_crs(crs,crs_cs)
data = pd.read_excel("C://Users//MH//Desktop//node.xlsx")
lon = data.lon.values
lat = data.lat.values
x2, y2 = transformer.transform(lat,lon)

pos ={}
i = 0
for node in unique_list:
    pos[node] = np.array([x2[i],y2[i]])
    i += 1

#----------空间位置

graph = {}  # 邻接表  字典形式
OD = []
v=[]
for node in unique_list:
    O_start, O_end = node.split('-', 1)[0], node.split('-', 1)[1]   # O路段的起始
    node_value = {}  #储存一个O的所有D
    for name in unique_list:  # 遍历所有可能的D
        if name != node:
            loc = name.split('-', 1)

            D_start, D_end = loc[0], loc[1] # D路段的起始


            if  O_end in loc:    # 如果O路段的起始在D中，则确定一对OD


            # 为每一个路段节点 赋予流出的权重
                if O_end == D_start:   # 按照O的正常顺序 赋予流出的权重
                    key = name
                if O_end == D_end:
                    key = '{}-{}'.format(D_end, D_start)


                value = (dict_attr[key][-3]+dict_attr[node][-3])/2   #两个路段距离的一半

                value1 = (dict_attr[key][-3] + dict_attr[node][-3]) / (dict_attr[key][-1] + dict_attr[node][-1])
                v.append(value1)
                weight={'weight':value,'time':value1}
                node_value[name]=weight
                # OD.append([node,name,1])
            if  O_start in loc:      # 按照O的反向顺序 赋予流出的权重

                if O_start == D_start:
                    key = name
                if O_start == D_end:
                    key = '{}-{}'.format(D_end, D_start)

                inv_node = '{}-{}'.format(O_end, O_start)
                value = (dict_attr[key][-3] + dict_attr[inv_node][-3]) / 2
                value1 = 50*(dict_attr[key][-3] + dict_attr[inv_node][-3])/(dict_attr[key][-1] + dict_attr[inv_node][-1])
                v.append(value1)
                weight={'weight':value, 'time':value1}
                node_value[name] = weight

                # OD.append([node, name, 1])

    graph[node] = node_value

vmax=max(v)
vmin=min(v)


graph = {}  # 邻接表  字典形式
OD = []
for node in unique_list:
    O_start, O_end = node.split('-', 1)[0], node.split('-', 1)[1]   # O路段的起始
    node_value = {}  #储存一个O的所有D
    for name in unique_list:  # 遍历所有可能的D
        if name != node:
            loc = name.split('-', 1)

            D_start, D_end = loc[0], loc[1] # D路段的起始


            if  O_end in loc:    # 如果O路段的起始在D中，则确定一对OD


            # 为每一个路段节点 赋予流出的权重
                if O_end == D_start:   # 按照O的正常顺序 赋予流出的权重
                    key = name
                if O_end == D_end:
                    key = '{}-{}'.format(D_end, D_start)


                value = (dict_attr[key][-3]+dict_attr[node][-3])/2   #两个路段距离的一半

                value1 = 50*(dict_attr[key][-3] + dict_attr[node][-3]) / (dict_attr[key][-1] + dict_attr[node][-1])
                weight={'weight':value,'time':pow((value1-vmin)/(vmax-vmin),2)*10}
                node_value[name]=weight
                # OD.append([node,name,1])
            if  O_start in loc:      # 按照O的反向顺序 赋予流出的权重

                if O_start == D_start:
                    key = name
                if O_start == D_end:
                    key = '{}-{}'.format(D_end, D_start)

                inv_node = '{}-{}'.format(O_end, O_start)
                value = (dict_attr[key][-3] + dict_attr[inv_node][-3]) / 2
                value1 = 50*(dict_attr[key][-3] + dict_attr[inv_node][-3])/(dict_attr[key][-1] + dict_attr[inv_node][-1])
                weight={'weight':value, 'time':pow((value1-vmin)/(vmax-vmin),2)*10}
                node_value[name] = weight

                # OD.append([node, name, 1])

    graph[node] = node_value
# {1: {"weight": 1}}
f = pd.DataFrame(OD)

G = nx.DiGraph(graph) #有向图
# G = nx.Graph(graph)

nx.draw_networkx(G, pos, node_size=80,font_size=5)
  #图结构可视化
plt.show()

A = nx.to_numpy_array(G,weight='time')  #图结构权重输出到数组A

i=0
for u in G.nodes():  # 遍历图的每个点
    path = nx.shortest_path_length(G,source=u,weight='time')  # 在网络G中计算从u开始到其他所有节点（注意包含自身）的最短路径长度。
    j = 0
    for v in unique_list:  # path是一个字典，里面存了所有目的地节点到u的最短路径长度

        dis = path[v]
        print(dis)#距离为最短路径
        if u == v:  # 如果起止点相同 距离=1
            dis = 1

        A[i,j] = dis*dis
            # 1/dis**2  # 如果起止点相同 距离=反距离的平方
        j += 1
    i+=1


# pd.read_csv =='2019-12-07.csv':
#
#             filepath = os.path.join(root, file)
#
#
#
#
#             data = pd.read_csv(filepath, encoding='gb2312')

# df = pd.DataFrame(A,index=unique_list,columns=unique_list)
matrix =np.zeros((9,9))
# df = pd.DataFrame(A, columns=unique_list)
# df.to_csv("C://Users//MH//Desktop//matrix.csv", encoding="UTF-8",line_terminator='\n')
# df=pd.DataFrame(unique_list)
# df.to_csv("C://Users//MH//Desktop//name.csv", encoding="UTF-8",line_terminator='\n')



I_moran=[]

n=1
for t in timeindex:

    attr_s = attr.loc[t]

    # data_fecha.loc[fecha_1: fecha_2]
    dict_attr={}
    for i in (range(attr_s.shape[0])):
        dict_attr[attr_s['路段名称'][i]] = [attr_s['路段长度'][i]
            ,attr_s['拥堵指数'][i], attr_s['平均速度'][i]]


    # A = nx.to_numpy_array(G)
    Z = np.random.rand(len(G.nodes))  # 观测属性
    i=0
    for name in unique_list:
        start, end = name.split('-', 1)[0], name.split('-', 1)[1]
        b_name = '{}-{}'.format(end, start)
        Z[i] = (dict_attr[name][-1]+dict_attr[b_name][-1])/2
        i+=1
    result = Moran.moranI(A, Z, imgPath='v.png')
    I_moran.append(result['I']['value'])
    # if n%12 == 0 and n > 60:
    #     Zi = abs(result['ZIi']['value'])
    #
    #     Zi = (Zi - min(Zi)) / (max(Zi) - min(Zi)).tolist()
    #     plt.figure(figsize=(15, 6))
    #     fig1=plt.subplot(1, 2, 1)
    #     nx.draw_networkx(G, pos, node_color=Zi, node_size=80, cmap=plt.get_cmap('Reds'), with_labels=False)  # 图结构可视化
    #     # fig1.colorbar()
    #     Z = (Z - min(Z)) / (max(Z) - min(Z)).tolist()
    #     fig2=plt.subplot(1, 2, 2)
    #     nx.draw_networkx(G, pos, node_color=Z, node_size=80, cmap=plt.get_cmap('PiYG'), with_labels=False)  # 图结构可视化
    #     # fig2.colorbar()
    #     plt.title(t)   # plt.show()
    #     pyplot.savefig("C://Users//MH//Desktop//bian//V20{}.png".format(n), dpi=512,
    #                    bbox_inches='tight')


    n+=1
#
# pyplot.figure(figsize=(9, 4))
# x=[0,12*6,12*12,12*18,12*24]
# xlabel=['0:00','6:00','12:00','18:00','24:00']
# pyplot.plot(I_moran,linestyle='--',marker='^',markersize=6, markerfacecolor='orange'
#             )
#
# pyplot.xticks(x,xlabel)
# pyplot.ylabel('全局莫兰指数（MoranI）',fontsize=10)
# pyplot.xlabel('时间（5min）',fontsize=10)
#
# pyplot.savefig("C://Users//MH//Desktop//result2.png", dpi=512,
#                                bbox_inches='tight')
#
# pyplot.show()
#



# A = np.random.rand(125,125)
# print(Moran.moranI(A,Z))
# mi = pysal.Moran(A, Z)


# nx.draw_networkx(G, pos,node_size=30, with_labels=False)   #图结构可视化
# plt.show()
#
# # ************************************************************************ #
# # part 1--creat a graph
# # ************************************************************************ #
#
#
# # plt.figure(figsize=(10, 10))
# # pos = nx.shell_layout(G)
# # nx.draw(G, pos, with_labels=True, **options)
# # nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#034b61")
# #
# # plt.show()
#
# # ************************************************************************ #
# # part 2--community detection
# # ************************************************************************ #

com = community.greedy_modularity_communities(G,weight='time')
num_com = len(com)
#
# find intra_com links
intra_links = {}
for i in range(num_com):
    intra_links[i] = []

for link in nx.edges(G):
    for i in range(num_com):
        if (link[0] in com[i]) & (link[1] in com[i]):
            intra_links[i].append(link)

# com_center = com_postion(num_com, scale=3)  # print(com_center)
# pos = dict()
# for val in range(num_com):
#     node_pos = node_postion(com[val], scale=1.3, center=com_center[val])
#     pos.update(node_pos)
#
# plt.figure(figsize=(10, 10))
# nx.draw(G, pos, with_labels=True, edgelist=[])
# nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

for val in range(num_com):
    nx.draw_networkx_nodes(G, pos, node_size=30, nodelist=list(com[val]), node_color=colors[val])
    nx.draw_networkx_edges(G, pos, alpha=0.7, edgelist=intra_links[val], width=1.5)

plt.axis("off")
# plt.savefig(savefig_path + 'greedy.png', format='png', dpi=500)
plt.show()


for val in range(num_com):
    nx.draw_networkx(G, pos, node_size=30, nodelist=list(com[val]), node_color=colors[val],with_labels=False)

plt.axis("off")
# plt.savefig(savefig_path + 'greedy.png', format='png', dpi=500)
plt.show()



