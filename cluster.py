from matplotlib import pyplot

from pylab import *
from datetime import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

import pandas as pd
import pyproj
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AffinityPropagation

import Moran

from pyproj import CRS
from sklearn.cluster import SpectralClustering
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


colors = ['#034b61', '#5d353e', '#b2d6ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca',
          '#fe5858', '#034b61', '#5d353e', '#b2d6ca', '#b725ca', '#a525ca', '#b2d6ca', '#b725ca', '#a525ca']
options = {'font_family': 'serif', 'font_weight': 'semibold', 'font_size': '12', 'font_color': '#ffffff'}
savefig_path = 'F:/Python/NetworkSci/'


def calculate_Wk(data, centroids, cluster):
    K = centroids.shape[0]
    wk = 0.0
    for k in range(K):
        data_in_cluster = data[cluster == k, :]
        center = centroids[k, :]
        num_points = data_in_cluster.shape[0]
        for i in range(num_points):
            wk = wk + np.linalg.norm(data_in_cluster[i, :] - center, ord=2) ** 2
    return wk


def bounding_box(data):
    dim = data.shape[1]
    boxes = []
    for i in range(dim):
        data_min = np.amin(data[:, i])
        data_max = np.amax(data[:, i])
        boxes.append((data_min, data_max))
    return boxes


def gap_statistic(data, max_K, B, cluster_algorithm):
    num_points, dim = data.shape
    K_range = np.arange(1, max_K, dtype=int)
    num_K = len(K_range)
    boxes = bounding_box(data)
    data_generate = np.zeros((num_points, dim))
    log_Wks = np.zeros(num_K)
    gaps = np.zeros(num_K)
    sks = np.zeros(num_K)
    for ind_K, K in enumerate(K_range):
        km = cluster_algorithm(K)
        km.fit(data)
        cluster_centers, labels = km.cluster_centers_, km.labels_
        log_Wks[ind_K] = np.log(calculate_Wk(data, cluster_centers, labels))
        # generate B reference data sets
        log_Wkbs = np.zeros(B)
        for b in range(B):
            for i in range(num_points):
                for j in range(dim):
                    data_generate[i][j] = \
                        np.random.uniform(boxes[j][0], boxes[j][1])
            km = cluster_algorithm(K)
            km.fit(data_generate)
            cluster_centers, labels = km.cluster_centers_, km.labels_
            log_Wkbs[b] = \
                np.log(calculate_Wk(data_generate, cluster_centers, labels))
        gaps[ind_K] = np.mean(log_Wkbs) - log_Wks[ind_K]
        sks[ind_K] = np.std(log_Wkbs) * np.sqrt(1 + 1.0 / B)

    return gaps, sks, log_Wks


def adj(unique_list, node_attr, dict_node):
    graph = {}  # 邻接表  字典形式
    OD = []
    for node in unique_list:
        O_start, O_end = node.split('-', 1)[0], node.split('-', 1)[1]  # O路段的起始
        node_value = {}  # 储存一个O的所有D
        for name in unique_list:  # 遍历所有可能的D
            if name != node:
                loc = name.split('-', 1)

                D_start, D_end = loc[0], loc[1]  # D路段的起始

                if O_end in loc:  # 如果O路段的起始在D中，则确定一对OD

                    # 为每一个路段节点 赋予流出的权重
                    if O_end == D_start:  # 按照O的正常顺序 赋予流出的权重
                        key = name
                    if O_end == D_end:
                        key = '{}-{}'.format(D_end, D_start)

                    value = (dict_attr[key][-3] + dict_attr[node][-3]) / 2  # 两个路段距离的一半

                    value1 = 0.5 * (node_attr[dict_node[key]] + node_attr[dict_node[node]])
                    # (node_attr[dict_node[key]] + node_attr[dict_node[node]])
                    value2 = abs(node_attr[dict_node[key]] - node_attr[dict_node[node]])

                    weight = {'weight': value}
                    node_value[name] = weight
                    # OD.append([node,name,1])
                if O_start in loc:  # 按照O的反向顺序 赋予流出的权重

                    if O_start == D_start:
                        key = name
                    if O_start == D_end:
                        key = '{}-{}'.format(D_end, D_start)

                    inv_node = '{}-{}'.format(O_end, O_start)
                    value = (dict_attr[key][-3] + dict_attr[inv_node][-3]) / 2
                    value1 = 0.5 * (node_attr[dict_node[key]] + node_attr[dict_node[inv_node]])
                    # (node_attr[dict_node[key]] + node_attr[dict_node[node]])
                    value2 = abs(node_attr[dict_node[key]] - node_attr[dict_node[inv_node]])
                    # value_t = 2*(dict_attr[key][-3] + dict_attr[inv_node][-3])/(node_attr[dict_node[key]] + node_attr[dict_node[inv_node]])
                    weight = {'weight': value}
                    node_value[name] = weight

                    # OD.append([node, name, 1])

        graph[node] = node_value

    return graph


def similiar(G, weight_in, node_attr=None, dict_node=None, alpah=0.05, beta=0.3, delta=3):
    A = nx.to_numpy_array(G)

    i = 0

    for u in G.nodes():  # 遍历图的每个点
        u_start, u_end = u.split('-', 1)[0], u.split('-', 1)[1]  # O路段的起始
        inv_u = '{}-{}'.format(u_end, u_start)

        path = nx.shortest_path_length(G, source=u, weight='weight')  # 在网络G中计算从u开始到其他所有节点（注意包含自身）的最短路径长度。
        j = 0
        for v in unique_list:  # path是一个字典，里面存了所有目的地节点到u的最短路径长度
            v_start, v_end = v.split('-', 1)[0], v.split('-', 1)[1]  # O路段的起始
            inv_v = '{}-{}'.format(v_end, v_start)

            dis = path[v]
            # print(dis)  # 距离为最短路径
            if u == v:  # 如果起止点相同 距离=1
                if weight_in == 'weight':
                    value = 1
                elif weight_in == 'similarity':
                    value = 1
                A[i, j] = value
                # dis = 1

            else:
                if weight_in == 'weight':
                    value = 1 / pow(dis, 2)
                    # print(value)
                elif weight_in == 'similarity':

                    V1 = node_attr[dict_node[u]] + node_attr[dict_node[inv_u]]
                    V2 = node_attr[dict_node[v]] + node_attr[dict_node[inv_v]]
                    V3 = abs(node_attr[dict_node[u]] - node_attr[dict_node[inv_u]])
                    V4 = abs(node_attr[dict_node[v]] - node_attr[dict_node[inv_v]])
                    value = math.exp(
                        -alpah * math.sqrt((V1 - V2) ** 2 + (V3 - V4) ** 2) - beta * dis * dis / (delta ** 2))
                    # print(dis , math.exp(-0.5*dis*dis/(4**2)),
                    #       math.exp(-0.05*math.sqrt((V1-V2)**2+(V3-V4)**2)),V1-V2,V3-V4)

                A[i, j] = value
                # print(value)

            j += 1
        i += 1
    A = A.astype(float16)
    return A


# attr = pd.read_csv('C://Users//MH//Desktop//anomoly//2019-12-08.csv', header=None,
#                    names=['路段名称', '路段长度', '拥堵指数', '平均速度'],
#                    encoding='utf-8', index_col=0)
# timeindex = np.unique(attr.index.values).tolist()
# attr_s = attr.loc[timeindex[0]]
# name = attr_s['路段名称'].values.reshape(1,250).tolist()
# pd.DataFrame(name).to_csv('C://Users//MH//Desktop//speed.csv', index=False,
#             mode='a', line_terminator='\n',header=None)
#
#
#
#
# for root, dirs, files in os.walk('C://Users//MH//Desktop//anomoly//'):
#     for file in files:
#         filepath = os.path.join(root, file)
#         print(filepath)
#
#         attr = pd.read_csv(filepath, header=None,
#                            names = ['路段名称','路段长度','拥堵指数','平均速度'] ,
#                            encoding='utf-8',index_col=0)
#         timeindex = np.unique(attr.index.values).tolist()
#         attr_s = attr.loc[timeindex[0]]
#         name = attr_s['路段名称'].values.tolist()
#         for t in timeindex:
#             attr_s = attr.loc[t]
#             pd.DataFrame(attr_s['平均速度'].values.reshape(1,250).astype(float16)).to_csv('C://Users//MH//Desktop//speed.csv', index=False,
#             mode='a', line_terminator='\n',header=None)


run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
subpath = 'C://Users//MH//Desktop//bian//cluster//{}//'.format(run_id)
if not os.path.exists(subpath):
    os.makedirs(subpath)

attr = pd.read_csv('C://Users//MH//Desktop//anomoly//2019-12-08.csv', header=None,
                   names=['路段名称', '路段长度', '拥堵指数', '平均速度'],
                   encoding='utf-8', index_col=0)
timeindex = np.unique(attr.index.values).tolist()
attr_s = attr.loc[timeindex[96]]
# data_fecha.loc[fecha_1: fecha_2]
dict_attr = {}
for i in (range(attr_s.shape[0])):
    dict_attr[attr_s['路段名称'][i]] = [attr_s['路段长度'][i]
        , attr_s['拥堵指数'][i], attr_s['平均速度'][i]]

data = pd.read_csv('C://Users//MH//Desktop//anomoly//2019-12-08.csv', header=None,
                   names=['路段名称', '路段长度', '拥堵指数', '平均速度'],
                   encoding='utf-8', index_col=0)

data.groupby('路段名称')
list_value = []
list_name = []
for name, group in data.groupby('路段名称'):
    list_name.append(name)
    list_value.append(group)

unique_list = []  # 重复路段名中取一个
for name in list_name:  # name:路段名    b_name:反向路段名

    start, end = name.split('-', 1)[0], name.split('-', 1)[1]
    b_name = '{}-{}'.format(end, start)
    if not name in unique_list:
        if not b_name in unique_list:
            unique_list.append(name)

# -------------坐标转换
crs = CRS.from_epsg(4326)
crs_cs = CRS.from_epsg(3857)

transformer = pyproj.Transformer.from_crs(crs, crs_cs)
data = pd.read_excel("C://Users//MH//Desktop//node.xlsx")
lon = data.lon.values
lat = data.lat.values
x2, y2 = transformer.transform(lat, lon)

pos = {}
i = 0
for node in unique_list:
    pos[node] = np.array([x2[i], y2[i]])
    i += 1

# ----------空间位置

graph = {}  # 邻接表  字典形式
OD = []
v = []
for node in unique_list:
    O_start, O_end = node.split('-', 1)[0], node.split('-', 1)[1]  # O路段的起始
    node_value = {}  # 储存一个O的所有D
    for name in unique_list:  # 遍历所有可能的D
        if name != node:
            loc = name.split('-', 1)

            D_start, D_end = loc[0], loc[1]  # D路段的起始

            if O_end in loc:  # 如果O路段的起始在D中，则确定一对OD

                # 为每一个路段节点 赋予流出的权重
                if O_end == D_start:  # 按照O的正常顺序 赋予流出的权重
                    key = name
                if O_end == D_end:
                    key = '{}-{}'.format(D_end, D_start)

                value = (dict_attr[key][-3] + dict_attr[node][-3]) / 2  # 两个路段距离的一半

                value1 = 10 * (dict_attr[key][-3] + dict_attr[node][-3]) / (dict_attr[key][-1] + dict_attr[node][-1])
                v.append(value1)
                weight = {'weight': value, 'time': value1}
                node_value[name] = weight
                # OD.append([node,name,1])
            if O_start in loc:  # 按照O的反向顺序 赋予流出的权重

                if O_start == D_start:
                    key = name
                if O_start == D_end:
                    key = '{}-{}'.format(D_end, D_start)

                inv_node = '{}-{}'.format(O_end, O_start)
                value = (dict_attr[key][-3] + dict_attr[inv_node][-3]) / 2
                value1 = 10 * (dict_attr[key][-3] + dict_attr[inv_node][-3]) / (
                        dict_attr[key][-1] + dict_attr[inv_node][-1])
                v.append(value1)
                weight = {'weight': value, 'time': value1}
                node_value[name] = weight

                # OD.append([node, name, 1])

    graph[node] = node_value

vmax = max(v)
vmin = min(v)

#    归一化
graph = {}  # 邻接表  字典形式
OD = []
for node in unique_list:
    O_start, O_end = node.split('-', 1)[0], node.split('-', 1)[1]  # O路段的起始
    node_value = {}  # 储存一个O的所有D
    for name in unique_list:  # 遍历所有可能的D
        if name != node:
            loc = name.split('-', 1)

            D_start, D_end = loc[0], loc[1]  # D路段的起始

            if O_end in loc:  # 如果O路段的起始在D中，则确定一对OD

                # 为每一个路段节点 赋予流出的权重
                if O_end == D_start:  # 按照O的正常顺序 赋予流出的权重
                    key = name
                if O_end == D_end:
                    key = '{}-{}'.format(D_end, D_start)

                value = (dict_attr[key][-3] + dict_attr[node][-3]) / 2  # 两个路段距离的一半

                value1 = 50 * (dict_attr[key][-3] + dict_attr[node][-3]) / (dict_attr[key][-1] + dict_attr[node][-1])
                weight = {'weight': value, 'time': pow((value1 - vmin) / (vmax - vmin), 2) * 10}
                node_value[name] = weight
                # OD.append([node,name,1])
            if O_start in loc:  # 按照O的反向顺序 赋予流出的权重

                if O_start == D_start:
                    key = name
                if O_start == D_end:
                    key = '{}-{}'.format(D_end, D_start)

                inv_node = '{}-{}'.format(O_end, O_start)
                value = (dict_attr[key][-3] + dict_attr[inv_node][-3]) / 2
                value1 = 50 * (dict_attr[key][-3] + dict_attr[inv_node][-3]) / (
                        dict_attr[key][-1] + dict_attr[inv_node][-1])
                weight = {'weight': value, 'time': pow((value1 - vmin) / (vmax - vmin), 2) * 10}
                node_value[name] = weight

                # OD.append([node, name, 1])

    graph[node] = node_value
# {1: {"weight": 1}}

# f = pd.DataFrame(OD)


G = nx.DiGraph(graph)  # 有向图
# G = nx.Graph(graph)
plt.figure(figsize=(9, 6))


nx.draw_networkx(G, pos, node_size=100, font_size=6,  with_labels=True)
  #图结构可视化
plt.show()
# pyplot.savefig(subpath + "graph.png", dpi=600)
# plt.close()

#  聚类----------------------------
data = pd.read_csv('C://Users//MH//Desktop//speed.csv', encoding='utf-8')

col = data.columns.tolist()

dict_col = {}
for i in range(len(col)):
    dict_col[col[i]] = i

# res = []
# for eps in [200,250,300]:
#     # 迭代不同的min_samples值
#     for min_samples in [200,500,1000]:
#         dbscan = DBSCAN(eps = eps, min_samples = min_samples)
#         # 模型拟合
#         dbscan.fit(data.values)
#         # 统计各参数组合下的聚类个数（-1表示异常点）
#         n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
#         # 异常点的个数
#         outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
#         # 统计每个簇的样本个数
#         stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
#         res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})
# # 将迭代后的结果存储到数据框中
# df = pd.DataFrame(res)


# gaps = gap_statistic(data.values, 11, 50, KMeans)[0]
# X = range(1,11)
# plt.xlabel('k')
# plt.ylabel('Gap Statistic')
# plt.plot(X,gaps,'o-')
# plt.show()

# plt.style.use('ggplot')
K = 4
Km = KMeans(n_clusters=K, random_state=0).fit(data.values)  # 构造聚类器

center = Km.cluster_centers_

labels = Km.labels_.tolist()
for i in range(len(labels)):
    labels[i] += 1


plt.figure(figsize=(15, 6))
colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#38B0DE', '#b725ca']
for day in range(1):
    day= day + 2

    for i in range(4):
        # plt.subplot(7, 1, i + 1)

        pyplot.plot(labels[288 * (day) + 288 * (i * 7):288 * (day) + 288 * (i * 7 + 1)],
                    linewidth=2,
                    color=colors[day-2])



    plt.xlabel('时间', fontsize=18)  # 设置x，y轴的标签
    plt.ylabel('交通状态类别', fontsize=18)
    x = [0, 12 * 6, 12 * 12, 12 * 18, 12 * 24]
    y = [1, 2, 3, 4]
    xlabel = ['0:00', '6:00', '12:00', '18:00', '24:00']
    ylabel = ['类别1', '类别2', '类别3', '类别4']
    plt.title("工作日", fontsize = 20)


    pyplot.xticks(x, xlabel)
    pyplot.yticks(y, ylabel)

    plt.tick_params(labelsize=15)
pyplot.savefig(subpath + "工作日.png", dpi=512,
               bbox_inches='tight')
plt.show()




plt.figure(figsize=(15, 6))
for i in range(3):
    # plt.subplot(7, 1, i + 1)
    i = i + 1
    pyplot.plot(labels[288 * (0) + 288 * (i * 7):288 * (0) + 288 * (i * 7 + 1)],
                linewidth=2, color = colors[0]
                )
for i in range(4):
    # plt.subplot(7, 1, i + 1)

    pyplot.plot(labels[288 * (1) + 288 * (i * 7):288 * (1) + 288 * (i * 7 + 1)],
                linewidth=2, color=colors[1]
                )


plt.xlabel('时间', fontsize=18)  # 设置x，y轴的标签
plt.ylabel('交通状态类别', fontsize=18)
x = [0, 12 * 6, 12 * 12, 12 * 18, 12 * 24]
y = [1, 2 ,3 ,4]
xlabel = ['0:00', '6:00', '12:00', '18:00', '24:00']
ylabel = ['类别1', '类别2', '类别3', '类别4']
plt.title("非工作日", fontsize = 20)
# pyplot.plot(I_moran,linestyle='--',marker='^',markersize=6, markerfacecolor='orange'
#             )
pyplot.xticks(x, xlabel)
pyplot.yticks(y, ylabel)

plt.tick_params(labelsize=15)
pyplot.savefig(subpath + "非工作日.png", dpi=512,
               bbox_inches='tight')
plt.show()
# lab = np.array(labels)
# a = np.argwhere(Km.labels_ == 1)

#
# for i in range(len(labels)-8000):
#     plt.plot(data.values[i, ], color=colors[labels[i]], alpha= 0.1, linewidth=2)

# marker = ['o', '*', '^', '+', 'o', '*', '^', '+', 'o', '*', '^', '+']
# label_num = []
# for i in range(K):
#     # lab = np.array(labels)
#
#     plt.figure(figsize=(21, 8))
#
#     index = np.argwhere(Km.labels_ == i).squeeze().tolist()
#     label_num.append(len(index))
#     x = np.array([i for i in range(250)])
#     # plt.plot(data.values[index,:].T,
#     #          color=colors[i],
#     #          solid_capstyle='butt',
#     #             alpha=0.02, linewidth=2)
#     upper = np.max(data.values[index,:].T,axis=1)
#     downer = np.min(data.values[index,:].T,axis=1)
#     upper =  np.percentile(data.values[index, :].T, 98, axis=1)
#     downer = np.percentile(data.values[index, :].T, 2, axis=1)
#     plt.fill_between(x, downer,upper, facecolor='green', alpha = 0.3)
#     upper =  np.percentile(data.values[index, :].T, 75, axis=1)
#     downer = np.percentile(data.values[index, :].T, 25, axis=1)
#
#     plt.fill_between(x, downer,upper, facecolor='green', alpha = 0.4)
#     plt.plot(center[i], color='red', marker =marker[i],
#              markersize =15,
#              linewidth=2, alpha=1,label ='Class:{}'.format(i+1))
#     y_tick = range(10, 90, 10)
#     plt.xlabel('路段编号',fontsize=30)  # 设置x，y轴的标签
#     plt.ylabel('平均速度（Km/h）',fontsize=30)
#
#     plt.yticks(y_tick)
#     plt.tick_params(labelsize=30)
#     plt.legend(fontsize = 30)
#     pyplot.savefig(subpath + "CLA{}.png".format(i+1), dpi=512,
#                    bbox_inches='tight')
#     plt.show()
#     plt.close()
#


# SSE = []  # 存放每次结果的误差平方和
# Scores = []  # 存放轮廓系数
# for k in range(1, 9):
#     estimator = KMeans(n_clusters=k)  # 构造聚类器
#     estimator.fit(data.values)
#     SSE.append(estimator.inertia_)
#     if k>1:
#         Scores.append(silhouette_score(data.values, estimator.labels_, metric='euclidean'))
#
# X = range(2, 9)
# plt.xlabel('k', fontsize=15)
# plt.ylabel('Silhouette Coefficient', fontsize=15)
# plt.plot(X, Scores, 'o-')
# pyplot.savefig(subpath + "sh.png", dpi=512,
#                bbox_inches='tight')
# plt.show()
# X = range(1, 9)
# plt.xlabel('k', fontsize=15)
# plt.ylabel('SSE', fontsize=15)
# plt.plot(X, SSE, 'o-')
#
#
# pyplot.savefig(subpath + "sse.png", dpi=512,
#                bbox_inches='tight')
# plt.show()
#
# CH = []
# for k in range(2,11):
#     estimator = KMeans(n_clusters=k)  # 构造聚类器
#     estimator.fit(data.values)
#     CH.append(davies_bouldin_score(data.values,estimator.labels_))
# X = range(2,11)
# plt.xlabel('k')
# plt.ylabel('davies_bouldin_score')
# plt.plot(X,CH,'o-')
# plt.show()
colors_list =[['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#38B0DE', '#b725ca'],
              ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#CDCDCD', '#38B0DE'],
              ['#034b61', '#5d353e', '#E9C2A6', '#FF2400', '#38B0DE', '#CDCDCD'],
              ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#b725ca', '#38B0DE']]
# colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#38B0DE', '#b725ca']
# colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#CDCDCD', '#38B0DE']
# colors = ['#034b61', '#5d353e', '#E9C2A6', '#FF2400', '#38B0DE', '#CDCDCD']
# colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#b725ca', '#38B0DE']


I_moran = []

Z = np.random.rand(len(G.nodes))  # 观测属性
Z_cx = np.random.rand(len(G.nodes))

Zi_list = []
Z_list = []
for n in range(Km.n_clusters):

    center = Km.cluster_centers_[n, :]

    # 利用每个类中心生成邻接矩阵
    graph = adj(unique_list, node_attr=center, dict_node=dict_col)

    G = nx.Graph(graph)  # 有向图
    # A = nx.to_numpy_array(G, weight='weight')
    A = similiar(G, weight_in='weight')

    # for i in range(len(A)):
    #     if sum(A[i, :]) != 0:  # 如果第i行的和不为0
    #         A[i, :] =  A[i, :] / sum(A[i, :])  # 第i行按行归一化
    i = 0
    for name in unique_list:
        start, end = name.split('-', 1)[0], name.split('-', 1)[1]
        b_name = '{}-{}'.format(end, start)
        Z[i] = (center[dict_col[name]] + center[dict_col[b_name]]) / 2
        Z_cx[i] = abs((center[dict_col[name]] - center[dict_col[b_name]]))
        i += 1
    Z = Z.astype(float16)
    Z_cx = Z_cx.astype(float16)
    # eval_on_center['class{}'.format(n)] = {'Z':pd.DataFrame(Z).describe(),
    #                                        'Z_cx':pd.DataFrame(Z_cx).describe()}
    # {'cx_range':[min(Z_cx),max(Z_cx)],
    #                                        'v_range':[min(Z),max(Z)],
    #                                        'cx_mean': [mean(Z_cx)],
    #                                        'v_mean': [mean(Z)],
    #                                        }
    result = Moran.moranI(A, Z,
                          imgPath=subpath + '莫兰散点图{}.png'.format(n))
    I_moran.append(result['I']['value'])

    Zi = result['Ii']['value']
    Zi_list.append(Zi)
    Z_list.append(Z)

    # Zi[Zi > 0] = Zi[Zi > 0] / max(Zi[Zi > 0] )
    # Zi[Zi < 0] = -Zi[Zi < 0] / min(Zi[Zi < 0])
Zi_list = np.array(Zi_list).flatten()
Z_list = np.array(Z_list).flatten()

eval_on_center = {}
plt.style.use('mpl20')
com_class = []
db_list = []
sh_list = []
for a in [0.05]:
    for b in [0]:
        for c in [4]:
            for num_com in [6]:
                # plt.figure(figsize=(21, 16))
                db_score=[]
                sh_score=[]
                for n in range(Km.n_clusters):

                    center = Km.cluster_centers_[n, :]

                    # 利用每个类中心生成邻接矩阵
                    graph = adj(unique_list, node_attr=center, dict_node=dict_col)

                    G = nx.Graph(graph)  # 有向图
                    # A = nx.to_numpy_array(G, weight='weight')

                    A = similiar(G, weight_in='weight')

                    # for i in range(len(A)):
                    #     if sum(A[i, :]) != 0:  # 如果第i行的和不为0
                    #         A[i, :] =  A[i, :] / sum(A[i, :])  # 第i行按行归一化
                    A_s = similiar(G, weight_in='similarity',
                                   alpah=a, beta=b, delta=c, node_attr=center, dict_node=dict_col)
                    #  利用相似度矩阵谱聚类
                    # num_com=8
                    clustering = SpectralClustering(n_clusters=num_com,
                                                    assign_labels="discretize",  # "discretize",
                                                    affinity='precomputed',
                                                    random_state=5).fit(A_s)

                    # CH = []
                    # for k in range(2,11):
                    # estimator = SpectralClustering(n_clusters=k,
                    #                             assign_labels="kmeans",#"discretize",
                    #                             affinity='precomputed',
                    #                             random_state=5).fit(A_s) # 构造聚类器

                    DB = davies_bouldin_score(A_s, clustering.labels_)
                    SSE = 0
                    Score = silhouette_score(A_s, clustering.labels_)
                    #     if k>1:
                    #         Scores.append(silhouette_score(data.values, estimator.labels_, metric='euclidean'))
                    # find intra_com links
                    i = 0
                    for name in unique_list:
                        start, end = name.split('-', 1)[0], name.split('-', 1)[1]
                        b_name = '{}-{}'.format(end, start)
                        Z[i] = (center[dict_col[name]] + center[dict_col[b_name]]) / 2
                        Z_cx[i] = abs((center[dict_col[name]] - center[dict_col[b_name]]))
                        i += 1

                    Z = Z.astype(float16)
                    Z_cx = Z_cx.astype(float16)

                    feature = np.concatenate((Z.reshape(125,1),
                                              Z_cx.reshape(125,1),
                                              lon.reshape(125,1),
                                              lat.reshape(125,1)),
                                             axis=1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled = scaler.fit_transform(feature)

                    DB = davies_bouldin_score(scaled, clustering.labels_)

                    SSE = 0
                    Score = silhouette_score(scaled, clustering.labels_,metric='euclidean')
                    db_score.append(DB)
                    sh_score.append(Score)
                    # print(a, num_com, DB, Score)




                    nodes = np.array(unique_list)
                    intra_links = {}
                    com = {}
                    sub_A={}
                    for i in range(num_com):
                        intra_links[i] = []
                        index = np.argwhere(clustering.labels_ == i).squeeze().tolist()
                        com[i] = nodes[index].tolist()
                        subA = A_s[index,:].copy()
                        sub_A[i] = subA[:, index]

                        result = {'DB': pd.DataFrame(Z[index]).describe().values,
                                  'Z_cx': pd.DataFrame(Z_cx[index]).describe().values}

                        # speed = pd.DataFrame(Z[index]).describe().T
                        # dif = pd.DataFrame(Z_cx[index]).describe().T
                        #
                        # df1 = pd.DataFrame(speed.values, columns=speed.columns,
                        #                    index=['SPEclass{}-{}'.format(n, i + 1)])
                        # df2 = pd.DataFrame(dif.values, columns=speed.columns,
                        #                    index=['DIFclass{}-{}'.format(n, i + 1)])
                        # if i == 0:
                        #     df1.to_csv(subpath + 'score.csv', mode='a', line_terminator='\n')
                        #     df2.to_csv(subpath + 'score.csv', mode='a', line_terminator='\n', header=None, )
                        # else:
                        #     df1.to_csv(subpath + 'score.csv', mode='a', header=None, line_terminator='\n')
                        #     df2.to_csv(subpath + 'score.csv', mode='a', header=None, line_terminator='\n')

                    for link in nx.edges(G):
                        for i in range(num_com):
                            if (link[0] in com[i]) & (link[1] in com[i]):
                                intra_links[i].append(link)

                    # plt.subplot(2,2,n+1)
                    plt.figure(figsize=(21, 16))
                    # plt.subplot(2, 2, 1)
                    for val in range(num_com):
                        if type(com[val]) != list:
                            com[val] = [com[val]]
                        nx.draw_networkx_nodes(G, pos, node_size=30, nodelist=com[val], node_color=colors[0])
                        nx.draw_networkx_edges(G, pos, alpha=0.7, edgelist=intra_links[val], width=1.5)
                    plt.show()


                    clu = clustering.labels_.tolist()
                    com_class.append(clu)
                    clu_color = [colors_list[1][x] for x in clu]

                    # plt.subplot(2, 2, 2)
                    nx.draw_networkx(G, pos, node_color=clu_color,
                                     node_size=80,
                                     cmap=plt.get_cmap('RdBu'),
                                     linewidths=0.5,
                                     arrowsize=5,
                                     with_labels=False)  # 图结构可视化
                    plt.show()
                    #
                    #

                    # plt.axis("off")
                    # plt.savefig(savefig_path + 'greedy.png', format='png', dpi=500)

                    # eval_on_center['class{}'.format(n)] = {'Z':pd.DataFrame(Z).describe(),
                    #                                        'Z_cx':pd.DataFrame(Z_cx).describe()}
                    # {'cx_range':[min(Z_cx),max(Z_cx)],
                    #                                        'v_range':[min(Z),max(Z)],
                    #                                        'cx_mean': [mean(Z_cx)],
                    #                                        'v_mean': [mean(Z)],
                    #                                        }

                    Zmin = min(Z_list)
                    Zmax = max(Z_list)
                    maxZ = max(Z)
                    minZ = min(Z)
                    meanZ = mean(Z)
                    medianZ = median(Z)
                    Z = ((Z - 15) / (85 - 15))

                    fig2 = plt.subplot(2, 2, 3)
                    nx.draw_networkx(G, pos, node_color=Z,
                                     node_size=80,
                                     cmap=plt.get_cmap('RdYlGn'),
                                     with_labels=False,
                                     linewidths=0.5,
                                     arrowsize=5,
                                     vmin=0.0,
                                     vmax=1.0,
                                     label='Speed[{:.2f}-{:.2f}]   mean:{:.2f}  median:{:.2f}'.
                                     format(minZ, maxZ, meanZ, medianZ),
                                     style='dashdot')  # 图结构可视化

                    plt.legend()

                    Z_cxmax = max(Z_cx)
                    Z_cxmin = min(Z_cx)

                    fig3 = plt.subplot(2, 2, 4)
                    nx.draw_networkx(G, pos, node_color=Z_cx,
                                     node_size=80,
                                     cmap=plt.get_cmap('Reds'),
                                     with_labels=False,
                                     vmin=0,
                                     vmax=50,
                                     linewidths=0.5,
                                     arrowsize=5,
                                     label='Speed_Dif[{:.2f}-{:.2f}]'.format(Z_cxmin, Z_cxmax),
                                     style='dashdot')  # 图结构可视化

                    # fig2.colorbar()
                    plt.legend()
                    plt.title("a={}b={}c={}k={}".format(a, b, c, num_com))
                    pyplot.savefig(subpath + "class{}.png".format(n), dpi=200,
                                   bbox_inches='tight')

                    result = {'DB': DB, 'SSE':SSE,'Score':Score}

                    plt.close()


                    plt.style.use('mpl20')
                    for val in range(num_com):
                        # clu_color = [colors_list[val][x] for x in com_class[val]]
                        subG = nx.Graph()
                        subG.add_nodes_from(com[val])
                        subG.add_edges_from(intra_links[val])
                        # sub_num_com = 3
                        for sub_num_com in [2,3,4,5,6]:
                            sub_clu = SpectralClustering(n_clusters=sub_num_com,
                                                     assign_labels="discretize",  # "discretize",
                                                     affinity='precomputed',
                                                     random_state=5).fit(sub_A[val])
                            sub_node = len(sub_clu.labels_.tolist())
                            cla_num_list=[]
                            for cla_num in range(sub_num_com):
                                cla_num_list.append(sub_clu.labels_.tolist().count(cla_num))

                            if max(cla_num_list) < 4:
                                sub_clu = SpectralClustering(n_clusters=sub_num_com-1,
                                                             assign_labels="discretize",  # "discretize",
                                                             affinity='precomputed',
                                                             random_state=5).fit(sub_A[val])
                                break

                            if min(cla_num_list)/sub_node < 0.2 and max(cla_num_list)/sub_node < 0.3:
                                sub_clu = SpectralClustering(n_clusters=sub_num_com-1,
                                                             assign_labels="discretize",  # "discretize",
                                                             affinity='precomputed',
                                                             random_state=5).fit(sub_A[val])

                                break

                        print(min(cla_num_list),sub_num_com)

                        # clu_colorsub_clu.labels_.tolist(),

                        sub_intra_links = {}
                        sub_com = {}

                        for i in range(sub_num_com):
                            sub_intra_links[i] = []
                            sub_index = np.argwhere(sub_clu.labels_ == i).squeeze().tolist()
                            sub_com[i] = np.array(com[val])[sub_index].tolist()



                        for link in nx.edges(subG):
                            for i in range(sub_num_com):
                                if (link[0] in sub_com[i]) & (link[1] in sub_com[i]):
                                    sub_intra_links[i].append(link)


                        edge_list=['black','red','green','blue','purple','pink']
                        shape_list = ['o', '^', 's', 'p', 'D','H']
                        for sub_val in range(sub_num_com):
                            if type(sub_com[sub_val]) != list:
                                sub_com[sub_val] = [sub_com[sub_val]]
                            nx.draw_networkx_nodes(subG, pos, node_size=50, nodelist=sub_com[sub_val],
                                                   # edgecolors = edge_list[sub_val],
                                                   node_shape =shape_list[sub_val] ,
                                                   node_color=colors_list[n][val])

                            nx.draw_networkx_edges(subG, pos, alpha=0.7, edgelist=sub_intra_links[sub_val], width=1.5)
                    pyplot.savefig(subpath + "sub_class{}.png".format(n), dpi=200,
                                   bbox_inches='tight')
                    plt.close()
                    # nx.draw_networkx(subG, pos, node_color=colors_list[n][val],
                    #                      node_size=80,
                    #                      cmap=plt.get_cmap('RdBu'),
                    #                      linewidths=0.5,
                    #                      arrowsize=5,
                    #                      with_labels=False)  # 图结构可视化


                db_list.append(mean(db_score))
                sh_list.append(mean(sh_score))


# db_list[16] = db_list[16] + 0.2
# db_list[17] = db_list[17] + 0.2
#
# n = 4
# title = [0.01, 0.05, 0.1, 0.5]
# plt.figure(figsize=(15, 6))
# x_tick = [0, 1, 2, 3, 4, 5]
# x_label = [3, 4, 5, 6, 7, 8]
#
# for i in range(n):
#     plt.subplot(2, n, i+1)
#
#     plt.plot(db_list[i*6:(i+1)*6], marker='o', markersize=10)
#     y_tick = [1.2, 1.4, 1.6, 1.8]
#
#
#     plt.xticks(x_tick, x_label, fontsize=15)
#     plt.title(r'$\alpha$'+'={}'.format(title[i]), fontsize=16)
#
#
#     if i == 0:
#         plt.yticks(y_tick, fontsize=15)
#         pyplot.ylabel('Davies-Bouldin Index',fontsize=15)
#     else:
#         plt.yticks(y_tick, [], fontsize=15)
#
#
#     plt.subplot(2, n, i+5)
#     plt.plot(sh_list[i*6:(i+1)*6], marker='^', markersize=10, color=colors[0])
#     y_tick = [0.1, 0.2, 0.3]
#
#     plt.xticks(x_tick, x_label, fontsize=15)
#     if i == 0:
#         plt.yticks(y_tick, fontsize=15)
#         pyplot.ylabel('Silhouette Coefficient',fontsize=15)
#     else:
#         plt.yticks(y_tick, [], fontsize=15)
#
# pyplot.savefig(subpath + "eval.png", dpi=512,
#                bbox_inches='tight')
# plt.show()
plt.style.use('mpl20')
for n in range(Km.n_clusters):

    center = Km.cluster_centers_[n, :]

    # 利用每个类中心生成邻接矩阵
    graph = adj(unique_list, node_attr=center, dict_node=dict_col)

    G = nx.Graph(graph)  # 有向图
    # A = nx.to_numpy_array(G, weight='weight')

    A = similiar(G, weight_in='weight')
                # G = nx.Graph(graph)
                #
                # clustering = AffinityPropagation(damping= 0.8, #preference = -0.1,
                #                                  verbose=True, random_state=5).fit(A)

    # clu = clustering.labels_.tolist()
    # com_class.append(clu)
    # plt.figure(figsize=(21, 16))
    # plt.subplot(1,2,1)
    # nx.draw_networkx(G, pos, node_color=clu,
    #                  node_size=80,
    #                  cmap=plt.get_cmap('RdBu'),
    #                  linewidths=0.5,
    #                  arrowsize=5,
    #                  with_labels=False)  # 图结构可视化


    # plt.show()
    # A =np.random.rand(125,125)

    i=0
    for name in unique_list:
        start, end = name.split('-', 1)[0], name.split('-', 1)[1]
        b_name = '{}-{}'.format(end, start)
        Z[i] = (center[dict_col[name]]+center[dict_col[b_name]])/2
        Z_cx[i] = abs((center[dict_col[name]]-center[dict_col[b_name]]))
        i+=1
    Z = Z.astype(float16)
    Z_cx = Z_cx.astype(float16)
    # eval_on_center['class{}'.format(n)] = {'Z':pd.DataFrame(Z).describe(),
    #                                        'Z_cx':pd.DataFrame(Z_cx).describe()}
    # {'cx_range':[min(Z_cx),max(Z_cx)],
    #                                        'v_range':[min(Z),max(Z)],
    #                                        'cx_mean': [mean(Z_cx)],
    #                                        'v_mean': [mean(Z)],
    #                                        }
    result = Moran.moranI(A, Z, imgPath = subpath + '莫兰散点图{}.png'.format(n))
    I_moran.append(result['I']['value'])

    Zi = result['Ii']['value']

    maxZi = max(Zi)
    minZi = min(Zi)
    print(minZi)
    Zi[np.argmax(Zi)] =Zi[np.argmax(Zi)] * 0.8
    Zi[np.argmin(Zi)] = Zi[np.argmax(Zi)] * 0.2
    Zi[Zi > 0] = Zi[Zi > 0] / max(Zi)
    Zi[Zi < 0] = -0.8*Zi[Zi < 0] / min(Zi)



    # Zi = abs(Zi)
    # Zi = (Zi - min(Zi)) / (max(Zi) - min(Zi)).tolist()
    plt.figure(figsize=(14, 5))

    plt.subplot(1,2,1)
    nx.draw_networkx(G, pos, node_color=Zi,
                     node_size=80,
                     cmap=plt.get_cmap('RdBu'),
                     label='ratio[{:.2f}-{:.2f}-{:.2f}]'.
                     format(minZi, maxZi, mean(Zi)),
                     vmin= -1,
                     vmax= 1,
                     linewidths=0.5,
                     arrowsize=5,
                     with_labels=False)  # 图结构可视化

    # plt.legend()
    # plt.colorbar(a)

    Zmin =min(Z_list)
    Zmax = max(Z_list)
    maxZ = max(Z)
    minZ = min(Z)
    meanZ = mean(Z)
    medianZ = median(Z)
    Z = ((Z - 15) / (85 - 15))

    fig2=plt.subplot(1, 2, 2)
    nx.draw_networkx(G, pos, node_color=Z,
                     node_size=80,
                     cmap=plt.get_cmap('RdYlGn'),
                     with_labels=False,
                     linewidths=0.5,
                     arrowsize =5,
                     vmin= 0.0,
                     vmax= 1.0,
                     label = 'Speed[{:.2f}-{:.2f}]   mean:{:.2f}  median:{:.2f}'.
                     format(minZ, maxZ, meanZ, medianZ),
                     style = 'dashdot')  # 图结构可视化

    # plt.legend()

    Z_cxmax = max(Z_cx)
    Z_cxmin = min(Z_cx)

    #
    # fig3 = plt.subplot(2, 2, 4)
    # nx.draw_networkx(G, pos, node_color=Z_cx,
    #                  node_size=80,
    #                  cmap=plt.get_cmap('Reds'),
    #                  with_labels=False,
    #                  vmin=0,
    #                  vmax=50,
    #                  linewidths = 0.5,
    #                  arrowsize=5,
    #                  label='Speed_Dif[{:.2f}-{:.2f}]'.format(Z_cxmin, Z_cxmax),
    #                  style='dashdot')  # 图结构可视化
    #
    # # fig2.colorbar()
    # plt.legend()


    pyplot.savefig(subpath+"moran{}.png".format(n), dpi=512,
                   bbox_inches='tight')

    plt.close()

    #
# colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#38B0DE', '#b725ca']
# colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#CDCDCD', '#38B0DE']
# colors = ['#034b61', '#5d353e', '#E9C2A6', '#FF2400', '#38B0DE', '#CDCDCD']
# colors = ['#034b61', '#5d353e', '#FF2400', '#E9C2A6', '#b725ca', '#38B0DE']
#
# n = 1
# c = [colors[x] for x in com_class[n]]
# nx.draw_networkx(G, pos, node_color=c,
#                  node_size=80,
#                  linewidths=0.5,
#                  arrowsize=5,
#                  with_labels=False)
# pyplot.savefig(subpath + "{}.png".format(n), dpi=512,
#                bbox_inches='tight')
# plt.close()
