import math
import tsfresh
import logging
from scipy import signal
from statsmodels.distributions.empirical_distribution import ECDF
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


def approximate_entropy(x, m, r,h):
    """
    Implements a vectorized Approximate entropy algorithm.

        https://en.wikipedia.org/wiki/Approximate_entropy

    For short time-series this method is highly dependent on the parameters,
    but should be stable for N > 2000, see:

        Yentes et al. (2012) -
        *The Appropriate Use of Approximate Entropy and Sample Entropy with Short Data Sets*


    Other shortcomings and alternatives discussed in:

        Richman & Moorman (2000) -
        *Physiological time-series analysis using approximate entropy and sample entropy*

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: Length of compared run of data
    :type m: int
    :param r: Filtering level, must be positive
    :type r: float

    :return: Approximate entropy
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m + 1:
        return 0

    def _phi(m):
        x_re = np.array([x[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                          axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + h))



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
        # e = approximate_entropy(x[:, i], n_fowards,5,n_fowards)
        # multi_entropy[i] = binned_entropy(x[:, i], max_bins) * range_x[i] * x_max[i]
        multi_entropy[i] = math.log(1/abs(e))  * x_max[i]
        # * range_x[i] * x_max[i]
        # print(1/abs(e),math.log(1/abs(e)))

    return sum(multi_entropy) * 1/(max(x_max))  #1/Max(range(Fi) *Σ I(Fi)*range(Fi)
    # return sum(multi_entropy)


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




def get_cdf(n_lookback,n_fowards):
    data_path = 'F://MH//Data//experiment//data//test1//'
    result_path = 'F://MH//Data//experiment//result//'

    n_features = 14
    n_obs = n_lookback + n_fowards

    Tests = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            # print(filepath)

            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)

            sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
            Test = sample.values.reshape(-1, n_features)
            Tests.append(Test)

    Testdata = np.concatenate(Tests, axis=0)

    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards


    list_info = []

    for t in range(epoch):
        stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()
        # 计算时间序列信息熵
        entropy = information(stream, n_fowards)
        # 记录所有信息熵
        list_info.append(entropy)

    return ECDF(np.array(list_info))




#
if __name__ == "__main__":



    data_path = 'F://MH//Data//experiment//data//test1//'
    result_path = 'F://MH//Data//experiment//result//'

    n_features, n_lookback, n_fowards = 14, 60, 5
    n_obs = n_lookback + n_fowards

    Tests = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            print(filepath)

            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)


            sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
            Test = sample.values.reshape(-1, n_features)
            Tests.append(Test)


    Testdata=np.concatenate(Tests, axis=0)

    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards
    k_count = 0

    list_info1 = []

    for t in range(epoch):
        stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()

        # 计算时间序列信息熵
        entropy = information(stream, n_fowards)

        # if entropy <= 0.5:
        #     plt.plot(stream)
        #     plt.title(str(entropy))
        #     plt.show()

        # 记录所有信息熵
        list_info1.append(entropy)





    n_features, n_lookback, n_fowards = 14, 90, 10
    n_obs = n_lookback + n_fowards

    Tests = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            print(filepath)

            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)


            sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
            Test = sample.values.reshape(-1, n_features)
            Tests.append(Test)


    Testdata=np.concatenate(Tests, axis=0)

    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards
    k_count = 0

    list_info2 = []

    for t in range(epoch):
        stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()

        # 计算时间序列信息熵
        entropy = information(stream, n_fowards)
        # 记录所有信息熵
        list_info2.append(entropy)


    n_features, n_lookback, n_fowards = 14, 90, 15
    n_obs = n_lookback + n_fowards

    Tests = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            print(filepath)

            data = np.load(filepath, allow_pickle=True)
            data = data.astype(np.float32)


            sample = series_to_supervised(data, n_fowards + n_lookback - 1, 1)
            Test = sample.values.reshape(-1, n_features)
            Tests.append(Test)


    Testdata=np.concatenate(Tests, axis=0)

    if len(Testdata) % n_obs != 0:
        raise ValueError("Invalid length of Testdata")
    epoch = int(len(Testdata) / n_obs) - n_fowards
    k_count = 0

    list_info3 = []

    for t in range(epoch):
        stream = Testdata[t * n_obs:(t + 1) * n_obs].copy()

        # 计算时间序列信息熵
        entropy = information(stream, n_fowards)
        # if abs(entropy) > 4:
        #     plt.plot(stream)
        #     plt.title(str(entropy))
        #     plt.show()
        # 记录所有信息熵
        list_info3.append(entropy)
        # truevalue.append()


    # list_info.sort()
    plt.plot(list_info1)
    plt.plot(list_info2)
    plt.plot(list_info3)
    plt.show()
    data1 = np.array(list_info1)
    data2 = np.array(list_info2)
    data3 = np.array(list_info3)
    # distributions = [st.laplace, st.norm, st.expon,
    #                  st.f, st.t]


    range1 = int(max(data1)-min(data1))+1
    range2 = int(max(data2)-min(data2))+1
    range3 = int(max(data3)-min(data3))+1
    n=5


    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)

    plt.hist(data1,density= True, bins = range1*2,color='grey')
    plt.title('Histogram of score',fontsize=18)
    # plt.yticks([0,0.1,0.2])
    plt.tick_params(labelsize=15)


    plt.subplot(3, 2, 2)
    ecdf = ECDF(data1)

    x = [i/n for i in range((range1)*n)]
    point=ecdf(x)
    plt.hist(data1,density= True,cumulative=True,bins = 20, color='grey')
    plt.title('ECDF of score',fontsize=18)
    plt.tick_params(labelsize=15)
    plt.plot(x,point,linewidth=4,color='red')


    plt.subplot(3, 2, 3)
    plt.hist(data2,density= True, bins = range2*2, color='grey')
    plt.ylabel('statistical frequency',fontsize=18)
    plt.tick_params(labelsize=15)


    plt.subplot(3, 2, 4)
    ecdf = ECDF(data2)

    x = [i/n for i in range((range2+1)*n)]
    point=ecdf(x)
    plt.hist(data2,density= True,cumulative=True,bins = range2, color='grey')
    plt.ylabel('$\mathregular{F_n}$(s)',fontsize=18)
    plt.plot(x,point,linewidth=4, color='red')
    plt.tick_params(labelsize=15)


    plt.subplot(3, 2, 5)
    plt.hist(data3,density= True, bins=range3*2, color='grey')

    plt.xlabel('score',fontsize=18)
    plt.tick_params(labelsize=15)



    plt.subplot(3, 2, 6)
    ecdf = ECDF(data3)

    x = [i/n for i in range((range3+1)*n)]
    point=ecdf(x)
    plt.hist(data3,density= True,cumulative=True,bins = range3, color='grey')
    plt.plot(x,point,linewidth=4, color='red')
    plt.tick_params(labelsize=15)
    plt.xlabel('score',fontsize=18)


    # plt.savefig(result_path+'s.png', dpi=512, bbox_inches='tight')
    plt.show()





    plt.hist(data3,density= True,cumulative=True,histtype='step',bins = range1,color='grey')
    plt.plot(x,point,linewidth=4)
    plt.title('Histogram of ',fontsize=18)
    # plt.yticks([0,0.1,0.2])
    plt.tick_params(labelsize=15)
    plt.show()

    n = 5
    plt.figure(figsize=(11, 4.2))
    plt.subplot(1, 2, 1)

    plt.hist(data1, density=True, bins=range1 * 2, color='grey')
    # plt.title(r'Histogram', fontsize=18)
    # plt.yticks([0,0.1,0.2])
    plt.tick_params(labelsize=15)
    plt.xlabel(r'$\mathregular{\lambda_c}$', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)

    plt.subplot(1, 2, 2)
    ecdf = ECDF(data1)

    x = [i / n + 0.5 for i in range((range1) * n)]
    point = ecdf(x)
    plt.hist(data1, density=True, cumulative=True, bins=30, color='grey')
    # plt.title(r'ECDF of $\mathregular{\lambda_c}$', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.plot(x, point, linewidth=4, color='red')
    # plt.xlim([0,5,10,15,20,25])
    plt.text(6, 0.9, r'$F(\mathregular{\lambda_c})$', fontdict={'size': '18', 'color': 'r'})
    plt.xlabel(r'$\mathregular{\lambda_c}$', fontsize=18)
    plt.ylabel('Cumulative Frequencies', fontsize=18)
    # plt.savefig(result_path + 'lp=5.png', dpi=300, bbox_inches='tight')
    plt.show()


