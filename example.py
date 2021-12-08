import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
torch.__version__





data1 = np.load('data.npy')
data2 = np.load('data1.npy')
data3 = np.load('data2.npy')
value1 = []
value2 = []
value3 = []

for i in range(100):

    if i > 20:
        if i % 10 == 0:
            value1.append(data1[i])
            value2.append(data2[i])
            value3.append(data3[i])
    else:
        value1.append(data1[i])
        value2.append(data2[i])
        value3.append(data3[i])


# plt.xticks([1,10,20,30],['0','20','50','100'])
plt.plot(value1)
plt.plot(value2)
plt.plot(value3)
plt.show()