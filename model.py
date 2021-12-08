import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
torch.__version__


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, h_size, n_layers):
        super(RNN, self).__init__()

        self.num_layers = n_layers
        self.hidden_size = h_size
        self.lstm= nn.LSTM(

            input_size=input_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.out = nn.Linear(h_size, 3)  #torch.nn.Linear（
                              # in_features：int，out_features：int，bias：bool = True ）


    def forward(self, x, h_state=None, c_state=None):
        # Set initial hidden and cell states

        # h_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if h_state.size(1) != x.size(0):
            h_state = h_state[:,[0],:]
            c_state = c_state[:,[0],:]

        r_out, (h_STATE, c_STATE) = self.lstm(x, (h_state, c_state))

        # outs = []  # 保存所有的预测值
        # for time_step in range(r_out.size(1)):  # 计算每一步长的预测值
        #    outs.append(self.out(r_out[:, time_step, :]))

        # outs = self.out(r_out[:,-1,:])
        outs = self.out(r_out)
        # outs = self.out(r_out.mean(axis=1, keepdim = True))
        # outs.squeeze()
        outs = outs[:,-1,:].view(r_out.size(0),-1)

        return outs, h_STATE, c_STATE



