# -*- coding: utf-8 -*-
"""
@File:lstm.py
@Date: 2021/11/22
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from torch import nn
import torch
from config import settings


class LSTM(nn.Module):
    def __init__(self, feature_size, target_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.feature_size = feature_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._init_net_layers()

    def _init_net_layers(self):
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=0.1)
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p, mean=0.0, std=0.001)
        self.dense = nn.Linear(in_features=self.hidden_size, out_features=self.target_size)

    def forward(self, x):
        # out: [batch_size, sequence_num, hidden_size]
        # hidden_prev: [layers_num, batch_size, hidden_size]
        output, (hidden_prev, cell) = self.lstm(x)

        out = self.dense(hidden_prev[-1])
        # print(out.shape)
        return out

    def begin_state(self, batch_size, device='cpu'):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)