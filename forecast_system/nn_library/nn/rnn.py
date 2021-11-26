# -*- coding: utf-8 -*-
"""
@File:link_gdf.geojson-PyCharm-rnn.py
@Date: 14/10/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from torch import nn
import torch
from config import settings


class RNN(nn.Module):
    def __init__(self, feature_size, target_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.feature_size = feature_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._init_net_layers()

    def _init_net_layers(self):
        self.rnn = nn.RNN(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.dense = nn.Linear(in_features=self.hidden_size, out_features=self.target_size)

    def forward(self, x):
        # out: [batch_size, sequence_num, hidden_size]
        # hidden_prev: [layers_num, batch_size, hidden_size]
        output, hidden_prev = self.rnn(x)

        out = self.dense(hidden_prev[-1])
        # print(out.shape)
        return out

    def begin_state(self, batch_size, device='cpu'):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)