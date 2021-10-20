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


class RNN(nn.Module):
    def __init__(self, feature_size, target_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(in_features=hidden_size * num_layers, out_features=target_size)

    def forward(self, x, hidden_prev):
        # out: [batch_size, sequence_num, hidden_size]
        # hidden_prev: [layers_num, batch_size, hidden_size]
        output, hidden_prev = self.rnn(x, hidden_prev)
        # print('---------------')
        # print(output.shape)
        # print(hidden_prev.shape)

        # hidden: [batch_size, hidden_size * layers_num)
        hidden_prev_ls = [hidden_prev[-i] for i in range(1, len(hidden_prev) + 1)]
        hidden = torch.cat(hidden_prev_ls, dim=1)
        # print(hidden.shape)
        out = self.linear(hidden)
        # print(out.shape)
        return out, hidden_prev
