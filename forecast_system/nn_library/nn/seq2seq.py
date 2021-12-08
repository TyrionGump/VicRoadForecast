# -*- coding: utf-8 -*-
"""
@File:seq2seq.py
@Date: 2021/11/22
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from torch import nn
import torch


# class Seq2SeqEncoder(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers):
#         super(Seq2SeqEncoder, self).__init__()
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self._init_net_layers()
#
#     def _init_net_layers(self):
#         self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.5)
#
#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         output, state = self.lstm(x)
#         return output, state
#
#
# class Seq2SeqDecoder(nn.Module):
#     def __init__(self, feature_size, target_size, hidden_size, num_layers):
#         super(Seq2SeqDecoder, self).__init__()
#         self.feature_size = feature_size
#         self.target_size = target_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self._init_net_layers()
#
#     def _init_net_layers(self):
#         self.lstm = nn.LSTM(input_size=1 + self.hidden_size, hidden_size=self.hidden_size,
#                           num_layers=self.num_layers, dropout=0.5)
#         self.dense = nn.Linear(in_features=self.hidden_size, out_features=1)
#
#
#     def init_state(self, enc_outputs):
#         return enc_outputs[1]
#
#     def forward(self, x, state):
#         x = x.permute(1, 0, 2)
#         context = state[0][-1].repeat(x.shape[0], 1, 1)
#         X_and_context = torch.cat((x, context), 2)
#         output, state = self.lstm(X_and_context, state)
#         output = self.dense(output).permute(1, 0, 2)
#         return output, state
#
#
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, enc_X, dec_X, *args):
#         enc_outputs = self.encoder(enc_X, *args)
#         dec_state = self.decoder.init_state(enc_outputs, *args)
#         return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(Seq2SeqEncoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._init_net_layers()

    def _init_net_layers(self):
        self.gru = nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, state = self.gru(x)
        return output, state


class Seq2SeqDecoder(nn.Module):
    def __init__(self, feature_size, target_size, hidden_size, num_layers):
        super(Seq2SeqDecoder, self).__init__()
        self.feature_size = feature_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._init_net_layers()

    def _init_net_layers(self):
        self.gru = nn.GRU(input_size=1 + self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers)
        self.dense = nn.Linear(in_features=self.hidden_size, out_features=1)


    def init_state(self, enc_outputs):
        return enc_outputs[1]

    def forward(self, x, state):
        x = x.permute(1, 0, 2)

        context = state[-1].repeat(x.shape[0], 1, 1)
        X_and_context = torch.cat((x, context), 2)
        output, state = self.gru(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)




