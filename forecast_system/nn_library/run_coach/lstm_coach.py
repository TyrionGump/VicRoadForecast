# -*- coding: utf-8 -*-
"""
@File:lstm_coach.py
@Date: 2021/11/22
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import torch
from torch.utils.data import DataLoader
from forecast_system.nn_library.nn_utils import TorchDataset
import torch.optim as optim
from config import args
from collections import OrderedDict
from forecast_system.nn_library.nn.lstm import LSTM
from forecast_system.nn_library import nn_utils
import torch.nn.functional as F
import logging


class LSTM_COACH:
    def __init__(self, training_set: TorchDataset, testing_set: TorchDataset):
        self.training_set = training_set
        self.testing_set = testing_set
        self.params = OrderedDict(lr=args.nn_config['LSTM']['lr'],
                                  epoch=args.nn_config['LSTM']['EPOCH'], batch_size=args.nn_config['LSTM']['BATCH_SIZE'],
                                  num_layers=args.nn_config['LSTM']['NUM_LAYERS'], hidden_size=args.nn_config['LSTM']['HIDDEN_SIZE'],
                                  device=args.nn_config['LSTM']['DEVICE'])
        self.run_builder = nn_utils.RunBuilder()
        self.run_manager = nn_utils.RunManager()

    def run(self):
        for run in self.run_builder.get_runs(params=self.params):
            training_loader = DataLoader(self.training_set, batch_size=run.batch_size, shuffle=True)
            testing_loader = DataLoader(self.testing_set)
            network = LSTM(feature_size=self.training_set.feature_size, target_size=self.training_set.target_size, hidden_size=run.hidden_size, num_layers=run.num_layers).to(run.device)
            optimizer = optim.Adam(network.parameters(), lr=run.lr)

            self.run_manager.begin_run(run=run, network=network, loader=training_loader)
            for epoch in range(run.epoch):
                self.run_manager.begin_epoch()
                for batch in training_loader:
                    feature = batch[0].to(run.device)
                    target = batch[1].to(run.device)

                    y_hat = network(feature)
                    loss = F.mse_loss(y_hat, target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn_utils.grad_clipping(network, theta=1)
                    optimizer.step()
                    self.run_manager.track_loss(y_hat, target)
                self.run_manager.end_epoch()
            self.run_manager.end_run()
        self.run_manager.save()





        loss_track = 0
        for batch in testing_loader:
            feature = batch[0].to(run.device)
            target = batch[1].to(run.device)
            y_hat= network(feature)
            loss = torch.mean(torch.abs((target - y_hat) / target))
            loss_track += loss.item()
            print(y_hat)
            print(target)
            print('********************')
        loss_track /= len(testing_loader)
        print(len(testing_loader))
        print("iteration: testing loss {}".format(loss_track))