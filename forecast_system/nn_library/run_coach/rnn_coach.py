# -*- coding: utf-8 -*-
"""
@File:rnn_coach.py
@Date: 2021/11/17
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from torch.utils.data import DataLoader
from forecast_system.nn_library.nn_utils import TorchDataset
import torch.optim as optim
from config import args
from collections import OrderedDict
from forecast_system.nn_library.nn.rnn import RNN
from forecast_system.nn_library import nn_utils
import torch.nn.functional as F


class RNN_COACH:
    def __init__(self, training_set: TorchDataset, testing_set: TorchDataset):
        self.training_set = training_set
        self.testing_set = testing_set
        self.params = OrderedDict(lr=args.nn_config['RNN']['lr'],
                                  epoch=args.nn_config['RNN']['EPOCH'],
                                  batch_size=args.nn_config['RNN']['BATCH_SIZE'],
                                  num_layers=args.nn_config['RNN']['NUM_LAYERS'],
                                  hidden_size=args.nn_config['RNN']['HIDDEN_SIZE'],
                                  device=args.nn_config['RNN']['DEVICE'])
        self.run_builder = nn_utils.RunBuilder()
        self.run_manager = nn_utils.RunManager()

    def run(self):
        for run in self.run_builder.get_runs(params=self.params):
            training_loader = DataLoader(self.training_set, batch_size=run.batch_size, shuffle=True)
            testing_loader = DataLoader(self.testing_set)
            network = RNN(feature_size=self.training_set.feature_size, target_size=self.training_set.target_size, hidden_size=run.hidden_size, num_layers=run.num_layers).to(run.device)
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
                    optimizer.step()
                    self.run_manager.track_loss(y_hat, target)
                self.run_manager.end_epoch()
            self.run_manager.end_run()
        self.run_manager.save()







