# -*- coding: utf-8 -*-
"""
@File:seq2seq_coach.py
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
from forecast_system.nn_library.nn.seq2seq import Seq2SeqEncoder
from forecast_system.nn_library.nn.seq2seq import Seq2SeqDecoder
from forecast_system.nn_library.nn.seq2seq import EncoderDecoder
from forecast_system.nn_library import nn_utils
import torch.nn.functional as F
import logging


class Seq2Seq_COACH:
    def __init__(self, training_set: TorchDataset, testing_set: TorchDataset):
        self.training_set = training_set
        self.testing_set = testing_set
        self.params = OrderedDict(lr=args.nn_config['Seq2Seq']['lr'],
                                  epoch=args.nn_config['Seq2Seq']['EPOCH'],
                                  batch_size=args.nn_config['Seq2Seq']['BATCH_SIZE'],
                                  num_layers=args.nn_config['Seq2Seq']['NUM_LAYERS'],
                                  hidden_size=args.nn_config['Seq2Seq']['HIDDEN_SIZE'],
                                  device=args.nn_config['Seq2Seq']['DEVICE'])
        self.run_builder = nn_utils.RunBuilder()
        self.run_manager = nn_utils.RunManager()

    def run(self):
        for run in self.run_builder.get_runs(params=self.params):
            training_loader = DataLoader(self.training_set, batch_size=run.batch_size, shuffle=True)
            testing_loader = DataLoader(self.testing_set, batch_size=run.batch_size)

            encoder = Seq2SeqEncoder(feature_size=self.training_set.feature_size,
                                     hidden_size=run.hidden_size,
                                     num_layers=run.num_layers).to(run.device)
            decoder = Seq2SeqDecoder(feature_size=self.training_set.feature_size,
                                     target_size=self.training_set.target_size,
                                     hidden_size=run.hidden_size,
                                     num_layers=run.num_layers).to(run.device)

            network = EncoderDecoder(encoder=encoder, decoder=decoder)
            network.apply(nn_utils.xavier_init_weights)
            network.to(run.device)

            optimizer = optim.Adam(network.parameters(), lr=run.lr)

            self.run_manager.begin_run(run=run, network=network, loader=training_loader)
            for epoch in range(run.epoch):
                self.run_manager.begin_epoch()
                for batch in training_loader:
                    enc_input = batch[0].to(run.device)
                    dec_begin = torch.zeros(batch[1].shape[0], 1, 1).to(run.device)
                    target = batch[1].view(batch[1].shape[0], batch[1].shape[1], 1).to(run.device)
                    dec_input = torch.cat([dec_begin, target[:, :-1, :]], dim=1).to(run.device)

                    y_hat, _ = network(enc_input, dec_input)
                    loss = F.mse_loss(y_hat, target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn_utils.grad_clipping(network, theta=1)
                    optimizer.step()
                    self.run_manager.track_loss(y_hat, target)
                self.run_manager.end_epoch()
            self.run_manager.end_run()

            network.eval()
            test_mse = 0
            test_mae = 0
            test_mape = 0
            with torch.no_grad():
                for batch in testing_loader:
                    enc_input = batch[0].to(run.device)
                    enc_output = network.encoder(enc_input)
                    dec_state = network.decoder.init_state(enc_output)
                    dec_begin = torch.zeros(batch[1].shape[0], 1, 1).to(run.device)

                    pred_sequence = []
                    for _ in range(args.research_config['window']['BACKWARD_STEPS']):
                        y_hat, dec_state = network.decoder(dec_begin, dec_state)
                        dec_begin = y_hat
                        pred_y = y_hat.detach().clone().to('cpu')
                        pred_sequence.append(pred_y)
                    pred_sequence = torch.cat(pred_sequence, dim=1).squeeze(dim=2)
                    # print('=======================================================')
                    # print(pred_sequence)
                    # print('*******************************')
                    # print(batch[1])
                    # print('=======================================================')
                    test_mse += F.mse_loss(pred_sequence, batch[1]).item() * batch[1].shape[0]
                    test_mae += F.l1_loss(pred_sequence, batch[1]).item() * batch[1].shape[0]
                    test_mape += torch.mean(F.l1_loss(pred_sequence, batch[1], reduction='none') / batch[1]).item() * batch[1].shape[0]
                print(test_mse / len(testing_loader.dataset))
                print(test_mae / len(testing_loader.dataset))
                print(test_mape / len(testing_loader.dataset))
        self.run_manager.save()





