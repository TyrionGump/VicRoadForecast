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
import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error


class Seq2Seq_COACH:
    def __init__(self, training_set: dict, testing_set: dict, normalize_encoder):
        self.link_ids = training_set['feature'].keys()
        self.training_set = training_set
        self.testing_set = testing_set
        self.normalize_encoder = normalize_encoder

        self.params = OrderedDict(lr=args.nn_config['Seq2Seq']['lr'],
                                  epoch=args.nn_config['Seq2Seq']['EPOCH'],
                                  batch_size=args.nn_config['Seq2Seq']['BATCH_SIZE'],
                                  num_layers=args.nn_config['Seq2Seq']['NUM_LAYERS'],
                                  hidden_size=args.nn_config['Seq2Seq']['HIDDEN_SIZE'],
                                  device=args.nn_config['Seq2Seq']['DEVICE'])
        self.run_builder = nn_utils.RunBuilder()
        self.run_manager = nn_utils.RunManager()

    def run(self):
        if args.separate:
            os.makedirs('result/seq2seq/separate/net_params', exist_ok=True)
            os.makedirs('result/seq2seq/separate/prediction_values', exist_ok=True)
            os.makedirs('result/seq2seq/separate/true_values', exist_ok=True)
            os.makedirs('result/seq2seq/separate/performance_results', exist_ok=True)
            os.makedirs('result/seq2seq/separate/training_log', exist_ok=True)

            for link_id in self.link_ids:
                logging.info('Running Seq2Seq for link {}'.format(link_id))
                training_torch_set = TorchDataset(self.training_set['feature'][link_id],
                                                  self.training_set['target'][link_id])
                model_params = self.train(training_torch_set)
                torch.save(model_params, 'result/seq2seq/separate/net_params/{}.pth'.format(link_id))
                run_log = self.run_manager.get_run_log()

                testing_torch_set = TorchDataset(self.testing_set['feature'][link_id],
                                                 self.testing_set['target'][link_id])
                scale_data_max = self.normalize_encoder[link_id].data_max_[0]
                scale_data_min = self.normalize_encoder[link_id].data_min_[0]
                y_pred = self.pred(testing_torch_set, model_params) * (scale_data_max - scale_data_min) + scale_data_min
                y_true = self.testing_set['target'][link_id] * (scale_data_max - scale_data_min) + scale_data_min
                performance = self.eval(y_true, y_pred)

                y_pred = pd.DataFrame(y_pred)
                y_true = pd.DataFrame(y_true)
                y_pred.to_csv('result/seq2seq/separate/prediction_values/{}.csv'.format(link_id), index=False)
                y_true.to_csv('result/seq2seq/separate/true_values/{}.csv'.format(link_id), index=False)
                performance.to_csv('result/seq2seq/separate/performance_results/{}.csv'.format(link_id), index=False)
                run_log.to_csv('result/seq2seq/separate/training_log/{}.csv'.format(link_id), index=False)

    def train(self, dataset: TorchDataset):
        optimal_mse = float('inf')
        optimal_model_params = {}

        for run in self.run_builder.get_runs(params=self.params):
            training_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=True)

            encoder = Seq2SeqEncoder(feature_size=dataset.feature_size,
                                     hidden_size=run.hidden_size,
                                     num_layers=run.num_layers).to(run.device)
            decoder = Seq2SeqDecoder(feature_size=dataset.feature_size,
                                     target_size=dataset.target_size,
                                     hidden_size=run.hidden_size,
                                     num_layers=run.num_layers).to(run.device)

            network = EncoderDecoder(encoder=encoder, decoder=decoder).to(run.device)
            network.apply(nn_utils.xavier_init_weights)
            optimizer = optim.Adam(network.parameters(), lr=run.lr)

            network.train()
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
                    self.run_manager.track_loss(y_hat.detach().squeeze(dim=2), target.detach().squeeze(dim=2))
                self.run_manager.end_epoch()
                if self.run_manager.run_data[-1]['Mean MSE'] < optimal_mse:
                    optimal_mse = self.run_manager.run_data[-1]['Mean MSE']
                    optimal_model_params = {'feature_size': dataset.feature_size, 'target_size': dataset.target_size,
                                            'hidden_size': run.hidden_size, 'num_layers': run.num_layers,
                                            'encoder_params': copy.deepcopy(encoder.state_dict()),
                                            'decoder_params': copy.deepcopy(decoder.state_dict())}
            self.run_manager.end_run()
        return optimal_model_params

    @torch.no_grad()
    def pred(self, dataset: TorchDataset, model_params: dict):
        encoder = Seq2SeqEncoder(feature_size=model_params['feature_size'],
                                 hidden_size=model_params['hidden_size'],
                                 num_layers=model_params['num_layers'])
        decoder = Seq2SeqDecoder(feature_size=model_params['feature_size'],
                                 target_size=model_params['target_size'],
                                 hidden_size=model_params['hidden_size'],
                                 num_layers=model_params['num_layers'])
        encoder.load_state_dict(model_params['encoder_params'])
        decoder.load_state_dict(model_params['decoder_params'])
        network = EncoderDecoder(encoder=encoder, decoder=decoder)

        testing_loader = DataLoader(dataset, batch_size=16)
        y_pred = []

        network.eval()
        for batch in testing_loader:
            enc_input = batch[0]
            enc_output = network.encoder(enc_input)
            dec_state = network.decoder.init_state(enc_output)
            dec_begin = torch.zeros(batch[1].shape[0], 1, 1)

            pred_sequence = []
            for _ in range(args.research_config['window']['BACKWARD_STEPS']):
                y_hat, dec_state = network.decoder(dec_begin, dec_state)
                dec_begin = y_hat
                pred_y = y_hat.detach().clone().to('cpu')
                pred_sequence.append(pred_y)
            y_pred.append(torch.cat(pred_sequence, dim=1).squeeze(dim=2).numpy())

        y_pred = np.array(y_pred)

        return y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])

    @staticmethod
    def eval(y_true, y_pred):
        performance_df = pd.DataFrame()
        performance_df['Step'] = ['t+{}'.format(i + 1) for i in range(y_pred.shape[1])]
        performance_df['MSE'] = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        performance_df['MAE'] = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        performance_df['MAPE'] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        return performance_df




