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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from forecast_system.nn_library.nn_utils import TorchDataset
import copy
import os


class LSTM_COACH:
    def __init__(self, training_set: dict, testing_set: dict, normalize_encoder):
        self.link_ids = training_set['feature'].keys()
        self.training_set = training_set
        self.testing_set = testing_set
        self.normalize_encoder = normalize_encoder
        self.params = OrderedDict(lr=args.nn_config['LSTM']['lr'],
                                  epoch=args.nn_config['LSTM']['EPOCH'], batch_size=args.nn_config['LSTM']['BATCH_SIZE'],
                                  num_layers=args.nn_config['LSTM']['NUM_LAYERS'], hidden_size=args.nn_config['LSTM']['HIDDEN_SIZE'],
                                  device=args.nn_config['LSTM']['DEVICE'])
        self.run_builder = nn_utils.RunBuilder()
        self.run_manager = nn_utils.RunManager()

    def train(self, dataset: TorchDataset):
        optimal_mse = float('inf')
        optimal_model_params = {}

        for run in self.run_builder.get_runs(params=self.params):
            logging.info(run)
            training_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=True)
            network = LSTM(feature_size=dataset.feature_size, target_size=dataset.target_size, hidden_size=run.hidden_size, num_layers=run.num_layers).to(run.device)
            optimizer = optim.Adam(network.parameters(), lr=run.lr)
            network.apply(nn_utils.xavier_init_weights)

            self.run_manager.begin_run(run=run, network=network, loader=training_loader)
            network.train()
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
                if self.run_manager.run_data[-1]['Mean MSE'] < optimal_mse:
                    optimal_mse = self.run_manager.run_data[-1]['Mean MSE']
                    optimal_model_params = {'feature_size': dataset.feature_size, 'target_size': dataset.target_size,
                                            'hidden_size': run.hidden_size, 'num_layers': run.num_layers,
                                            'net_params': copy.deepcopy(network.state_dict())}
            self.run_manager.end_run()
        return optimal_model_params

    @torch.no_grad()
    def pred(self, dataset: TorchDataset, model_params: dict):
        network = LSTM(feature_size=model_params['feature_size'], target_size=model_params['target_size'],
                     hidden_size=model_params['hidden_size'], num_layers=model_params['num_layers'])
        network.load_state_dict(model_params['net_params'])
        testing_loader = DataLoader(dataset, batch_size=16)
        y_pred = []

        network.eval()
        for batch in testing_loader:
            feature = batch[0]
            y_hat = network(feature)
            y_pred.append(y_hat.numpy())
        y_pred = np.array(y_pred)

        return y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])

    @staticmethod
    def eval(y_true, y_pred):
        performance_df = pd.DataFrame()
        performance_df['Step'] = ['t+{}'.format(i+1) for i in range(y_pred.shape[1])]
        performance_df['MSE'] = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        performance_df['MAE'] = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        performance_df['MAPE'] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        return performance_df

    def run(self):
        if args.separate:
            os.makedirs('result/lstm/separate/net_params', exist_ok=True)
            os.makedirs('result/lstm/separate/prediction_values', exist_ok=True)
            os.makedirs('result/lstm/separate/true_values', exist_ok=True)
            os.makedirs('result/lstm/separate/performance_results', exist_ok=True)
            os.makedirs('result/lstm/separate/training_log', exist_ok=True)

            for link_id in self.link_ids:
                logging.info('Running LSTM for link {}'.format(link_id))
                training_torch_set = TorchDataset(self.training_set['feature'][link_id], self.training_set['target'][link_id])
                model_params = self.train(training_torch_set)
                torch.save(model_params, 'result/lstm/separate/net_params/{}.pth'.format(link_id))
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
                y_pred.to_csv('result/lstm/separate/prediction_values/{}.csv'.format(link_id), index=False)
                y_true.to_csv('result/lstm/separate/true_values/{}.csv'.format(link_id), index=False)
                performance.to_csv('result/lstm/separate/performance_results/{}.csv'.format(link_id), index=False)
                run_log.to_csv('result/lstm/separate/training_log/{}.csv'.format(link_id), index=False)

        else:
            os.makedirs('result/lstm/global/net_params', exist_ok=True)
            os.makedirs('result/lstm/global/prediction_values', exist_ok=True)
            os.makedirs('result/lstm/global/true_values', exist_ok=True)
            os.makedirs('result/lstm/global/performance_results', exist_ok=True)
            os.makedirs('result/lstm/global/training_log', exist_ok=True)

            training_feature_all = [self.training_set['feature'][link_id] for link_id in self.link_ids]
            training_target_all = [self.training_set['target'][link_id] for link_id in self.link_ids]
            training_torch_set = TorchDataset(np.concatenate(training_feature_all, axis=0),
                                              np.concatenate(training_target_all, axis=0))

            model_params = self.train(training_torch_set)
            torch.save(model_params, 'result/lstm/global/net_params/global.pth')
            run_log = self.run_manager.get_run_log()

            for link_id in self.link_ids:
                testing_torch_set = TorchDataset(self.testing_set['feature'][link_id], self.testing_set['target'][link_id])
                scale_data_max = self.normalize_encoder.data_max_[0]
                scale_data_min = self.normalize_encoder.data_min_[0]
                y_pred = self.pred(testing_torch_set, model_params) * (scale_data_max - scale_data_min) + scale_data_min
                y_true = self.testing_set['target'][link_id] * (scale_data_max - scale_data_min) + scale_data_min
                performance = self.eval(y_true, y_pred)

                y_pred = pd.DataFrame(y_pred)
                y_true = pd.DataFrame(y_true)
                y_pred.to_csv('result/lstm/global/prediction_values/{}.csv'.format(link_id), index=False)
                y_true.to_csv('result/lstm/global/true_values/{}.csv'.format(link_id), index=False)
                performance.to_csv('result/lstm/global/performance_results/{}.csv'.format(link_id), index=False)

            run_log.to_csv('result/lstm/global/training_log/global.csv', index=False)



