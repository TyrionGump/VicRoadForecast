# -*- coding: utf-8 -*-
"""
@File:nn_utils.py
@Date: 2021/11/17
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import time
import itertools
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
from collections import namedtuple
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F
from config import args
import logging
from torch import nn

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('max_rows', 100)


class TorchDataset(Dataset):
    def __init__(self, feature_set, target_set):
        self.features = torch.from_numpy(feature_set).float()
        self.targets = torch.from_numpy(target_set).float()
        self.feature_size = feature_set.shape[-1]
        self.target_size = target_set.shape[-1]

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager:
    def __init__(self):
        self.epoch_count = 0

        self.epoch_mse = None
        self.epoch_mape = None
        self.epoch_mae = None

        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = 0

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        # self.tb = SummaryWriter(log_dir='run_log', comment=f'-{run}')
        # features, target = next(iter(self.loader))
        # self.tb.add_graph(self.network, features.to(getattr(run, 'device', 'cpu')))

    def end_run(self):
        # self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1

        backward_steps = args.research_config['window']['BACKWARD_STEPS']
        self.epoch_mse = torch.zeros(backward_steps).to(self.run_params.device)
        self.epoch_mape = torch.zeros(backward_steps).to(self.run_params.device)
        self.epoch_mae = torch.zeros(backward_steps).to(self.run_params.device)

    def end_epoch(self):
        self.epoch_mse = self.epoch_mse.to('cpu').numpy()
        self.epoch_mape = self.epoch_mape.to('cpu').numpy()
        self.epoch_mae = self.epoch_mae.to('cpu').numpy()

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        mse_loss = (self.epoch_mse / len(self.loader.dataset)).tolist()
        mae_loss = (self.epoch_mae / len(self.loader.dataset)).tolist()
        mape_loss = (self.epoch_mape / len(self.loader.dataset)).tolist()

        mean_mse = np.mean(mse_loss)
        mean_mae = np.mean(mae_loss)
        mean_mape = np.mean(mape_loss)

        # self.tb.add_scalar('MSE', mean_mse, self.epoch_count)
        # self.tb.add_scalar('MAP', mean_mae, self.epoch_count)
        # self.tb.add_scalar('MAPE', mean_mape, self.epoch_count)
        #
        # for name, param in self.network.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_count)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['Mean MSE'] = mean_mse
        results['Mean MAE'] = mean_mae
        results['MEAN MAPE'] = mean_mape
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        results['MSEs'] = mse_loss
        results['MAEs'] = mae_loss
        results['MAPEs'] = mape_loss
        if self.epoch_count % 10 == 0:
            logging.info('epoch: {} MSE: {} MAE: {} MAPE: {}'.format(self.epoch_count, mean_mse, mean_mae, mean_mape))
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)


    @torch.no_grad()
    def track_loss(self, pred, target):
        self.epoch_mse += torch.mean(F.mse_loss(pred, target, reduction='none'), dim=0) * self.loader.batch_size
        self.epoch_mae += torch.mean(F.l1_loss(pred, target, reduction='none'), dim=0) * self.loader.batch_size
        self.epoch_mape += torch.mean(F.l1_loss(pred, target, reduction='none') / target, dim=0) * self.loader.batch_size

    def get_run_log(self):
        return pd.DataFrame.from_dict(self.run_data, orient='columns')



def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

