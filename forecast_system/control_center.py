# -*- coding: utf-8 -*-
"""
@File:control_center.py
@Date: 2021/11/16
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""
import copy
from config import args
from forecast_system.road_network import RoadNetwork
from forecast_system.traffic_harvester import TrafficHarvester
from forecast_system.data_aggregator import DataAggregator
from forecast_system.nn_library.nn_utils import TorchDataset
from forecast_system.nn_library.run_coach.rnn_coach import RNN_COACH
from forecast_system.nn_library.run_coach.lstm_coach import LSTM_COACH
from forecast_system.nn_library.run_coach.seq2seq_coach import Seq2Seq_COACH
from forecast_system.sklearn_models.linear_regression import LR_COACH
from forecast_system.dataset import RNN_Dataset
from forecast_system.dataset import LR_Dataset
import pandas as pd
import numpy as np
import time
import os
from forecast_system.utils import save_raw_data
from forecast_system.utils import save_dataset


class ControlCenter:
    def __init__(self):
        self.road_network = RoadNetwork()
        self.traffic_harvester = TrafficHarvester()
        self.data_aggregator = DataAggregator(road_network=self.road_network)
        self.research_links = []  # Links ids which are predicted in the models

        # Period of the training set and the testing set
        self.training_start_time = args.research_config['period']['TRAINING_START_TIME']
        self.training_end_time = args.research_config['period']['TRAINING_END_TIME']
        self.testing_start_time = args.research_config['period']['TESTING_START_TIME']
        self.testing_end_time = args.research_config['period']['TESTING_END_TIME']

        # Duration of each time window (default = 0.5 min)
        self.window_minute = args.research_config['window']['WINDOW_MIN']

        # Steps to feed and predict
        self.forward_steps = args.research_config['window']['FORWARD_STEPS']
        self.backward_steps = args.research_config['window']['BACKWARD_STEPS']

        # Buffer Distance for POI density
        self.buffer_distances = args.research_config['poi']['BUFFER_DISTANCES']

        # Dictionary of raw travel time data for each link
        self.raw_training_tt = None
        self.raw_testing_tt = None

        # The processed dataset used to feed and test models
        self.dataset = None

    def pull_traffic_data(self):
        if args.research_config['region']['RESEACH_LINKS']:
            self.research_links = args.research_config['region']['RESEACH_LINKS']
        else:
            for region in args.research_config['region']['RESEARCH_REGIONS']:
                self.research_links.extend(self.road_network.get_regional_link_ids(region_name=region))

        pulling_link_ids = copy.deepcopy(self.research_links)

        if args.neighbours_feature:
            link_neighbours = self.road_network.get_link_neighbours()
            for link_id in self.research_links:
                pulling_link_ids.extend(link_neighbours[link_id])
            pulling_link_ids = list(set(pulling_link_ids))

        self.raw_training_tt = self.traffic_harvester.get_df_dict(link_ids=pulling_link_ids,
                                                                  start_time=self.training_start_time,
                                                                  end_time=self.training_end_time)

        self.raw_testing_tt = self.traffic_harvester.get_df_dict(link_ids=pulling_link_ids,
                                                                 start_time=self.testing_start_time,
                                                                 end_time=self.testing_end_time)

        if args.save:
            raw_training_path = 'data/traffic_data/subset_network/{}~{}'.format(self.training_start_time, self.training_end_time)
            raw_testing_path = 'data/traffic_data/subset_network/{}~{}'.format(self.testing_start_time, self.testing_end_time)
            save_raw_data(self.raw_training_tt, root_path=raw_training_path)
            save_raw_data(self.raw_testing_tt, root_path=raw_testing_path)

    def add_features(self):
        if args.temporal_feature:
            self.data_aggregator.add_temporal_features(self.raw_training_tt)
            self.data_aggregator.add_temporal_features(self.raw_testing_tt)

        if args.link_feature:
            self.data_aggregator.add_link_features(self.raw_training_tt, buffer_distances=self.buffer_distances)
            self.data_aggregator.add_link_features(self.raw_testing_tt, buffer_distances=self.buffer_distances)

    def create_dataset(self):
        if args.lstm or args.rnn or args.seq2seq:
            self.dataset = RNN_Dataset(training_data_dict=self.raw_training_tt, testing_data_dict=self.raw_testing_tt, separate=args.separate)
            self.dataset.aggregate_time_series(minute=self.window_minute)
            self.dataset.normalize()
            self.dataset.reshape(forward_steps=self.forward_steps, backward_steps=self.backward_steps)
        elif args.lr:
            self.dataset = LR_Dataset(training_data_dict=self.raw_training_tt, testing_data_dict=self.raw_testing_tt, separate=args.separate)
            self.dataset.aggregate_time_series(minute=self.window_minute)
            self.dataset.normalize()
            self.dataset.reshape(forward_steps=self.forward_steps, backward_steps=self.backward_steps)

        if args.save:
            save_dataset(self.dataset.get_training_set(), self.dataset.get_testing_set(), root_path='data/dataset')

    def train(self):
        training_set = self.dataset.get_training_set()
        testing_set = self.dataset.get_testing_set()
        normalize_encoder = self.dataset.get_normalize_encoder()
        if args.lstm:
            model = LSTM_COACH(training_set=training_set, testing_set=testing_set, normalize_encoder=normalize_encoder)
        elif args.rnn:
            model = RNN_COACH(training_set=training_set, testing_set=testing_set, normalize_encoder=normalize_encoder)
        elif args.seq2seq:
            model = Seq2Seq_COACH(training_set=training_set, testing_set=testing_set, normalize_encoder=normalize_encoder)
        elif args.lr:
            model = LR_COACH(training_set=training_set, testing_set=testing_set, normalize_encoder=normalize_encoder)
        model.run()







