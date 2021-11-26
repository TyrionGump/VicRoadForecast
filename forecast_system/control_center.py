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
from forecast_system.dataset import NN_Dataset
import pandas as pd
import numpy as np


class ControlCenter:
    def __init__(self):
        self.road_network = RoadNetwork()
        self.traffic_harvester = TrafficHarvester()
        self.data_aggregator = DataAggregator(road_network=self.road_network)

        self.raw_training_tt = None
        self.raw_testing_tt = None
        self.dataset = None

    def pull_traffic_data(self):
        research_links = []
        for region in args.research_config['region']['RESEARCH_REGIONS']:
            research_links.extend(self.road_network.get_regional_link_ids(region_name=region))
        pulling_link_ids = copy.deepcopy(research_links)

        if args.temporal_feature:
            link_neighbours = self.road_network.get_link_neighbours()

            for link_id in research_links:
                pulling_link_ids.extend(link_neighbours[link_id])
            pulling_link_ids = list(set(pulling_link_ids))

        self.raw_training_tt = self.traffic_harvester.get_df_dict(link_ids=pulling_link_ids,
                                                                  start_time=args.research_config['period']['TRAINING_START_TIME'],
                                                                  end_time=args.research_config['period']['TRAINING_END_TIME'])
        self.raw_training_tt = self.traffic_harvester.get_df_dict(link_ids=pulling_link_ids,
                                                                  start_time=args.research_config['period']['TESTING_START_TIME'],
                                                                  end_time=args.research_config['period']['TESTING_START_TIME'])

    def add_features(self):
        if args.temporal_feature:
            self.data_aggregator.add_temporal_features(self.raw_training_tt)
            self.data_aggregator.add_temporal_features(self.raw_testing_tt)

        if args.link_feature:
            self.data_aggregator.add_link_features(self.raw_training_tt, buffer_distances=[400, 800])
            self.data_aggregator.add_link_features(self.raw_testing_tt, buffer_distances=[400, 800])

    def create_dataset(self):
        self.dataset = NN_Dataset(training_data_dict=self.raw_training_tt, testing_data_dict=self.raw_testing_tt)
        self.dataset.aggregate_time_series(minute=args.research_config['window']['WINDOW_MIN'])
        # self.dataset.tt_to_speed()
        self.dataset.normalize()
        self.dataset.reshape_for_NN(forward_steps=args.research_config['window']['FORWARD_STEPS'],
                                backward_steps=args.research_config['window']['BACKWARD_STEPS'])

    def train(self):
        training_set = TorchDataset(np.load('data/processed_data/training_feature_6.npy'), np.load('data/processed_data/training_target_6.npy'))
        testing_set = TorchDataset(np.load('data/processed_data/testing_feature_6.npy'), np.load('data/processed_data/testing_target_6.npy'))
        # training_set = self.dataset.get_training_set()
        # testing_set = self.dataset.get_testing_set()
        if args.lstm:
            model = LSTM_COACH(training_set=training_set, testing_set=testing_set)
        elif args.rnn:
            model = RNN_COACH(training_set=training_set, testing_set=testing_set)
        else:
            model = Seq2Seq_COACH(training_set=training_set, testing_set=testing_set)
        model.run()





