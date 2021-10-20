# -*- coding: utf-8 -*-
"""
@File:VicRoadForecast-PyCharm-model_builder.py
@Date: 7/10/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from data_harvester import DataHarvester
from road_network import RoadNetwork
from data_processor import DataProcessor
from forecaster import SeparateModel
from evaluator import Evaluator
from datetime import datetime
import pytz
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import copy
from tqdm import tqdm
import time
import logging

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class ModelBuilder:
    def __init__(self, data_root_path, link_geo_path, lga_geo_path, poi_geo_path, link_gdf_path,
                 ca_cert, client_cert, client_key, query_url, table_name,
                 training_start_time, training_end_time, testing_start_time, testing_end_time, region):

        self.road_network = RoadNetwork(link_geo_path=link_geo_path, lga_geo_path=lga_geo_path,
                                        poi_geo_path=poi_geo_path, processed_link_path=link_gdf_path)
        self.data_harvester = DataHarvester(ca_cert=ca_cert, client_crt=client_cert, client_key=client_key,
                                            query_url=query_url, table_name=table_name)
        self.training_data_processor = None
        self.testing_data_processor = None
        self.training_start_time = training_start_time
        self.training_end_time = training_end_time
        self.testing_start_time = testing_start_time
        self.testing_end_time = testing_end_time
        self.region_name = region
        self.regional_link_ids = self.road_network.get_regional_link_ids(region_name=region)

        self.training_data_path = os.path.join(data_root_path, region,
                                               '{}-{}'.format(training_start_time.strftime('%m%d%H%M%S'),
                                                              training_end_time.strftime('%m%d%H%M%S')))
        self.testing_data_path = os.path.join(data_root_path, region,
                                              '{}-{}'.format(testing_start_time.strftime('%m%d%H%M%S'),
                                                             testing_end_time.strftime('%m%d%H%M%S')))

        self.training_raw_data_dict = None
        self.testing_raw_data_dict = None
        self.training_feature_dict = None
        self.training_target_dict = None
        self.testing_feature_dict = None
        self.testing_target_dict = None

        self.model = None
        self.evaluator = None

        self.logger = None
        self._set_logger()
        self.logger.info('Building models for {}. '
                         'The training period is between {} and {}. '
                         'The testing period is between {} and {}.'.format(self.region_name,
                                                                           self.training_start_time, self.training_end_time,
                                                                           self.testing_start_time, self.testing_end_time))

    def request_raw_data(self, save=False):
        self.logger.info('Requesting raw link data...')
        self.training_raw_data_dict = self._load_raw_data(start_time=self.training_start_time,
                                                          end_time=self.training_end_time,
                                                          file_path=self.training_data_path,
                                                          save=save)
        self.testing_raw_data_dict = self._load_raw_data(start_time=self.testing_start_time,
                                                         end_time=self.testing_end_time,
                                                         file_path=self.testing_data_path,
                                                         save=save)

    def data_preprocessing(self, forward_steps=10, backward_steps=10, minute_interval=0.5, save=False):
        self.logger.info('Preprocessing raw link data...')
        if self.training_raw_data_dict is None or self.testing_raw_data_dict is None:
            self.request_raw_data(save)

        self.training_data_processor = DataProcessor(link_df_dict=self.training_raw_data_dict,
                                                     road_network=self.road_network,
                                                     region_name=self.region_name)
        self.testing_data_processor = DataProcessor(link_df_dict=self.testing_raw_data_dict,
                                                    road_network=self.road_network,
                                                    region_name=self.region_name)

        processors = [self.training_data_processor, self.testing_data_processor]

        for processor in processors:
            if minute_interval != 0.5:
                processor.aggregate_time_series(minute=minute_interval)
            processor.rolling_time_series(forward_steps=forward_steps, backward_steps=backward_steps)
            # processor.add_neighbours_data()
            #
            # processor.add_hour_of_day()
            # processor.add_day_of_week()
            # processor.add_link_features()
            processor.drop_id_timestamp()
            processor.filter_regional_data()

        self.training_feature_dict, self.training_target_dict = self.training_data_processor.get_dataset()
        self.testing_feature_dict, self.testing_target_dict = self.testing_data_processor.get_dataset()

        if save:
            os.makedirs(os.path.join(self.training_data_path, 'feature'), exist_ok=True)
            os.makedirs(os.path.join(self.training_data_path, 'target'), exist_ok=True)
            os.makedirs(os.path.join(self.testing_data_path, 'feature'), exist_ok=True)
            os.makedirs(os.path.join(self.testing_data_path, 'target'), exist_ok=True)
            for k in self.regional_link_ids:
                self.training_feature_dict[k].to_csv(os.path.join(self.training_data_path, 'feature', '{}.csv'.format(k)), index=False)
                self.training_target_dict[k].to_csv(os.path.join(self.training_data_path, 'target', '{}.csv'.format(k)), index=False)
                self.testing_feature_dict[k].to_csv(os.path.join(self.testing_data_path, 'feature', '{}.csv'.format(k)), index=False)
                self.testing_target_dict[k].to_csv(os.path.join(self.testing_data_path, 'target', '{}.csv'.format(k)), index=False)

    def load_datasets(self):
        self.logger.info('Loading preprocessed link data for...')

        self.training_feature_dict, self.training_target_dict = self._find_datasets(file_path=self.training_data_path)
        self.testing_feature_dict, self.testing_target_dict = self._find_datasets(file_path=self.testing_data_path)

    def train_model_in_sklearn(self, model_type, save=True):
        self.logger.info('Training models...')
        self.model = SeparateModel(training_feature_dict=self.training_feature_dict,
                                   training_target_dict=self.training_target_dict,
                                   testing_feature_dict=self.testing_feature_dict,
                                   testing_target_dict=self.testing_target_dict,
                                   model_type=model_type)

        self.model.train_models(save)

    def evaluate_model(self, save, model_type):
        validation_true, validation_pred = self.model.get_validation_true_pred()
        testing_true, testing_pred = self.model.get_testing_true_pred()
        self.evaluator = Evaluator(validation_true=validation_true, validation_pred=validation_pred,
                                   testing_true=testing_true, testing_pred=testing_pred, model_type=model_type)
        self.evaluator.evaluate(save)

    def _find_datasets(self, file_path):
        feature_df = {}
        target_df = {}
        if not os.listdir(os.path.join(file_path, 'feature')) or not os.listdir(os.path.join(file_path, 'target')):
            self.logger.error('The dataset is invalid. Please try to load raw dataset and preprocess it')
            exit()
        else:
            for link_id in self.regional_link_ids:
                feature_df[link_id] = pd.read_csv(os.path.join(file_path, 'feature', '{}.csv'.format(link_id)))
                target_df[link_id] = pd.read_csv(os.path.join(file_path, 'target', '{}.csv'.format(link_id)))
        return feature_df, target_df

    def _load_raw_data(self, start_time, end_time, file_path, save):
        os.makedirs(os.path.join(file_path, 'raw'), exist_ok=True)
        raw_data_df = {}
        if not os.listdir(os.path.join(file_path, 'raw')):
            raw_data_df = self._pulling_data(start_time, end_time)
            if save:
                for k, v in raw_data_df.items():
                    v.to_csv(os.path.join(file_path, 'raw', '{}.csv'.format(k)), index=False)
            return raw_data_df
        else:
            for file_name in os.listdir(os.path.join(file_path, 'raw')):
                raw_data_df[int(file_name[:-4])] = pd.read_csv(os.path.join(os.path.join(file_path, 'raw'), file_name))
        return raw_data_df

    def _pulling_data(self, start_time, end_time):
        link_neighbours = self.road_network.get_link_neighbours()
        pulling_link_ids = copy.deepcopy(self.regional_link_ids)

        for idx in self.regional_link_ids:
            pulling_link_ids.extend(link_neighbours[idx])
        pulling_link_ids = list(set(pulling_link_ids))

        return self.data_harvester.get_df_dict(link_ids=pulling_link_ids, start_time=start_time, end_time=end_time)

    def _set_logger(self):
        self.logger = logging.getLogger('model_builder')
        format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        # sh = logging.StreamHandler()
        # sh.setFormatter(format_str)
        th = logging.FileHandler(filename='../logs/model_builder.log', mode='w', encoding='utf-8')
        th.setFormatter(format_str)
        # self.logger.addHandler(sh)
        self.logger.addHandler(th)


