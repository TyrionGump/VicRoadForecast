# -*- coding: utf-8 -*-
"""
@File:data_aggregator.py
@Date: 2021/11/14
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import numpy as np
import pandas as pd
from datetime import datetime
from forecast_system.road_network import RoadNetwork
import logging

pd.options.display.float_format = '{:.1f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_rows', None)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class DataAggregator:
    def __init__(self, road_network: RoadNetwork):
        self.road_network = road_network
        self.processed_df_dict = None

    @staticmethod
    def add_temporal_features(df_dict: dict):
        logging.info('adding temporal features...')
        for link_id in df_dict.keys():
            df_dict[link_id]['temporal_hour_of_day'] = df_dict[link_id]['timestamp'].map(lambda x: datetime.fromtimestamp(x).hour)
            df_dict[link_id]['temporal_day_of_week'] = df_dict[link_id]['timestamp'].map(
                lambda x: datetime.fromtimestamp(x).weekday())

    def add_link_features(self, df_dict: dict, buffer_distances=[400]):
        logging.info('adding link features...')
        self.road_network.cal_poi_density(buffer_distances)
        feature_name = ['id', 'length', 'min_number_of_lanes', 'is_freeway']
        for d in buffer_distances:
            feature_name.append('poi_density_{}m'.format(d))
        link_df = self.road_network.link_gdf[feature_name]
        for link_id in df_dict.keys():
            df_dict[link_id] = df_dict[link_id].merge(link_df, left_on='ID', right_on='id')
            df_dict[link_id].drop(columns=['id'], inplace=True)




