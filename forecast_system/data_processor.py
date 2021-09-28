"""
@file:VicRoadForecast-PyCharm-data_processor.py
@time: 24/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""

import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

pd.options.display.float_format = '{:.1f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class DataProcessor:
    def __init__(self, delay_df_dict: dict, link_data: pd.DataFrame, link_neighbours: dict, window_size=3):
        self.features_dict = {}
        self.target_dict = {}

        self._delay_df_dict = delay_df_dict
        self._window_size = window_size
        self._research_min = None
        self._rolled_delay_dict = {}

        self._link_data = link_data
        self._link_neighbours = link_neighbours

        self.logger = logging.getLogger(__name__)
        self._create_delay_dataset()

    def _create_delay_dataset(self):
        self._aggregate_delays()
        self._rolling_delays()
        self._add_neighbours_delays()
        self._add_datetime()
        self._add_hour_of_day()
        self._add_day_of_week()
        self._add_link_features()
        self._drop_id_timestamp()

    def _aggregate_delays(self, minute=3):
        self.logger.info("Aggregating delay data based on a {}-minute window...".format(minute))

        self.research_min = minute
        for link_id in self._delay_df_dict.keys():
            aggregate_res = []
            for i in range(0, len(self._delay_df_dict[link_id]), minute * 2):
                row = [link_id, self._delay_df_dict[link_id].loc[i + minute, 'TimeStamp'],
                       self._delay_df_dict[link_id].loc[i:i + minute * 2, 'LATEST_ED'].mean()]
                aggregate_res.append(row)
            self._delay_df_dict[link_id] = pd.DataFrame(aggregate_res, columns=['id', 'TimeStamp', 'LATEST_ED'])

    def _rolling_delays(self):
        self.logger.info("Creating the dataset based on a {}-interval window...".format(self._window_size))

        for link_id, delay_df in self._delay_df_dict.items():
            rolled_df = delay_df.rolling(window=self._window_size, center=False)
            rolled_features = []
            rolled_target = []
            for delays in rolled_df:
                if len(delays) < self._window_size or delays.index.to_list()[-1] + 1 >= len(delay_df):
                    continue

                features = [link_id, delays['TimeStamp'].to_list()[self._window_size // 2]]
                features.extend(delays['LATEST_ED'].to_list())
                rolled_features.append(features)

                target = [delay_df.loc[delays.index.to_list()[-1] + 1, 'LATEST_ED']]
                rolled_target.append(target)

            column_names = ['id', 'TimeStamp']
            column_names.extend(['{}_-{}'.format(link_id, i) for i in range(self._window_size, 0, -1)])
            self._rolled_delay_dict[link_id] = pd.DataFrame(rolled_features, columns=column_names, dtype='float32')
            self._rolled_delay_dict[link_id].drop(columns=['id', 'TimeStamp'], inplace=True)

            self.features_dict[link_id] = pd.DataFrame(rolled_features, columns=column_names, dtype='float32')
            self.target_dict[link_id] = pd.DataFrame(rolled_target, columns=['{}_0'.format(link_id)], dtype='float32')

            self._delay_df_dict = None

    def _add_neighbours_delays(self):
        self.logger.info("Adding neighbours delays for each link...")
        flag = 0
        for link_id in tqdm(self.features_dict.keys()):
            flag += 1
            neighbour_ids = self._link_neighbours[link_id]
            for neighbour_id in neighbour_ids:
                # Some data are not downloaded
                try:
                    self.features_dict[link_id] = pd.concat([self.features_dict[link_id], self._rolled_delay_dict[neighbour_id]], axis=1)
                except KeyError:
                    self.logger.warning('Miss data of link_{}'.format(neighbour_id))
                    continue

    def _add_datetime(self):
        self.logger.info("Adding datetime each link...")
        for link_id in self.features_dict:
            self.features_dict[link_id]['timestamp'] = \
                self.features_dict[link_id]['TimeStamp'].map(lambda x: datetime.fromtimestamp(x))

    def _add_hour_of_day(self):
        self.logger.info("Adding hour of day for each link...")
        for link_id in self.features_dict:
            self.features_dict[link_id]['HourOfDay'] = \
                self.features_dict[link_id]['TimeStamp'].map(lambda x: datetime.fromtimestamp(x).hour)

    def _add_day_of_week(self):
        self.logger.info("Adding hour of day for each link...")
        for link_id in self.features_dict:
            self.features_dict[link_id]['DayOfWeek'] = \
                self.features_dict[link_id]['TimeStamp'].map(lambda x: datetime.fromtimestamp(x).weekday())

    def _add_link_features(self):
        self.logger.info("Adding link features for each link...")
        feature_name = ['id', 'length', 'min_number_of_lanes', 'is_freeway']
        link_df = self._link_data[feature_name]
        for link_id in self.features_dict:
            self.features_dict[link_id] = self.features_dict[link_id].merge(link_df, left_on='id', right_on='id')

    def _drop_id_timestamp(self):
        for link_id in self.features_dict:
            self.features_dict[link_id].drop(columns=['id', 'TimeStamp'], inplace=True)

    def save(self):
        for k, v in self.features_dict.items():
            v.to_csv('../data/dataset_features/{}.csv'.format(k), index=False)
        for k, v in self.target_dict.items():
            v.to_csv('../data/dataset_target/{}.csv'.format(k), index=False)
