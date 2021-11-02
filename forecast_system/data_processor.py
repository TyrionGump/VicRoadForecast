# -*- coding: utf-8 -*-
"""
@File:VicRoadForecast-PyCharm-data_processor.py
@Date: 24/9/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import os

import numpy as np
import pandas as pd
from datetime import datetime
from road_network import RoadNetwork
import logging

pd.options.display.float_format = '{:.1f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_rows', None)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class DataProcessor:
    """
    This class receive the processed data before then process and integrate them to create dataset for future
    machine learning analysis. According to the requirements, the user can add features by implementing methods
    in _create_delay_dataset. The user can also customer these features in the corresponding methods.
    """
    def __init__(self, link_df_dict: dict, road_network: RoadNetwork, region_name):
        """Constructor

        The inputs are the data which will be integrated in the following steps.

        Args:
            link_df_dict: a dictionary containing link id and the responding data
            road_network: an object of road network
        """
        self._link_df_dict = link_df_dict
        self._road_network = road_network
        self._region_name = region_name

        self._forward_steps = None  # The number of forward steps (e.g. t, t-1, t-2...)
        self._backward_steps = None  # The number of forward steps (e.g. t+1, t+2, t+3...)
        self._interval_min = 0.5  # The minutes that the 30s is aggregated
        self._rolled_data_dict = {}  # The rolled time series of each link
        self._region_link_ids = self._road_network.get_regional_link_ids(self._region_name)

        self.features_dict = {}  # The features of each link for machine learning algorithms
        self.target_dict = {}  # The target of each link for machine learning algorithms

        self.logger = None
        self._set_logger()

    def get_dataset(self):
        """Get the processed dataset

        Extract the links within the research region and ignore their neighbours out of the region. Then, return
        the feature set and the target set.

        Returns: A dictionary contain the feature set and the target set of each link

        """
        self._filter_regional_data()
        return self.features_dict, self.target_dict

    def aggregate_time_series(self, minute=5):
        """ Aggregate interval of time series from 30 seconds to m minutes

        Firstly, use the median value of original timestamps as the new timestamp.
        Secondly, use the mean value of delay data as the new data

        Args:
            minute: the duration of new interval

        """
        self.logger.info("Aggregating delay data based on a {}-minute window...".format(minute))

        self._interval_min = minute
        column_name = self._link_df_dict[self._region_link_ids[0]].columns

        for link_id, link_df in self._link_df_dict.items():
            aggregate_data = []
            t = 0
            while (t + 1) * minute * 2 < len(link_df):
                aggregate_data.append(link_df.loc[t * minute * 2: (t + 1) * minute * 2, :].mean().to_list())
                t += 1
            self._link_df_dict[link_id] = pd.DataFrame(aggregate_data, columns=column_name, dtype='int32')

    def rolling_time_series(self, forward_steps=3, backward_steps=1):
        """Concat the data in the forward steps with the data at the current point

        Rolling the original data and use each rolling result as a new row in the dataset. It is noted that
        the head of the rolling result will be discarded since there will be some nan value in them.

        Args:
            forward_steps: The number of forward steps (e.g. t, t-1, t-2...)
            backward_steps: The number of backward steps (e.g. t+1, t+2, t+3...)

        """
        self.logger.info("Creating the dataset based on {} forward steps and {} backward steps".format(forward_steps, backward_steps))

        self._forward_steps = forward_steps
        self._backward_steps = backward_steps

        window_size = self._forward_steps + self._backward_steps
        traffic_feature_name = list(self._link_df_dict[self._region_link_ids[0]].columns)
        traffic_feature_name.remove('ID')
        traffic_feature_name.remove('TIMESTAMP')

        for link_id, link_df in self._link_df_dict.items():
            rolled_df = link_df.rolling(window=window_size, center=False)
            rolled_features = []
            rolled_target = []
            for series in rolled_df:
                if len(series) < window_size or series.index.to_list()[-1] >= link_df.shape[0] - 1:
                    continue
                timestamp = series['TIMESTAMP'].to_list()[self._forward_steps - 1]
                extracted_features = [link_id, timestamp]
                extracted_targets = [link_id, timestamp]

                # Try to reorder the extracted features for the future analysis
                tmp_extracted_features = []
                for column in traffic_feature_name:
                    tmp_extracted_features.append(series[column].to_list()[:self._forward_steps])
                tmp_extracted_features = np.array(tmp_extracted_features).T.reshape(-1).tolist()

                extracted_features.extend(tmp_extracted_features)
                extracted_targets.extend(series['TRAVEL_TIME'].to_list()[self._forward_steps:])

                rolled_features.append(extracted_features)
                rolled_target.append(extracted_targets)

            new_features_columns = ['ID', 'TIMESTAMP']
            new_target_columns = ['ID', 'TIMESTAMP']

            for step in range(self._forward_steps):
                for column in traffic_feature_name:
                    if step != 0:
                        new_features_columns.append('{}_{}_t-{}'.format(link_id, column, step))
                    else:
                        new_features_columns.append('{}_{}_t'.format(link_id, column))

            for step in range(self._backward_steps):
                new_target_columns.append('{}_TRAVEL_TIME_t+{}'.format(link_id, step + 1))

            self.features_dict[link_id] = pd.DataFrame(rolled_features, columns=new_features_columns, dtype='int32')
            self.target_dict[link_id] = pd.DataFrame(rolled_target, columns=new_target_columns, dtype='int32')

    def add_neighbours_data(self):
        """Add neighbours time series for each link

        According to the neighbour information in the road network, horizontally concat these these
        link delays columns. When encountering missing link data, the program will warn and skip that neighbour.

        """
        self.logger.info("Adding neighbours delays for each link...")

        tmp_features_dict = {}

        for link_id in self._region_link_ids:
            neighbour_ids = self._road_network.link_neighbours[link_id]
            for neighbour_id in neighbour_ids:
                # Some data are not downloaded
                try:
                    tmp_features_dict[link_id] = pd.concat(
                        [self.features_dict[link_id], self.features_dict[neighbour_id].drop(columns=['ID', 'TIMESTAMP'])], axis=1)
                except KeyError:
                    self.logger.warning('Miss neighbours link_{} for link_{}'.format(neighbour_id, link_id))
                    continue

        self.features_dict = tmp_features_dict

    def add_hour_of_day(self):
        """Add feature of the hour in each day

        According to the column of timestamp, add a new feature of the hour in each day when that row is recorded.

        """
        self.logger.info("Adding hour of day for each link...")
        for link_id in self._region_link_ids:
            self.features_dict[link_id]['HourOfDay'] = \
                self.features_dict[link_id]['TIMESTAMP'].map(lambda x: datetime.fromtimestamp(x).hour)

    def add_day_of_week(self):
        """Add feature of the day of each week

        According to the column of timestamp, add a new feature of the day in each week when that row is recorded.

        """
        self.logger.info("Adding hour of day for each link...")
        for link_id in self._region_link_ids:
            self.features_dict[link_id]['DayOfWeek'] = \
                self.features_dict[link_id]['TIMESTAMP'].map(lambda x: datetime.fromtimestamp(x).weekday())

    def add_link_features(self, buffer_distances=[400]):
        """Add geographic feature of each link

        Join the table of delay data with the table of geographic data.

        """
        self.logger.info("Adding link features for each link...")

        self._road_network.poi_density(buffer_distances)

        feature_name = ['id', 'length', 'min_number_of_lanes', 'is_freeway']
        for d in buffer_distances:
            feature_name.append('poi_density_{}m'.format(d))

        link_df = self._road_network.link_gdf[feature_name]
        for link_id in self._region_link_ids:
            self.features_dict[link_id] = self.features_dict[link_id].merge(link_df, left_on='ID', right_on='id')
            self.features_dict[link_id].drop(columns=['id'], inplace=True)

    def drop_density(self):
        """Drop the column of density

        We found that the density in the raw data does not have a strong relationship with travel time. Therefore,
        We try to drop this column

        """
        for link_id in self._link_df_dict.keys():
            self._link_df_dict[link_id].drop(columns=['DENSITY'], inplace=True)

    def drop_id_timestamp(self):
        """Drop the column of ID and TIMESTAMP

        ID and TIMESTAMP do not have to appear in the dataset of machine learning model. Therefore, these two
        columns are dropped.

        """
        for link_id in self._region_link_ids:
            self.features_dict[link_id].drop(columns=['ID', 'TIMESTAMP'], inplace=True)
            self.target_dict[link_id].drop(columns=['ID', 'TIMESTAMP'], inplace=True)

    def _filter_regional_data(self):
        """ Extract links within the research region

        Remove the neighbouring links out of the research region

        """
        regional_features_dict = {}
        regional_target_dict = {}

        for link_id in self._region_link_ids:
            regional_features_dict[link_id] = self.features_dict[link_id]
            regional_target_dict[link_id] = self.target_dict[link_id]
        self.features_dict = regional_features_dict
        self.target_dict = regional_target_dict

    def _set_logger(self):
        self.logger = logging.getLogger('data_processor')
        format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        # sh = logging.StreamHandler()
        # sh.setFormatter(format_str)
        th = logging.FileHandler(filename='../logs/data_processor.log', mode='w', encoding='utf-8')
        th.setFormatter(format_str)
        # self.logger.addHandler(sh)
        self.logger.addHandler(th)

