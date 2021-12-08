# -*- coding: utf-8 -*-
"""
@File:VicRoadForecast-PyCharm-traffic_harvester.py
@Date: 17/9/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import json
import logging
import os
import time
import warnings
from itertools import chain
from datetime import datetime
from config import args
import copy

import pandas as pd
import requests

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('max_rows', 100)


class TrafficHarvester:
    """

    This class request data from ksqlDB based on the pre-defined link ids and research duration.
    The missing data are filtered based on the research duration. A dictionary of link id and
    a table (DataFrame) can be obtained. The columns of dataframe can be redefined based on the
    demands.

    """

    def __init__(self):
        """Constructor of Class DataHarvester

        The inputs of this constructor are the paths to keys, the address of ksqlDB and the table
        name.

        Args:
            ca_cert: certification of client
            client_crt: certification of client
            client_key: public key of client
            query_url: url of ksqlDB
            table_name: The table storing data

        """
        self.link_ids = None
        self.start_timestamp = None  # Start time point of the research duration
        self.end_timestamp = None  # End time point of the research duration

        self.raw_df = None
        self.link_df_dict = {}  # link ids and their corresponding data table (DataFrame)

        self.logger = None
        self._set_logger()

    def get_df_dict(self, link_ids, start_time, end_time):
        """Get data that the delays of links in link_ids during a certain time duration.

        According to the required link id and time duration, request data from ksqlDB. Then, separate the
        raw integrate table (DataFrame) into a dictionary in which the keys represent link id and the values
        are corresponding data table (DataFrame).

        Args:
            link_ids: a list of link id
            start_time: start time of data
            end_time: end time of data

        Returns:
            A dictionary in which the the key is ID of links and the value is dataframe of delays for each link

        """
        self.logger.info("Downloading raw data from table {}...".format(args.id['db']['TABLE_NAME']))
        self.link_ids = link_ids
        self.start_timestamp = int(datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp())
        self.end_timestamp = int(datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').timestamp())
        start_time_ms = self.start_timestamp * 1000
        end_time_ms = self.end_timestamp * 1000
        ids_str = ','.join([str(link) for link in link_ids])

        self._pull_data(ids_str, start_time_ms, end_time_ms)
        self._reform_raw_df()
        self._separate_df_for_each_link()
        self._fill_missing_or_drop_duplication()
        self.logger.info("Returning a dictionary contains raw data for each link...")
        return copy.deepcopy(self.link_df_dict)

    def save(self, raw_path=''):
        for k, v in self.link_df_dict.items():
            v.to_csv(os.path.join(raw_path, '{}.csv'.format(k)), index=False)

    def _pull_data(self, ids_str, start_time_ms, end_time_ms):
        """request raw data from ksqlDB

        Request data from ksqlDB. The query is based on requirements and it will influence how you process
        them in Pandas. The result is returned in json type and it will be transformed into a DataFrame table.
        It is noted that long time querying will abort the connection.

        Args:
            ids_str: a string of link ids separated by comma
            start_time_ms: the timestamp of start time with the unit of milliseconds
            end_time_ms: the timestamp of end time with the unit of milliseconds

        """
        # ksql_query = "SELECT * FROM {};".format(self.table_name)
        ksql_query = "SELECT * FROM {} " \
                     "WHERE ID IN({}) " \
                     "AND WINDOWSTART >= {} " \
                     "AND WINDOWEND <= {};".format(args.id['db']['TABLE_NAME'], ids_str, start_time_ms, end_time_ms)

        ksql_body = json.dumps({'ksql': ksql_query, 'streamProperties': {}})
        columns = []
        data = []

        download_flag = True
        while download_flag:
            try:
                response = requests.post(args.id['db']['QUERY_URL'],
                                         data=ksql_body,
                                         verify=args.id['key']['CA_CERT'],
                                         cert=(args.id['key']['CLIENT_CERT'], args.id['key']['CLIENT_KEY']),
                                         stream=False)

                raw_response = json.loads(response.text)
                for col in raw_response[0]['header']['schema'].split(','):
                    columns.append(col.split('`')[1])
                for link in raw_response[1:]:
                    data.append(link['row']['columns'])

                download_flag = False
            except (KeyError, TypeError, json.decoder.JSONDecodeError) as e:
                self.logger.error(e)
                self.logger.info('There are some errors !!! Trying to request data again...')
                time.sleep(15)

        self.raw_df = pd.DataFrame(data, columns=columns)

    def _reform_raw_df(self):
        """Reform data in self.raw_df

        The original data from ksqlDB is a collection of lists. We try to extract the elements in these lists as a
        single grid.

        """
        self.raw_df = pd.DataFrame({
            'ID': self.raw_df['ID'].values.repeat(self.raw_df['TRAVEL_TIME'].str.len()),
            'travel_time': list(chain.from_iterable(self.raw_df['TRAVEL_TIME'].tolist())),
            'timestamp': list(chain.from_iterable(self.raw_df['INTERVAL_START'].tolist())),
        })
        self.raw_df['timestamp'] = self.raw_df['timestamp'].apply(lambda x:int(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))

    def _separate_df_for_each_link(self):
        """Separate the original table for each link id

        Args:
            link_ids: a list of link id

        """
        for link_id in self.link_ids:
            self.link_df_dict[link_id] = self.raw_df.loc[self.raw_df['ID'] == link_id, :]
            if len(self.link_df_dict[link_id]) == 0:
                self.logger.error("The table of link {} is empty!".format(link_id))

    def _fill_missing_or_drop_duplication(self):
        """Fill missing value of each link

        Join original data with the timeline. For the missing ID, we use next valid observation to fill gap. For the
        missing LATEST_ED, we use linear interpolation to fill gap. The reason why each link is processed separately
        is that the rows of each link should be compared to the timeline.

        """
        duration = self.end_timestamp - self.start_timestamp
        interval_num = duration // 30
        timeline = pd.Series([self.start_timestamp + i * 30 for i in range(interval_num)], name='timeline')

        for link_id in self.link_df_dict.keys():
            expected_rows_num = len(timeline)
            actual_rows_num = len(self.link_df_dict[link_id])

            if actual_rows_num == 0:
                continue
            if actual_rows_num != expected_rows_num:
                missing_rows_num = expected_rows_num - actual_rows_num
                missing_rate = round(missing_rows_num * 100 / expected_rows_num, 2)
                if missing_rate >= 5:
                    self.logger.warning("Missing rate is {}% in link {}".format(missing_rate, link_id))
                elif missing_rate <= -5:
                    self.logger.warning("Duplication rate is {}% in link {}".format(-missing_rate, link_id))

            self.link_df_dict[link_id] = self.link_df_dict[link_id].merge(timeline,
                                                                          how='right',
                                                                          left_on='timestamp',
                                                                          right_on='timeline')
            self.link_df_dict[link_id].drop_duplicates(inplace=True)
            self.link_df_dict[link_id]['ID'].fillna(link_id, inplace=True)
            self.link_df_dict[link_id].drop(columns=['timestamp'], inplace=True)
            self.link_df_dict[link_id].interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
            self.link_df_dict[link_id].rename(columns={'timeline': 'timestamp'}, inplace=True)
            self.link_df_dict[link_id].reset_index(drop=True, inplace=True)

            try:
                self.link_df_dict[link_id] = self.link_df_dict[link_id].astype('int32')
            except pd.errors.IntCastingNaNError:
                self.link_df_dict[link_id] = pd.DataFrame([], columns=self.link_df_dict[link_id].columns)
                self.logger.error("The table of link {} is empty!".format(link_id))

    def _set_logger(self):
        self.logger = logging.getLogger('data_harvester')
        format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        # sh = logging.StreamHandler()
        # sh.setFormatter(format_str)
        th = logging.FileHandler(filename=args.file_config['log_file']['DATA_HARVESTER'], mode='w', encoding='utf-8')
        th.setFormatter(format_str)
        # self.logger.addHandler(sh)
        self.logger.addHandler(th)
