"""
@file:VicRoadForecast-PyCharm-data_harvester.py
@time: 17/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""

import logging
from pytz import timezone
import json
import requests
import pandas as pd
import datetime
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class DataHarvester:
    def __init__(self, ca_cert, client_crt, client_key, query_url, table_name):
        """Initial the variables in class DataHarvester.

        :param ca_cert: certification of ca
        :param client_crt: certification of client
        :param client_key: public key of client
        :param query_url: url of ksqlDB
        :param table_name: The table storing data
        """
        self.logger = logging.getLogger(__name__)
        # self.tz = timezone_aus
        self.ca_cert = ca_cert
        self.client_crt = client_crt
        self.client_key = client_key
        self.query_url = query_url
        self.table_name = table_name

        self.response = None
        self.start_timestamp = None
        self.end_timestamp = None

        self.raw_df = None
        self.link_df_dict = None

    def get_df_dict(self, link_ids, start_time, end_time):
        """Return data that the delays of links in link_ids during a certain time range.

        :param link_ids: a list of link id
        :param start_time: start time of data
        :param end_time: end time of data
        :return: a dict in which the the key is ID of links and the value is dataframe of delays for each link
        """
        self.logger.info("Downloading raw data from KsqlDB...")
        self.start_timestamp = int(start_time.timestamp())
        self.end_timestamp = int(end_time.timestamp())
        start_time_ms = self.start_timestamp * 1000
        end_time_ms = self.end_timestamp * 1000
        ids_str = ','.join([str(link) for link in link_ids])

        self.link_df_dict = {}
        self._pull_data(ids_str, start_time_ms, end_time_ms)
        self._json_data_to_df()
        self._separate_df_for_each_link(link_ids)
        self._fill_missing_value()
        self.logger.info("Returning a dictionary contains data for each link...")
        return self.link_df_dict

    def _pull_data(self, ids_str, start_time_ms, end_time_ms):
        """Pull data from the source database.

        :param ids_str: a string of link ids separated by comma
        :param start_time_ms: the timestamp of start time with the unit of milliseconds
        :param end_time_ms: the timestamp of end time with the unit of milliseconds
        """
        # start_time_ms = 1621123200000  # 2021-05-16 10:00:00
        # end_time_ms = 1621124400000  # 2021-05-16 10:20:00
        # Downloading raw data from KsqlDB
        # rows_required = 5
        # ksql_query = "SELECT ID, DELAY_LIST, ED_LIST " \
        #              "FROM 24HR_BLOCKS " \
        #              "WHERE ID = 1558 " \
        #              "AND WINDOWSTARTTIME >= {} " \
        #              "AND WINDOENDTIME <={} EMIT CHANGES LIMIT {};".format(start_time_ms, end_time_ms, rows_required)

        ksql_query = "SELECT ID, WINDOWSTART, LATEST_ED FROM {} " \
                     "WHERE ID IN({}) " \
                     "AND WINDOWSTART >= {} " \
                     "AND WINDOWSTART <= {};".format(self.table_name, ids_str, start_time_ms, end_time_ms)

        ksql_body = json.dumps({'ksql': ksql_query, 'streamProperties': {}})
        self.response = requests.post(self.query_url, data=ksql_body, verify=self.ca_cert,
                                      cert=(self.client_crt, self.client_key), stream=False)
        self.logger.info("Raw data from KsqlDB have been downloaded...")
        # print(self.response.text)

    def _json_data_to_df(self):
        """
        Transform the json type to dataframe.
        """
        columns = []
        data = []

        try:
            raw_response = json.loads(self.response.text)
            try:
                for col in raw_response[0]['header']['schema'].split(','):
                    columns.append(col.split('`')[1])
                for link in raw_response[1:]:
                    data.append(link['row']['columns'])
            except KeyError:
                self.logger.error("Failed to format query response: 'KeyError': Either header or row is missing!")
        except TypeError:
            self.logger.error("Failed to get data from database: TypeError: Response test is Null!")

        self.raw_df = pd.DataFrame(data, columns=columns)
        self.raw_df['WINDOWSTART'] /= 1000

    def _separate_df_for_each_link(self, link_ids):
        for link_id in link_ids:
            self.link_df_dict[link_id] = self.raw_df.loc[self.raw_df['ID'] == link_id, :]

    def _fill_missing_value(self):
        """
        Join original data with the timeline. For the missing ID, we use next valid observation to fill gap. For the
        missing LATEST_ED, we use linear interpolation to fill gap.
        """
        duration = self.end_timestamp - self.start_timestamp
        interval_num = duration // 30
        timeline = pd.Series([self.start_timestamp + i * 30 for i in range(interval_num)], name='TimeStamp')
        for k in self.link_df_dict.keys():
            print(self.link_df_dict[k])
            self.link_df_dict[k] = self.link_df_dict[k].merge(timeline,
                                                              how='right',
                                                              left_on='WINDOWSTART',
                                                              right_on='TimeStamp')
            print(self.link_df_dict[k])
            self.link_df_dict[k]['ID'].fillna(k, inplace=True)
            self.link_df_dict[k].drop(columns=['WINDOWSTART'], inplace=True)
            self.link_df_dict[k].interpolate(method='linear', axis=0, inplace=True)
            print(self.link_df_dict[k])
            self.link_df_dict[k][['ID', 'TimeStamp', 'LATEST_ED']] = self.link_df_dict[k][['ID', 'TimeStamp', 'LATEST_ED']].astype(int)



