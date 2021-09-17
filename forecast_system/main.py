"""
@file:VicRoadForecast-PyCharm-main.py
@time: 17/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""

from data_harvester import DataHarvester
from road_network import RoadNetwork
from datetime import datetime
import pytz
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Time Zone
TZ = pytz.timezone('Australia/Sydney')

# KsqlDB Address
QUERY_URL = 'https://api.dev.unimelb-traffico.cloud.edu.au/query'
# TABLE_NAME = 'DELAY_CONGESTION_TAGGED'
TABLE_NAME = 'ED_5MINHW8'

# Personal Cert
KEY_ROOT = os.path.join('..', 'key')
CA_CERT = os.path.join(KEY_ROOT, 'ca.crt')
CLIENT_CERT = os.path.join(KEY_ROOT, 'client.crt')
CLIENT_KEY = os.path.join(KEY_ROOT, 'client.key')

# Research Configuration
# LINK_IDS = [2727]
# START_TIME = datetime(year=2021, month=9, day=10, hour=00, minute=0, second=0)
# END_TIME = datetime(year=2021, month=9, day=16, hour=00, minute=0, second=0)
#
# data_harvester = DataHarvester(ca_cert=CA_CERT, client_crt=CLIENT_CERT, client_key=CLIENT_KEY,
#                                query_url=QUERY_URL, table_name=TABLE_NAME)
#
# df_dict = data_harvester.get_df_dict(link_ids=LINK_IDS, start_time=START_TIME, end_time=END_TIME)

# Local File Path
DATA_ROOT = os.path.join('../data')
LINK_GEO_PATH = os.path.join(DATA_ROOT, 'GeoLinkData.geojson')
LGA_GEO_PATH = os.path.join(DATA_ROOT, 'LinkLGAData.geojson')
POI_GEO_PATH = os.path.join(DATA_ROOT, 'POI', 'VicPOIData.geojson')

n = RoadNetwork(link_geo_path=LINK_GEO_PATH, lga_geo_path=LGA_GEO_PATH, poi_geo_path=POI_GEO_PATH)
n.count_poi_around_link(buffer_distance=400)
print(n.get_link_neighbours()[2727])
