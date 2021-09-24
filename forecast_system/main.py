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
from data_processor import DataProcessor
from forecaster import SeparateModel
from datetime import datetime
import pytz
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import copy
from tqdm import tqdm

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Time Zone
TZ = pytz.timezone('Australia/Sydney')

# KsqlDB Address
QUERY_URL = 'https://api.dev.unimelb-traffico.cloud.edu.au/query'
# TABLE_NAME = '24HR_BLOCKS'
TABLE_NAME = 'ED_5MINHW8'

# Personal Cert
KEY_ROOT = os.path.join('..', 'key')
CA_CERT = os.path.join(KEY_ROOT, 'ca.crt')
CLIENT_CERT = os.path.join(KEY_ROOT, 'client.crt')
CLIENT_KEY = os.path.join(KEY_ROOT, 'client.key')

# Local File Path
DATA_ROOT = os.path.join('../data')
LINK_GDF_PATH = os.path.join(DATA_ROOT, 'link_gdf.geojson')
LINK_GEO_PATH = os.path.join(DATA_ROOT, 'GeoLinkData.geojson')
LGA_GEO_PATH = os.path.join(DATA_ROOT, 'LinkLGAData.geojson')
POI_GEO_PATH = os.path.join(DATA_ROOT, 'VicPOIData.geojson')

# Research Configuration
RESEARCH_REGION = 'MELBOURNE CITY'  # MELBOURNE CITY has 323 links and BENALLA RURAL CITY has 2 links
START_TIME = datetime(year=2021, month=9, day=15, hour=10, minute=0, second=0)
END_TIME = datetime(year=2021, month=9, day=22, hour=10, minute=0, second=0)

road_nx = RoadNetwork(link_geo_path=LINK_GEO_PATH, lga_geo_path=LGA_GEO_PATH,
                      poi_geo_path=POI_GEO_PATH, processed_link_path=LINK_GDF_PATH)
data_harvester = DataHarvester(ca_cert=CA_CERT, client_crt=CLIENT_CERT, client_key=CLIENT_KEY,
                               query_url=QUERY_URL, table_name=TABLE_NAME)

# Initialize network information
road_nx.count_poi_around_link(buffer_distance=400)
regional_link_ids = road_nx.get_regional_link_ids(region_name=RESEARCH_REGION)
link_neighbours = road_nx.get_link_neighbours()

# Pulling data from ksqlDB and store them at local client (Only when you don't have them at local client)
pulling_link_ids = copy.deepcopy(regional_link_ids)
for idx in regional_link_ids:
    pulling_link_ids.extend(link_neighbours[idx])
pulling_link_ids = list(set(pulling_link_ids))

for i in tqdm(range(60, len(pulling_link_ids), 20)):
    print(pulling_link_ids[i:i+20])
    delay_df = data_harvester.get_df_dict(pulling_link_ids[i:i+20], start_time=START_TIME, end_time=END_TIME)
    for k, df in delay_df.items():
        df.to_csv('../data/ex_delay_091510-092210/{}.csv'.format(k), index=False)

# Loading local files
delay_df_dict = {}
for file_name in os.listdir('../data/ex_delay_091510-092210'):
    delay_df_dict[int(file_name[:-4])] = pd.read_csv('../data/ex_delay_091510-092210/{}'.format(file_name))

dataset = DataProcessor(delay_df_dict=delay_df_dict, link_data=road_nx.get_link_gdf(),
                        link_neighbours=link_neighbours,
                        window_size=3)
dataset.save()

# Training models
features_dict = {}
target_dict = {}
for file_name in os.listdir('../data/dataset_features'):
    features_dict[int(file_name[:-4])] = pd.read_csv('../data/dataset_features/{}'.format(file_name))

for file_name in os.listdir('../data/dataset_target'):
    target_dict[int(file_name[:-4])] = pd.read_csv('../data/dataset_target/{}'.format(file_name))

model = SeparateModel(features_dict=features_dict, target_dict=target_dict)
model.train_models()
model.cal_mse()
model.cal_mape()








