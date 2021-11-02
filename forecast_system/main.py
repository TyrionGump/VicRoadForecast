"""
@file:VicRoadForecast-PyCharm-main.py
@time: 17/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""

from datetime import datetime
import pytz
import os
import pandas as pd
from model_builder import ModelBuilder
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Time Zone
TZ = pytz.timezone('Australia/Sydney')

# KsqlDB Address
QUERY_URL = 'https://api.dev.unimelb-traffico.cloud.edu.au/query'
# TABLE_NAME = 'ED_5MINHW8'
TABLE_NAME = '24HR_TT_DENSITY_V2'

# Personal Cert
KEY_ROOT = os.path.join('..', 'key')
CA_CERT = os.path.join(KEY_ROOT, 'ca.crt')
CLIENT_CERT = os.path.join(KEY_ROOT, 'client.crt')
CLIENT_KEY = os.path.join(KEY_ROOT, 'client.key')

# Research Configuration
RESEARCH_REGION = 'MELBOURNE CITY'  # MELBOURNE CITY has 323 links and BENALLA RURAL CITY has 2 links
TRAINING_START_TIME = datetime(year=2021, month=10, day=7, hour=11, minute=0, second=0)
TRAINING_END_TIME = datetime(year=2021, month=10, day=14, hour=11, minute=0, second=0)
TESTING_START_TIME = datetime(year=2021, month=10, day=14, hour=11, minute=0, second=0)
TESTING_END_TIME = datetime(year=2021, month=10, day=15, hour=11, minute=0, second=0)

# Local File Path
DATA_ROOT = os.path.join('../data')
LINK_GDF_PATH = os.path.join(DATA_ROOT, 'link_gdf.geojson')
LINK_GEO_PATH = os.path.join(DATA_ROOT, 'GeoLinkData.geojson')
LGA_GEO_PATH = os.path.join(DATA_ROOT, 'LinkLGAData.geojson')
POI_GEO_PATH = os.path.join(DATA_ROOT, 'VicPOIData.geojson')

if __name__ == '__main__':
    # Define the research configuration
    system = ModelBuilder(data_root_path=DATA_ROOT,
                          link_geo_path=LINK_GEO_PATH, lga_geo_path=LGA_GEO_PATH, poi_geo_path=POI_GEO_PATH,
                          link_gdf_path=LINK_GDF_PATH,
                          ca_cert=CA_CERT, client_cert=CLIENT_CERT, client_key=CLIENT_KEY,
                          query_url=QUERY_URL, table_name=TABLE_NAME,
                          training_start_time=TRAINING_START_TIME, training_end_time=TRAINING_END_TIME,
                          testing_start_time=TESTING_START_TIME, testing_end_time=TESTING_END_TIME,
                          region=RESEARCH_REGION)

    # Request data from ksqlDB
    system.request_raw_data(save=True)
    # Processing raw data. Check the detailed feature selection from line 103 to line 113 in ModelBuilder.py.
    system.data_preprocessing(forward_steps=20, backward_steps=10, minute_interval=5, save=True)

    # Load the local pre-processed dateset (Ignore the process of line 63 and line 65)
    # system.load_datasets()

    # Train model with class from sklearn
    system.train_model_in_sklearn(model_type=LinearRegression, save=True)

    # Evaluate model
    system.evaluate_model(model_type=LinearRegression, save=True)





