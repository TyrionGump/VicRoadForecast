# -*- coding: utf-8 -*-
"""
@File:link_gdf.geojson-PyCharm-map_printer.py
@Date: 12/10/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""
import os
import geopandas as gpd
import pandas as pd
import folium
import numpy as np
from folium.plugins import TimestampedGeoJson
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as ss
import branca
import matplotlib.pyplot as plt
import matplotlib
import branca.colormap as cm


melb_map = folium.Map(location=[-37.840935, 144.946457], zoom_start=12, tiles='cartodbdark_matter', fill_opacity=0.001)

DATA_ROOT = os.path.join('../data')
LINK_GDF_PATH = os.path.join(DATA_ROOT, 'link_gdf.geojson')
raw_data_2727 = '../data/MELBOURNE CITY/1005110000-1005230000/raw/2727.csv'

gdf = gpd.read_file(LINK_GDF_PATH)
df = pd.read_csv('error.csv')

gdf = pd.merge(left=gdf, right=df, left_on='id', right_on='ID', how='inner')

normolizer = MinMaxScaler()
# normolizer.fit(gdf[['testing_mape']])
gdf['color'] = ss.rankdata(gdf[['testing_mse']])
# gdf['color'] = gdf['color'] * 255
gdf['color'] = gdf['color'].astype('int32')
color_bar = plt.cm.get_cmap('RdYlBu', len(gdf['color'].unique()))

def color_func(feature):
    # print(feature['properties']['color'])
    # print("#%02x%02x%02x"%(200,feature['properties']['color'],40))
    # colorscale = branca.colormap.linear.YlGnBu_09.scale(0, 60)
    color = color_bar(feature['properties']['color'])
    return matplotlib.colors.rgb2hex(color)

folium.GeoJson(gdf.to_json(),
               name='Road Network',
               style_function= lambda feature:{'color': color_func(feature), 'weight': 2},
               tooltip=folium.features.GeoJsonTooltip(fields=['id', 'validation_mse', 'validation_mape', 'testing_mse', 'testing_mape'])).add_to(melb_map)

html_title = '<h3 align="center" style="font-size:10px" > <b>Training on 2021-10-05 & Testing on 2021-10-06</b></h3>'
melb_map.get_root().html.add_child(folium.Element(html_title))
melb_map.save('2727.html')


import os
import geopandas as gpd
import pandas as pd
import folium
import numpy as np
from folium.plugins import TimestampedGeoJson
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as ss
import branca
import matplotlib
import branca.colormap as cm



melb_map = folium.Map(location=[-37.840935, 144.946457], zoom_start=12, tiles='stamentoner', fill_opacity=0.001)

DATA_ROOT = os.path.join('../data')
LINK_GDF_PATH = os.path.join(DATA_ROOT, 'link_gdf.geojson')

gdf = gpd.read_file(LINK_GDF_PATH)
corr_ratio_df = tt_time_corr_ratio
gdf = pd.merge(left=gdf, right=corr_ratio_df, left_on='id', right_on='ID', how='inner')

stats_desc = gdf.describe()['HOUR_TT_CORR']
first_quartile, second_quartile, third_quartile= stats_desc['25%'], stats_desc['50%'], stats_desc['75%']
min_value, max_value = stats_desc['min'], stats_desc['max']




colormap = cm.LinearColormap(colors=['darkblue', 'cyan', 'yellow', 'orange', 'red'],
                                    index=[min_value, first_quartile, second_quartile, third_quartile, max_value],
                                    vmin=min_value,
                                    vmax=max_value,
                             caption='<p style="color:red;">This is a paragraph.</p>')


def color_func(feature):
    return colormap(feature['properties']['HOUR_TT_CORR'])

folium.GeoJson(gdf.to_json(),
               name='HOUR_TT_CORR',
               style_function= lambda feature:{'color': color_func(feature), 'weight': 2},
               tooltip=folium.features.GeoJsonTooltip(fields=['id', 'HOUR_TT_CORR'])).add_to(melb_map)

# # gdf['color'] = ss.rankdata(gdf[['DAYOFWEEK_TT_CORR']])
# # color_bar = plt.cm.get_cmap('RdYlGn_r', len(gdf['color'].unique()))

# # folium.GeoJson(gdf.to_json(),
# #                name='DAYOFWEEK_TT_CORR',
# #                style_function= lambda feature:{'color': color_func(feature), 'weight': 2},
# #                tooltip=folium.features.GeoJsonTooltip(fields=['id', 'DAYOFWEEK_TT_CORR'])).add_to(melb_map)

# html_title = '<h3 align="center" style="font-size:10px" > <b>Correlation Ratio between Travel Time and Time (Melbourne City)</b></h3>'
# melb_map.get_root().html.add_child(folium.Element(html_title))
# folium.LayerControl().add_to(melb_map)
melb_map.add_child(colormap)


