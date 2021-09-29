"""
@file:VicRoadForecast-PyCharm-road_network.py
@time: 17/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""

import logging
import os
import warnings

import geopandas as gpd
import networkx as nx
import numpy
import pandas as pd
import shapely

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class RoadNetwork:
    def __init__(self, link_geo_path, lga_geo_path, poi_geo_path, processed_link_path=""):
        self.logger = logging.getLogger(__name__)
        self.link_gdf = None
        self.lga_gdf = gpd.read_file(lga_geo_path)
        self.poi_gdf = gpd.read_file(poi_geo_path)
        self.link_graph = None
        self.link_neighbours = {}
        self.regional_link_ids = {}

        if os.path.exists(processed_link_path):
            self.link_gdf = gpd.read_file(processed_link_path)
        else:
            self.link_gdf = gpd.read_file(link_geo_path)
            self._filter_info_for_link()

        self._init_link_graph()
        self._init_link_neighbours()
        self._init_regional_link_ids()

    def get_link_gdf(self):
        return self.link_gdf

    def get_link_neighbours(self):
        return self.link_neighbours

    def get_regional_link_ids(self, region_name='MELBOURNE CITY'):
        return self.regional_link_ids[region_name]

    def count_poi_around_link(self, buffer_distance=400):
        """Count the number of places of interest(POI) surrounding links within a certain distance.

        Create buffer for each link and find count the number of POI within the buffers. The operation
        of spatial join is implemented under the coordinate system Web Mercator Projection inAustralia (EPSG:3857)
        since the operation is based on the unit of meter instead of degree.

        Args:
            buffer_distance: The distance between the boundaries of buffers and the link.

        Returns:

        """
        research_buffer_gdf = self.link_gdf.copy(deep=True)
        research_buffer_gdf = research_buffer_gdf.to_crs(epsg=3857)
        research_buffer_gdf['geometry'] = self.link_gdf.to_crs(epsg=3857).buffer(distance=buffer_distance)
        buffer_stats = gpd.sjoin(research_buffer_gdf, self.poi_gdf.to_crs(epsg=3857), how='left', op='contains')
        self.link_gdf['poi_num_{}m'.format(buffer_distance)] = list(buffer_stats.groupby('id').size())

    def _filter_info_for_link(self):
        """Filter useful information from the raw geojson file of link.

        Firstly, remove the columns which are useless for geometry. Secondly, remove the columns in which all
        vales are null. Then, extract the ids of origin and destination of each link from href. Finally, find the
        local government area(LGA) for each link.

        """
        # These attributes is not helpful in this section
        check_res = ['latest_stats', 'href', 'enabled']

        # Find columns in which all values are null
        for i in self.link_gdf.columns:
            if self.link_gdf[i].isnull().all():
                check_res.append(i)
        self.link_gdf.drop(columns=check_res, inplace=True)

        self.link_gdf['origin'] = self.link_gdf['origin'].apply(lambda x: x['id'])
        self.link_gdf['destination'] = self.link_gdf['destination'].apply(lambda x: x['id'])

        self.link_gdf = self.link_gdf.apply(lambda row: self._match_link_with_lga(row), axis=1)
        self.link_gdf = self.link_gdf[['id', 'origin', 'destination', 'minimum_tt', 'length',
                                       'min_number_of_lanes', 'is_freeway', 'start_lga', 'end_lga', 'geometry']]

    def _match_link_with_lga(self, link_row):
        start_point = shapely.geometry.Point(link_row['geometry'].coords[0])
        end_point = shapely.geometry.Point(link_row['geometry'].coords[-1])
        start_lga = None
        end_lga = None

        for idx, row in self.lga_gdf.iterrows():
            if row['geometry'].intersects(start_point):
                start_lga = row['vic_lga__2']
            if row['geometry'].intersects(end_point):
                end_lga = row['vic_lga__2']
        link_row['start_lga'] = start_lga
        link_row['end_lga'] = end_lga

        return link_row

    def _init_link_graph(self):
        self.link_graph = nx.DiGraph()
        for idx, row in self.link_gdf.iterrows():
            edge_attr = row.to_dict()
            self.link_graph.add_edge(row['origin'], row['destination'], attr=edge_attr)

    def _init_link_neighbours(self):
        intersected_links = gpd.sjoin(self.link_gdf, self.link_gdf, how='left', op='intersects')
        for idx, row in self.link_gdf.iterrows():
            self.link_neighbours[row['id']] = []
            self.link_neighbours[row['id']] = intersected_links[intersected_links['id_left']
                                                                == row['id']]['id_right'].to_list()
            up_edges = self.link_graph.in_edges(row['origin'])
            down_edges = self.link_graph.out_edges(row['destination'])
            intersects_edges_ids = []

            for e in up_edges:
                intersects_edges_ids.append(self.link_graph[e[0]][e[1]]['attr']['id'])

            for e in down_edges:
                intersects_edges_ids.append(self.link_graph[e[0]][e[1]]['attr']['id'])
            self.link_neighbours[row['id']].extend(intersects_edges_ids)
            self.link_neighbours[row['id']] = list(set(self.link_neighbours[row['id']]))
            self.link_neighbours[row['id']].remove(row['id'])

    def _init_regional_link_ids(self):
        for idx, row in self.lga_gdf.iterrows():
            start_or_end_in_region = self.link_gdf[(self.link_gdf['start_lga'] == row['vic_lga__2']) | (self.link_gdf['end_lga'] == row['vic_lga__2'])]
            start_or_end_in_region = start_or_end_in_region['id'].to_list()
            self.regional_link_ids[row['vic_lga__2']] = start_or_end_in_region

    def build_neighbour_graph(self):
        size = max(self.link_neighbours.keys()) + 1
        matrix = numpy.zeros(shape=(size, size))
        for key in self.link_neighbours.keys():
            for value in self.link_neighbours[key]:
                matrix[key][value] = 1
        neighbour_df = pd.DataFrame(data=matrix)
        neighbour_df.to_csv('../data/neighbour.csv', index=False)




