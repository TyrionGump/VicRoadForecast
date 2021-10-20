# -*- coding: utf-8 -*-
"""
@File:VicRoadForecast-PyCharm-road_network.py
@Date: 17/9/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import logging
import os
import warnings

import geopandas as gpd
import networkx as nx
import shapely

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class RoadNetwork:
    """

    This class store and process the geographic information of links, local government areas (LGA)
    and places of interests (POI) surrounding links. There are some methods to concat links with
    LGA and POI to help to extract links in particular conditions and add more geographic features
    to links.

    """

    def __init__(self, link_geo_path, lga_geo_path, poi_geo_path, processed_link_path=""):
        """Constructor of Class RoadNetwork

        The input of the this constructor is four GeoJson file paths of links, LGA, POI and Pre-precessed
        links data. The link-processed links data has been filtered and stored in the local and it can be
        loaded without filter useful information again.

        Args:
            link_geo_path: Path to Raw GeoJson file of links
            lga_geo_path: Path to Raw GeoJson file of LGA
            poi_geo_path: Path to Raw GeoJson file of POI
            processed_link_path: Path to GeoJson file of filtered links

        """
        self.logger = logging.getLogger(__name__)
        self.link_gdf = None  # Table (GeoDataFrame) of link features
        self.lga_gdf = gpd.read_file(lga_geo_path)  # Table (GeoDataFrame) of LGA features
        self.poi_gdf = gpd.read_file(poi_geo_path)  # Table (GeoDataFrame) of POI features
        self.link_graph = None  # An object of networkx.DiGraph()
        self.link_neighbours = {}  # Links and their neighbours which intersect with them
        self.regional_link_ids = {}  # LGA and link ids within each LGA

        if os.path.exists(processed_link_path):
            self.link_gdf = gpd.read_file(processed_link_path)
        else:
            self.link_gdf = gpd.read_file(link_geo_path)
            self._filter_info_for_link()

        self._init_link_graph()  # Create a Graph for the road network
        self._init_link_neighbours()  # Find neighbours for each link
        self._init_regional_link_ids()  # Find links within each LGA
        self._count_poi_around_link(buffer_distance=400)

    def get_link_gdf(self):
        return self.link_gdf

    def get_link_neighbours(self):
        return self.link_neighbours

    def get_regional_link_ids(self, region_name='MELBOURNE CITY'):
        return self.regional_link_ids[region_name]

    def _count_poi_around_link(self, buffer_distance=400):
        """Count the number of places of interest(POI) surrounding links within a certain distance.

        Create buffer for each link and find count the number of POI within the buffers. The operation
        of spatial join is implemented under the coordinate system Web Mercator Projection inAustralia (EPSG:3857)
        since the operation is based on the unit of meter instead of degree.

        Args:
            buffer_distance: The distance between the boundaries of buffers and the link.

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

        # Extract id from href of origin and destination columns
        self.link_gdf['origin'] = self.link_gdf['origin'].apply(lambda x: x['id'])
        self.link_gdf['destination'] = self.link_gdf['destination'].apply(lambda x: x['id'])

        # Add LGA information for each link
        self.link_gdf = self.link_gdf.apply(lambda row: self._match_link_with_lga(row), axis=1)

        # Reorder the columns
        self.link_gdf = self.link_gdf[['id', 'origin', 'destination', 'minimum_tt', 'length',
                                       'min_number_of_lanes', 'is_freeway', 'start_lga', 'end_lga', 'geometry']]

    def _match_link_with_lga(self, link_row):
        """Add LGA information for each link

        According to the start point and the end point of each link, find which LGA these points are within.
        Then, add the LGA names of start points and end points for each link.

        Args:
            link_row: A row of GeoDataFrame link_gdf

        """
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
        """Create a graph (A object of networkx.DiGraph()) based on the current road network

        According to the origin id and the destination id of each link, create a graph for this
        network to provide convenience for the future network analysis.

        """
        self.link_graph = nx.DiGraph()
        for idx, row in self.link_gdf.iterrows():
            edge_attr = row.to_dict()
            self.link_graph.add_edge(row['origin'], row['destination'], attr=edge_attr)

    def _init_link_neighbours(self):
        """Find neighbour ids of each link

        Firstly, according to the origin and destination id in the link GeoJson file to find the upstreaming
        and downstreaming links of each link. Secondly, find the intersected links of each link. Finally,
        remove the duplicated neighbours of each link.

        """
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
        """Find link ids in each LGA

        Find links whose start points or end points are within each LGA. Then, use a dictionary to store these
        link ids for each LGA.

        """
        for idx, row in self.lga_gdf.iterrows():
            start_or_end_in_region = self.link_gdf[
                (self.link_gdf['start_lga'] == row['vic_lga__2']) | (self.link_gdf['end_lga'] == row['vic_lga__2'])]
            start_or_end_in_region = start_or_end_in_region['id'].to_list()
            self.regional_link_ids[row['vic_lga__2']] = start_or_end_in_region
