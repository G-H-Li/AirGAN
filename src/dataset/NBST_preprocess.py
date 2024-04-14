import os

import numpy as np
import pandas as pd
import pyproj
import shapely
from pyproj import Transformer
from shapely import Polygon, LineString, MultiPolygon
from shapely.wkt import loads
from geopy.distance import distance

from src.utils.config import Config, ReferConfig


class NBST_preprocess:
    def __init__(self, config):
        self.urban_air = pd.read_csv(os.path.join(config.dataset_dir, 'UrbanAir_processed.csv'))
        self.link_path = os.path.join(config.dataset_dir, 'UrbanAir_link_processed.csv')
        self.poi_path = os.path.join(config.dataset_dir, 'UrbanAir_poi_processed.csv')
        self.link = pd.read_csv(self.link_path)
        self.poi = pd.read_csv(self.poi_path)

        # process link and poi data again
        # self.process_link()
        # self.process_poi()

        self.urban_air_data, self.urban_air_loc_data = self.resort_urban_air()
        # np.save(os.path.join(config.dataset_dir, 'UrbanAir_loc_filled.npy'), self.urban_air_loc_data)
        self.urban_air_data = self.time_split()

        self.calc_link_data()
        self.calc_poi_data()
        self.urban_air_loc_data['other'] = self.urban_air_loc_data['trunk'] + self.urban_air_loc_data['motorway']
        self.urban_air_loc_data = self.urban_air_loc_data.drop(columns=['trunk', 'motorway', 'entertainment', 'food and beverage', 'decoration/furniture markets', ])
        self.urban_air_loc_data = self.urban_air_loc_data.values
        np.save(os.path.join(config.dataset_dir, 'UrbanAir_features.npy'), self.urban_air_data)
        np.save(os.path.join(config.dataset_dir, 'UrbanAir_loc.npy'), self.urban_air_loc_data)

    def resort_urban_air(self):
        self.urban_air['time'] = pd.to_datetime(self.urban_air['time'])
        self.urban_air = self.urban_air.drop(columns=['filled'])
        self.urban_air = self.urban_air.sort_values(['time', 'station_id'])
        self.urban_air['month'] = self.urban_air['time'].map(lambda x: x.month)
        self.urban_air['weekday'] = self.urban_air['time'].map(lambda x: x.weekday())
        self.urban_air['hour'] = self.urban_air['time'].map(lambda x: x.hour)

        urban_air_time_list = self.urban_air['time'].unique().tolist()
        urban_air_station_list = self.urban_air['station_id'].unique().tolist()
        urban_air_loc_data = self.urban_air.head(len(urban_air_station_list))[['longitude', 'latitude']]
        self.urban_air = self.urban_air.drop(columns=['station_id', 'time', 'longitude', 'latitude'])
        urban_air_data = self.urban_air.values
        urban_air_data = urban_air_data.reshape((len(urban_air_time_list), len(urban_air_station_list), -1))
        return urban_air_data, urban_air_loc_data

    def process_poi(self):
        self.poi['geometry'] = self.poi['geometry'].map(lambda g: loads(g))
        self.poi['geometry'] = self.poi['geometry'].map(
            lambda g: Polygon([self._lonlat2xy(lon, lat) for lon, lat in list(g.exterior.coords)])
            if g.geom_type == 'Polygon' else
            MultiPolygon(list(map(lambda sub_g: Polygon(list(map(lambda c: self._lonlat2xy(*c), list(sub_g.exterior.coords)))), list(g.geoms)))))
        self.poi.to_csv(self.poi_path)

    def process_link(self):
        self.link['geometry'] = self.link['geometry'].map(lambda g: loads(g))
        self.link['geometry'] = self.link['geometry'].map(
            lambda g: LineString([self._lonlat2xy(lon, lat) for lon, lat in list(g.coords)]))
        self.link.to_csv(self.link_path)

    def time_split(self):
        time_len, node_num = self.urban_air_data.shape[0], self.urban_air_data.shape[1]
        urban_air_dataset = []
        for i in range(time_len-24+1):
            urban_air_dataset.append(self.urban_air_data[i: i+24])
        urban_air_dataset = np.stack(urban_air_dataset, dtype='float32')
        return urban_air_dataset

    def calc_link_data(self):
        self.link['geometry'] = self.link['geometry'].map(lambda g: loads(g))
        link_type_name = self.link['link_type_name'].unique().tolist()
        for link_type in link_type_name:
            self.urban_air_loc_data[link_type] = 0.
        for n_i, n_row in self.urban_air_loc_data.iterrows():
            node_area = self.create_area_by_center_node(n_row['longitude'], n_row['latitude'])
            node_area = shapely.box(*node_area)
            for i, row in self.link.iterrows():
                inter = row['geometry'].intersection(node_area)
                if inter.length > 0:
                    self.urban_air_loc_data.loc[n_i, row['link_type_name']] += inter.length  # unit: m

    def calc_poi_data(self):
        self.poi['geometry'] = self.poi['geometry'].map(lambda g: loads(g))
        poi_type_name = self.poi['poi_type'].unique().tolist()
        for poi_type in poi_type_name:
            self.urban_air_loc_data[poi_type] = 0.
        for n_i, n_row in self.urban_air_loc_data.iterrows():
            node_area = self.create_area_by_center_node(n_row['longitude'], n_row['latitude'])
            node_area = shapely.box(*node_area)
            for i, row in self.poi.iterrows():
                inter = row['geometry'].intersection(node_area)
                if inter.area > 0:
                    self.urban_air_loc_data.loc[n_i, row['poi_type']] += inter.area  # unit: m^2

    def create_area_by_center_node(self, lon, lat):
        side_half_km = 2 ** 0.5  # km
        max_point = distance(kilometers=side_half_km).destination((lat, lon), bearing=45)
        min_point = distance(kilometers=side_half_km).destination((lat, lon), bearing=-135)
        x_max, y_max = self._lonlat2xy(max_point.longitude, max_point.latitude)
        x_min, y_min = self._lonlat2xy(min_point.longitude, min_point.latitude)
        return x_max, y_max, x_min, y_min

    def _lonlat2xy(self, lon, lat):
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        x2, y2 = transformer.transform(lon, lat)
        return x2, y2


if __name__ == '__main__':
    preprocess = NBST_preprocess(ReferConfig())
