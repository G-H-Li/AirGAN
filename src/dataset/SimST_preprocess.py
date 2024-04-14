import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from bresenham import bresenham
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from metpy.units import units
from scipy.spatial import distance
import metpy.calc as mpcalc

from src.dataset.l1ft import l1tf
from src.utils.config import Config


class SimSTGraph:
    def __init__(self, config):
        self.altitude_threshold = None
        self.distance_threshold = None
        self.features_path = None
        self.use_altitude = None
        self.norm_adj_matrix = None
        self.node_attr = None
        self.weight_adj_matrix = None
        self.node_num = None
        self.nodes = None
        self.altitude = None
        self.pm25_path = None
        self.altitude_path = None
        self.city_path = None

        self.config = config
        if self.config.dataset_name == 'KnowAir':
            self.convert_KnowAir_to_SimST()
        elif self.config.dataset_name == 'UrbanAir':
            self.convert_UrbanAir_to_SimST()
        else:
            raise NotImplementedError

    def convert_UrbanAir_to_SimST(self):
        self.distance_threshold = 50
        self.city_path = os.path.join(self.config.dataset_dir, 'UrbanAir_loc_filled.npy')
        self.features_path = os.path.join(self.config.dataset_dir, 'UrbanAir_features.npy')
        features = np.load(self.features_path)
        feature = self._resort_urban_air_features(features)
        pm25 = feature[:, :, [0]]
        wind_speed = feature[:, :, [4]]
        wind_direction = (feature[:, :, [5]] - 1) * 45
        wind_direction = np.where(wind_direction < 0, 0, wind_direction)
        hour = feature[:, :, [8]]
        weekday = feature[:, :, [7]] + 1
        month = feature[:, :, [6]]
        feature = np.concatenate((feature[:, :, 1:4], hour, weekday, month, wind_speed, wind_direction), axis=-1)

        self.nodes = self._gen_nodes_urban_air()
        self.node_num = len(self.nodes)
        self.weight_adj_matrix, self.node_attr = self.get_weight_adj_UrbanAir()
        self.norm_adj_matrix = self.get_norm_wighted_adjacency_matrix()

        np.save(os.path.join(self.config.dataset_dir, 'UrbanAir_pm25.npy'), pm25)
        np.save(os.path.join(self.config.dataset_dir, 'UrbanAir_feature.npy'), feature)
        np.save(os.path.join(self.config.dataset_dir, 'UrbanAir_weighted_adj.npy'), self.norm_adj_matrix)
        np.save(os.path.join(self.config.dataset_dir, 'UrbanAir_node_attr.npy'), self.node_attr)

    def convert_KnowAir_to_SimST(self):
        self.use_altitude = True
        self.distance_threshold = 3
        self.altitude_threshold = 1200
        self.city_path = os.path.join(self.config.dataset_dir, 'KnowAir_city.txt')
        self.altitude_path = os.path.join(self.config.dataset_dir, 'KnowAir_altitude.npy')
        self.pm25_path = os.path.join(self.config.dataset_dir, 'KnowAir_pm25.npy')
        # self.pm25_level = self._cal_pm25_category()
        # self.pm25_trend = self._cal_pm25_trend(0.3)
        if self.use_altitude:
            self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_num = len(self.nodes)
        self.weight_adj_matrix, self.node_attr = self.get_weight_adj()
        self.norm_adj_matrix = self.get_norm_wighted_adjacency_matrix()
        # np.save(os.path.join(config.dataset_dir, 'KnowAir_PM25_level.npy'), self.pm25_level)
        # np.save(os.path.join(config.dataset_dir, 'KnowAir_PM25_trend.npy'), self.pm25_trend)
        np.save(os.path.join(self.config.dataset_dir, 'KnowAir_weighted_adj.npy'), self.norm_adj_matrix)
        np.save(os.path.join(self.config.dataset_dir, 'KnowAir_node_attr.npy'), self.node_attr)

    def _cal_pm25_category(self):
        assert os.path.isfile(self.pm25_path)
        pm25: np.ndarray = np.load(self.pm25_path)
        level_1 = np.where((0 <= pm25) & (pm25 <= 35), 1, 0)
        level_2 = np.where((35 < pm25) & (pm25 <= 75), 2, 0)
        level_3 = np.where((75 < pm25) & (pm25 <= 115), 3, 0)
        level_4 = np.where((115 < pm25) & (pm25 <= 150), 4, 0)
        level_5 = np.where((150 < pm25) & (pm25 <= 250), 5, 0)
        level_6 = np.where(250 < pm25, 6, 0)
        pm25_level = level_1 + level_2 + level_3 + level_4 + level_5 + level_6
        return pm25_level

    def _cal_pm25_trend(self, delta):
        assert os.path.isfile(self.pm25_path)
        pm25 = np.load(self.pm25_path)
        pm25_trend = []
        for i in range(pm25.shape[1]):
            node_pm25 = pm25[:, i, :].squeeze()
            node_trend = l1tf(np.float64(node_pm25), delta)
            pm25_trend.append(node_trend.reshape((-1, 1, 1)))
        pm25_trend = np.concatenate(pm25_trend, axis=1)
        return pm25_trend

    def _load_altitude(self):
        """
        Load altitude dataset
        :return:
        """
        assert os.path.isfile(self.altitude_path)
        altitude = np.load(self.altitude_path)
        return altitude

    def _gen_nodes(self):
        assert os.path.isfile(self.city_path)
        nodes = OrderedDict()
        with open(self.city_path, 'r') as f:
            for line in f:
                idx, city, lon, lat = line.rstrip('\n').split(' ')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                x, y = self._lonlat2xy(lon, lat, True)
                altitude = self.altitude[y, x]
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _gen_nodes_urban_air(self):
        return np.load(self.city_path)

    def _resort_urban_air_features(self, features):
        size = features.shape[0]
        time_len = features.shape[1]
        feature = []
        for i in range(size):
            if i != size - 1:
                feature.append(features[i, 0])
            else:
                for j in range(time_len):
                    feature.append(features[i, j])
        feature = np.stack(feature, axis=0, dtype='float32')
        return feature

    def _lonlat2xy(self, lon, lat, is_altitude):
        """
        Convert longitude and latitude to xy coordinates
        :param lon: longitude
        :param lat: latitude
        :param is_altitude: is altitude calculated
        :return:
        """
        if is_altitude:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def get_weight_adj(self):
        coords = []
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        # cal distance
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        dist_arr = np.zeros((self.node_num, self.node_num, 1), dtype=np.float32)
        direction_arr = np.zeros((self.node_num, self.node_num, 1), dtype=np.float32)
        adj[dist <= self.distance_threshold] = 1
        # cal altitude
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j and adj[i, j] == 1:
                    src_lat, src_lon = self.nodes[i]['lat'], self.nodes[i]['lon']
                    dest_lat, dest_lon = self.nodes[j]['lat'], self.nodes[j]['lon']
                    src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
                    dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)

                    src_location = (src_lat, src_lon)
                    dest_location = (dest_lat, dest_lon)
                    dist_km = geodesic(src_location, dest_location).kilometers
                    v, u = src_lat - dest_lat, src_lon - dest_lon

                    u = u * units.meter / units.second
                    v = v * units.meter / units.second
                    direction = mpcalc.wind_direction(u, v)._magnitude

                    points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))
                    altitude_points = self.altitude[points[0], points[1]]
                    altitude_src = self.altitude[src_y, src_x]
                    altitude_dest = self.altitude[dest_y, dest_x]
                    if np.sum(altitude_points - altitude_src > self.altitude_threshold) < 3 and \
                            np.sum(altitude_points - altitude_dest > self.altitude_threshold) < 3:
                        adj[i, j] = 1
                        dist_arr[i, j, 0] = dist_km
                        direction_arr[i, j, 0] = direction
                    else:
                        adj[i, j] = 0

        dis_std = np.std(dist)
        weight_adj = np.exp(-np.square(dist) / np.square(dis_std)) * adj

        node_attr = np.concatenate((dist_arr, direction_arr), axis=-1)

        return weight_adj, node_attr

    def get_norm_wighted_adjacency_matrix(self):
        unit_m = np.eye(self.weight_adj_matrix.shape[0])
        mat = self.weight_adj_matrix - unit_m  # 原文是mat + 单位矩阵,为了保留自身权重，但是此处self.weight_adj_matrix,已经包含了单位矩阵值
        degrees = np.sum(mat, axis=1)
        degree_matrix = np.diag(degrees)

        v, Q = np.linalg.eig(degree_matrix)  # v特征值，Q特征向量
        V = np.diag(v ** (-0.5))
        B = Q @ V @ np.linalg.inv(Q)

        norm_m = B @ mat @ B
        return norm_m

    def get_weight_adj_UrbanAir(self):
        # cal distance
        distance = []
        direction = []
        for i in range(self.node_num):
            dist, direc = self._cal_distance(self.nodes[i, 1], self.nodes[i, 0], self.nodes[:, 1], self.nodes[:, 0])
            distance.append(dist)
            direction.append(direc)
        dist = np.array(distance)
        direc = np.array(direction)
        node_attr = np.concatenate((dist[..., np.newaxis], direc[..., np.newaxis]), axis=-1)
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.distance_threshold] = 1
        dis_std = np.std(dist)
        weight_adj = np.exp(-np.square(dist) / np.square(dis_std)) * adj

        return weight_adj, node_attr

    def _cal_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # 地球平均半径（单位：公里）

        # 将经纬度转换为弧度
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # 计算经纬度差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # 使用 Haversine 公式计算距离
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        # 计算方向角度
        y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
        direction = np.degrees(np.arctan2(y, x))
        direction = np.where(direction < 0, direction + 360, direction)  # 确保方向角在 0 到 360 度之间

        return distance, direction

if __name__ == '__main__':
    a = SimSTGraph(Config())
