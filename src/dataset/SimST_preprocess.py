import os
from collections import OrderedDict

import numpy as np
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
        self.distance_threshold = 3
        self.altitude_threshold = 1200
        self.use_altitude = True
        self.city_path = os.path.join(config.dataset_dir, 'KnowAir_city.txt')
        self.altitude_path = os.path.join(config.dataset_dir, 'KnowAir_altitude.npy')
        self.pm25_path = os.path.join(config.dataset_dir, 'KnowAir_pm25.npy')
        # self.pm25_level = self._cal_pm25_category()
        # self.pm25_trend = self._cal_pm25_trend(0.3)
        self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_num = len(self.nodes)
        self.weight_adj_matrix, self.node_attr = self.get_weight_adj()
        self.norm_adj_matrix = self.get_norm_wighted_adjacency_matrix()
        # np.save(os.path.join(config.dataset_dir, 'KnowAir_PM25_level.npy'), self.pm25_level)
        # np.save(os.path.join(config.dataset_dir, 'KnowAir_PM25_trend.npy'), self.pm25_trend)
        np.save(os.path.join(config.dataset_dir, 'KnowAir_weighted_adj.npy'), self.norm_adj_matrix)
        np.save(os.path.join(config.dataset_dir, 'KnowAir_node_attr.npy'), self.node_attr)

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


if __name__ == '__main__':
    a = SimSTGraph(Config())
