import os
from collections import OrderedDict

import numpy as np
from bresenham import bresenham
from scipy.spatial import distance

from src.utils.config import Config


class SimSTGraph:
    def __init__(self, config):
        self.distance_threshold = 3
        self.altitude_threshold = 1200
        self.use_altitude = True
        self.city_path = os.path.join(config.dataset_dir, 'KnowAir_city.txt')
        self.altitude_path = os.path.join(config.dataset_dir, 'KnowAir_altitude.npy')
        self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_num = len(self.nodes)
        self.weight_adj_matrix = self.get_weight_adj()
        self.norm_adj_matrix = self.get_norm_wighted_adjacency_matrix()
        np.save(os.path.join(config.dataset_dir, 'KnowAir_weighted_adj.npy'), self.norm_adj_matrix)

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
        adj[dist <= self.distance_threshold] = 1
        # cal altitude
        for i in range(self.node_num):
            for j in range(i, self.node_num):
                if i != j and adj[i, j] == 1:
                    src_lat, src_lon = self.nodes[i]['lat'], self.nodes[i]['lon']
                    dest_lat, dest_lon = self.nodes[j]['lat'], self.nodes[j]['lon']
                    src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
                    dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
                    points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))
                    altitude_points = self.altitude[points[0], points[1]]
                    altitude_src = self.altitude[src_y, src_x]
                    altitude_dest = self.altitude[dest_y, dest_x]
                    if np.sum(altitude_points - altitude_src > self.altitude_threshold) < 3 and \
                            np.sum(altitude_points - altitude_dest > self.altitude_threshold) < 3:
                        adj[i, j] = 1
                        adj[j, i] = 1
                    else:
                        adj[i, j] = 0
                        adj[j, i] = 0

        dis_std = np.std(dist)
        weight_adj = np.exp(-np.square(dist) / np.square(dis_std)) * adj

        return weight_adj

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
    SimSTGraph(Config())
