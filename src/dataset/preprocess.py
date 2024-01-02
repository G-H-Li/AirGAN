import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import arrow
from bresenham import bresenham
from geopy.distance import geodesic
from scipy.spatial import distance
from metpy.units import units
import metpy.calc as mpcalc
from torch_geometric.utils import dense_to_sparse

from src.utils.config import Config, get_time


class KnowAirGraph:
    def __init__(self, config):
        self.distance_threshold = 3
        self.altitude_threshold = 1200
        self.use_altitude = True
        self.city_path = os.path.join(config.dataset_dir, 'KnowAir_city.txt')
        self.altitude_path = os.path.join(config.dataset_dir, 'KnowAir_altitude.npy')
        # process graph
        self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index, self.edge_attr = self._gen_edges()
        if self.use_altitude:
            self._update_edges()
        coords = []
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        np.save(os.path.join(config.dataset_dir, 'KnowAir_loc_filled.npy'), np.array(coords))
        np.save(os.path.join(config.dataset_dir, 'KnowAir_edge_index.npy'), self.edge_index)
        np.save(os.path.join(config.dataset_dir, 'KnowAir_edge_attr.npy'), self.edge_attr)

    def _load_altitude(self):
        """
        Load altitude dataset
        :return:
        """
        assert os.path.isfile(self.altitude_path)
        altitude = np.load(self.altitude_path)
        return altitude

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

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):
        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))
        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.distance_threshold] = 1
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direction_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direction = mpcalc.wind_direction(u, v)._magnitude

            direction_arr.append(direction)
            dist_kilometer.append(dist_km)

        direction_arr = np.stack(direction_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direction_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.altitude_threshold) < 3 and \
                    np.sum(altitude_points - altitude_dest > self.altitude_threshold) < 3:
                edge_index.append(self.edge_index[:, i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


class KnowAirFeatureAndPm25:
    def __init__(self, config):
        self.node_num = 184
        self.start_time = get_time(config.data_start)
        self.end_time = get_time(config.data_end)
        self.data_start = get_time(config.data_start)
        self.data_end = get_time(config.data_end)
        self.dataset_path = os.path.join(config.dataset_dir, 'KnowAir.npy')
        self.dataset, self.feature, self.pm25 = self._load_know_air_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature(config)
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        np.save(os.path.join(config.dataset_dir, 'KnowAir_feature.npy'), self.feature)
        np.save(os.path.join(config.dataset_dir, 'KnowAir_pm25.npy'), self.pm25)

    def _process_feature(self, config):
        meteo_var = config.meteo_params
        meteo_use = config.meteo_params
        meteo_idx = [meteo_var.index(var) for var in meteo_use]
        self.feature = self.feature[:, :, meteo_idx]

        u = self.feature[:, :, -4] * units.meter / units.second
        v = self.feature[:, :, -3] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direction = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.node_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direction[:, :, None]
                                       ], axis=-1)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx + 1, :]
        self.feature = self.feature[start_idx: end_idx + 1, :]
        self.time_arr = self.time_arr[start_idx: end_idx + 1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+3), 3):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_know_air_npy(self):
        assert os.path.isfile(self.dataset_path)
        dataset = np.load(self.dataset_path)
        return dataset, dataset[:, :, :-1], dataset[:, :, -1:]

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * 3))


if __name__ == '__main__':
    KnowAirFeatureAndPm25(Config())
