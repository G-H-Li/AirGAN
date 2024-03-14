import os

import numpy as np
from torch.utils import data

from src.utils.config import ReferConfig
from src.utils.scaler import StandardScaler


class NBSTParser(data.Dataset):
    def __init__(self, config, node_ids: list = None, mode: str = 'train'):
        if mode not in ['train', 'valid']:
            raise ValueError(f'Invalid mode: {mode}')
        self.mode = mode
        self.nodes = np.load(os.path.join(config.dataset_dir, 'UrbanAir_loc.npy'))
        self.features = np.load(os.path.join(config.dataset_dir, 'UrbanAir_features.npy'))
        self.pm25 = self.features[:, :, :, 0]
        self.time_len = self.features.shape[1]
        self._calc_mean_std()

        self.get_stations(node_ids)
        self.local_features, self.local_nodes, self.station_features, self.station_nodes = self.split_station_and_local(
            node_ids)
        process_pm25 = np.vectorize(self._process_pm25)
        self.pm25_label = self.local_features[:, :, :, 0]
        pm25_features = process_pm25(self.station_features[:, :, :, [0]])
        self.station_features = np.concatenate((pm25_features, self.station_features[:, :, :, 1:]), axis=-1)
        # self.pm25_label = process_pm25(self.local_features[:, :, :, 0])
        # 使用 one-hot 编码将原始张量转换为形状为（128，24，1，6）的张量
        # self.pm25_label = np.eye(6)[self.pm25_label.squeeze()]
        station_dist, station_direction = self._cal_distance(self.local_nodes[:, :, 1], self.local_nodes[:, :, 0],
                                                             self.station_nodes[:, :, 1], self.station_nodes[:, :, 0])
        self.station_dist = np.concatenate((station_dist[:, :, np.newaxis], station_direction[:, :, np.newaxis]),
                                           axis=-1)
        # 去除节点数据中的经纬度信息
        self.local_nodes = self.local_nodes[:, :, 2:]
        self.station_nodes = self.station_nodes[:, :, 2:]

        self._process_feature(config)
        # 去除local的pm25数据
        self.local_emb_features = self.local_emb_features[:, :, :, 1:]
        self.features_mean = self.station_features.mean(axis=(2, 1, 0))
        self.features_std = self.station_features.std(axis=(2, 1, 0))
        self.station_nodes_mean = self.station_nodes.mean(axis=(1, 0))
        self.station_nodes_std = self.station_nodes.std(axis=(1, 0))
        self.station_dist_mean = self.station_dist.mean(axis=(1, 0))
        self.station_dist_std = self.station_dist.std(axis=(1, 0))

        self.pm25_scaler = StandardScaler(mean=self.pm25_mean, std=self.pm25_std)
        self.feature_scaler = StandardScaler(mean=self.features_mean, std=self.features_std)
        self.station_nodes_scaler = StandardScaler(mean=self.station_nodes_mean, std=self.station_nodes_std)
        self.station_dist_scaler = StandardScaler(mean=self.station_dist_mean, std=self.station_dist_std)

        self.pm25_label = self.pm25_scaler.normalize(self.pm25_label)
        self.station_features = self.feature_scaler.normalize(self.station_features)
        self.local_features = self.feature_scaler.normalize(self.local_features)
        self.station_nodes = self.station_nodes_scaler.normalize(self.station_nodes)
        self.local_nodes = self.station_nodes_scaler.normalize(self.local_nodes)
        self.station_dist = self.station_dist_scaler.normalize(self.station_dist)

    def _calc_mean_std(self):
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def get_stations(self, node_ids: list):
        if self.mode == 'train':
            self.nodes = self.nodes[node_ids]
            self.features = self.features[:, :, node_ids]

    def split_station_and_local(self, node_ids):
        node_num = len(node_ids)
        local_features = []
        local_nodes = []
        station_features = []
        station_nodes = []
        if self.mode == 'train':
            for i in range(self.features.shape[0]):
                for j in range(node_num):
                    local_features.append(self.features[i, :, j].reshape(1, self.time_len, 1, -1))
                    local_nodes.append(self.nodes[j].reshape(1, 1, -1))
                    bool_index = np.ones(node_num, dtype=bool)
                    bool_index[j] = False
                    station_features.append(self.features[i, :, bool_index].reshape(1, self.time_len, node_num - 1, -1))
                    station_nodes.append(self.nodes[bool_index, :].reshape(1, node_num - 1, -1))
            local_features = np.concatenate(local_features, axis=0)
            station_features = np.concatenate(station_features, axis=0)
            local_nodes = np.concatenate(local_nodes, axis=0)
            station_nodes = np.concatenate(station_nodes, axis=0)
        elif self.mode == 'valid':
            for i in node_ids:
                local_features.append(self.features[:, :, i].reshape(self.features.shape[0], self.time_len, 1, -1))
                local_nodes.append(self.nodes[i].reshape(1, 1, -1).repeat(self.features.shape[0], axis=0))
            bool_index = np.ones(self.features.shape[2], dtype=bool)
            bool_index[node_ids] = False
            station_features = self.features[:, :, bool_index]
            station_features = station_features.repeat(len(node_ids), axis=0)
            station_nodes = (self.nodes[bool_index, :]
                             .reshape(1, station_features.shape[2], -1).repeat(station_features.shape[0], axis=0))
            local_features = np.concatenate(local_features, axis=0)
            local_nodes = np.concatenate(local_nodes, axis=0)
        return local_features, local_nodes, station_features, station_nodes

    def _process_feature(self, config):
        feature_var = config.feature_params
        feature_use = config.used_feature_params
        norm_feature = [feature_use[i] for i in range(len(config.feature_process)) if config.feature_process[i] == 1]
        em_feature = [feature_use[i] for i in range(len(config.feature_process)) if config.feature_process[i] == 2]
        norm_feature_idx = [feature_var.index(var) for var in norm_feature]
        em_feature_idx = [feature_var.index(var) for var in em_feature]
        self.station_emb_features = np.unique(self.station_features[:, :, :, em_feature_idx], axis=1).squeeze()
        self.station_emb_features = self.station_emb_features.astype(np.int32)
        self.station_features = self.station_features[:, :, :, norm_feature_idx]
        self.local_emb_features = np.unique(self.local_features[:, :, :, em_feature_idx], axis=1)
        self.local_emb_features = self.local_emb_features.astype(np.int32)
        self.local_features = self.local_features[:, :, :, norm_feature_idx]

    def _process_pm25(self, pm25):
        if pm25 <= 35:
            return 0
        elif 35 < pm25 <= 75:
            return 1
        elif 75 < pm25 <= 115:
            return 2
        elif 115 < pm25 <= 150:
            return 3
        elif 150 < pm25 <= 250:
            return 4
        elif 250 < pm25:
            return 5

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

    def __len__(self):
        return len(self.pm25_label)

    def __getitem__(self, index):
        return (self.pm25_label[index], self.station_dist[index],
                self.local_nodes[index], self.local_features[index], self.local_emb_features[index],
                self.station_nodes[index], self.station_features[index], self.station_emb_features[index])


class ADAINParser(data.Dataset):
    def __init__(self, config, node_ids: list = None, mode: str = 'train'):
        super(ADAINParser, self).__init__()
        if mode not in ['train', 'valid']:
            raise ValueError(f'Invalid mode: {mode}')
        self.mode = mode
        self.nodes = np.load(os.path.join(config.dataset_dir, 'UrbanAir_loc.npy'))
        self.features = np.load(os.path.join(config.dataset_dir, 'UrbanAir_features.npy'))
        self.pm25 = self.features[:, :, :, 0]
        self.time_len = self.features.shape[1]
        self._calc_mean_std()

        self.get_stations(node_ids)
        self.local_features, self.local_nodes, self.station_features, self.station_nodes = self.split_station_and_local(
            node_ids)
        self.pm25_label = self.local_features[:, :, :, 0]
        self.local_features = self.local_features[:, :, :, 1:]

        station_dist, station_direction = self._cal_distance(self.local_nodes[:, :, 1], self.local_nodes[:, :, 0],
                                                             self.station_nodes[:, :, 1], self.station_nodes[:, :, 0])
        self.station_dist = np.concatenate((station_dist[:, :, np.newaxis], station_direction[:, :, np.newaxis]), axis=-1)
        self.local_nodes = self.local_nodes[:, :, 2:]
        self.station_nodes = self.station_nodes[:, :, 2:]

        self.station_features_mean = self.station_features.mean(axis=(2, 1, 0))
        self.station_features_std = self.station_features.std(axis=(2, 1, 0))
        self.local_features_mean = self.local_features.mean(axis=(2, 1, 0))
        self.local_features_std = self.local_features.std(axis=(2, 1, 0))
        self.station_nodes_mean = self.station_nodes.mean(axis=(1, 0))
        self.station_nodes_std = self.station_nodes.std(axis=(1, 0))
        self.station_dist_mean = self.station_dist.mean(axis=(1, 0))
        self.station_dist_std = self.station_dist.std(axis=(1, 0))

        self.pm25_scaler = StandardScaler(mean=self.pm25_mean, std=self.pm25_std)
        self.station_feature_scaler = StandardScaler(mean=self.station_features_mean, std=self.station_features_std)
        self.local_feature_scaler = StandardScaler(mean=self.local_features_mean, std=self.local_features_std)
        self.station_nodes_scaler = StandardScaler(mean=self.station_nodes_mean, std=self.station_nodes_std)
        self.station_dist_scaler = StandardScaler(mean=self.station_dist_mean, std=self.station_dist_std)

        self.pm25_label = self.pm25_scaler.normalize(self.pm25_label)
        self.station_features = self.station_feature_scaler.normalize(self.station_features)
        self.local_features = self.local_feature_scaler.normalize(self.local_features)
        self.station_nodes = self.station_nodes_scaler.normalize(self.station_nodes)
        self.local_nodes = self.station_nodes_scaler.normalize(self.local_nodes)
        self.station_dist = self.station_dist_scaler.normalize(self.station_dist)

    def _calc_mean_std(self):
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def get_stations(self, node_ids: list):
        if self.mode == 'train':
            self.nodes = self.nodes[node_ids]
            self.features = self.features[:, :, node_ids]

    def split_station_and_local(self, node_ids):
        node_num = len(node_ids)
        local_features = []
        local_nodes = []
        station_features = []
        station_nodes = []
        if self.mode == 'train':
            for i in range(self.features.shape[0]):
                for j in range(node_num):
                    local_features.append(self.features[i, :, j].reshape(1, self.time_len, 1, -1))
                    local_nodes.append(self.nodes[j].reshape(1, 1, -1))
                    bool_index = np.ones(node_num, dtype=bool)
                    bool_index[j] = False
                    station_features.append(self.features[i, :, bool_index].reshape(1, self.time_len, node_num - 1, -1))
                    station_nodes.append(self.nodes[bool_index, :].reshape(1, node_num - 1, -1))
            local_features = np.concatenate(local_features, axis=0)
            station_features = np.concatenate(station_features, axis=0)
            local_nodes = np.concatenate(local_nodes, axis=0)
            station_nodes = np.concatenate(station_nodes, axis=0)
        elif self.mode == 'valid':
            for i in node_ids:
                local_features.append(self.features[:, :, i].reshape(self.features.shape[0], self.time_len, 1, -1))
                local_nodes.append(self.nodes[i].reshape(1, 1, -1).repeat(self.features.shape[0], axis=0))
            bool_index = np.ones(self.features.shape[2], dtype=bool)
            bool_index[node_ids] = False
            station_features = self.features[:, :, bool_index]
            station_features = station_features.repeat(len(node_ids), axis=0)
            station_nodes = (self.nodes[bool_index, :]
                             .reshape(1, station_features.shape[2], -1).repeat(station_features.shape[0], axis=0))
            local_features = np.concatenate(local_features, axis=0)
            local_nodes = np.concatenate(local_nodes, axis=0)
        return local_features, local_nodes, station_features, station_nodes

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

    def __len__(self):
        return len(self.pm25_label)

    def __getitem__(self, index):
        return (self.pm25_label[index],
                self.local_nodes[index], self.local_features[index],
                self.station_nodes[index], self.station_features[index], self.station_dist[index])


if __name__ == '__main__':
    parser = ADAINParser(ReferConfig(config_filename='refer_base_config.yaml'), [2, 5, 6, 8, 10, 11], mode='train')
    a = parser.__getitem__(1)
    print(a)
