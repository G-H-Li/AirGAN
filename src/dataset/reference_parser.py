import os

import numpy as np
from torch.utils import data

from src.utils.config import ReferConfig
from src.utils.scaler import StandardScaler


class NBSTParser(data.Dataset):
    def __init__(self, config, node_ids: list = None, mode: str = 'train'):
        super(NBSTParser, self).__init__()
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
        self.local_pm25 = self.local_features[:, :, :, 0]
        self._process_feature(config)
        self.features_mean = self.station_features.mean(axis=(0, 1))
        self.features_std = self.station_features.std(axis=(0, 1))
        self.station_nodes_mean = self.station_nodes.mean(axis=(0, 1))
        self.station_nodes_std = self.station_nodes.std(axis=(0, 1))
        self.local_nodes_mean = self.local_nodes.mean(axis=(0, 1))
        self.local_nodes_std = self.local_nodes.std(axis=(0, 1))

        self.pm25_scaler = StandardScaler(mean=self.pm25_mean, std=self.pm25_std)
        self.feature_scaler = StandardScaler(mean=self.features_mean, std=self.features_std)
        self.station_nodes_scaler = StandardScaler(mean=self.station_nodes_mean, std=self.station_nodes_std)
        self.local_nodes_scaler = StandardScaler(mean=self.local_nodes_mean, std=self.local_nodes_std)

        self.local_pm25 = self.pm25_scaler.normalize(self.local_pm25)
        self.station_features = self.feature_scaler.normalize(self.station_features)
        self.station_nodes = self.station_nodes_scaler.normalize(self.station_nodes)
        self.local_nodes = self.local_nodes_scaler.normalize(self.local_nodes)

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

    def __len__(self):
        return len(self.local_pm25)

    def __getitem__(self, index):
        return self.local_pm25[index], self.local_nodes[index], self.station_features[index], self.station_emb_features[index], self.station_nodes[index]


if __name__ == '__main__':
    parser = NBSTParser(ReferConfig(config_filename='refer_base_config.yaml'), [2, 5, 6, 8, 10, 11], mode='train')
    a = parser.__getitem__(1)
    print(a)
