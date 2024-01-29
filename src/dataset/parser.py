import numpy as np
import os

import torch

from torch.utils import data

from src.utils.config import Config, get_time
from src.utils.scaler import StandardScaler


class CityAirDataset(data.Dataset):
    # TODO 尚未针对配置文件进行GAGNN数据读取适配
    def __init__(self, config: Config, mode: str = 'train'):
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        self.x = np.load(os.path.join(config.dataset_dir, f'CityAir_{mode}_x.npy'), allow_pickle=True)
        self.u = np.load(os.path.join(config.dataset_dir, f'CityAir_{mode}_u.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(config.dataset_dir, f'CityAir_{mode}_y.npy'), allow_pickle=True)
        self.edge_w = np.load(os.path.join(config.dataset_dir, 'CityAir_edge_w.npy'), allow_pickle=True)
        self.edge_index = np.load(os.path.join(config.dataset_dir, 'CityAir_edge_index.npy'), allow_pickle=True)
        self.loc = np.load(os.path.join(config.dataset_dir, 'CityAir_loc_filled.npy'), allow_pickle=True)
        self.loc = self.loc.astype(np.float64)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        x = x.transpose(0, 1)
        y = torch.FloatTensor(self.y[index])
        y = y.transpose(0, 1)
        u = torch.tensor(self.u[index])
        edge_index = torch.tensor(self.edge_index)
        # edge_index = edge_index.expand((x.size[0],edge_index.size[0],edge_index.size[1]))
        edge_w = torch.FloatTensor(self.edge_w)
        # edge_w = edge_w.expand((x.size[0],edge_w.size[0]))
        loc = torch.FloatTensor(self.loc)

        return [x, u, y, edge_index, edge_w, loc]

    def __len__(self):
        return self.x.shape[0]


class KnowAirDataset(data.Dataset):
    """
    Preprocess for KnowAir dataset
    """
    def __init__(self, config: Config, mode='train'):
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        # process graph
        self.nodes = np.load(os.path.join(config.dataset_dir, 'KnowAir_loc_filled.npy'))
        self.edge_index = np.load(os.path.join(config.dataset_dir, 'KnowAir_edge_index.npy'))
        self.edge_attr = np.load(os.path.join(config.dataset_dir, 'KnowAir_edge_attr.npy'))
        self.node_num = self.nodes.shape[0]
        self.edge_num = self.edge_index.shape[1]
        # self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]
        # process pm25 and meteo feature
        self.feature = np.load(os.path.join(config.dataset_dir, 'KnowAir_feature.npy'))
        self.pm25 = np.load(os.path.join(config.dataset_dir, 'KnowAir_pm25.npy'))
        self._process_time(config, mode)
        self._process_feature(config)
        self._calc_mean_std()
        # sequence data general preprocess, consider the history data
        seq_len = config.hist_len + config.pred_len
        self._add_time_dim(seq_len)
        # data preprocess
        self.pm25_scaler = StandardScaler(mean=self.pm25_mean, std=self.pm25_std)
        self.feature_scaler = StandardScaler(mean=self.feature_mean, std=self.feature_std)
        self.pm25 = self.pm25_scaler.normalize(self.pm25)
        self.feature = self.feature_scaler.normalize(self.feature)

    def _add_time_dim(self, seq_len):
        def _add_t(arr, seq_length):
            t_len = arr.shape[0]
            assert t_len > seq_length
            arr_ts = []
            for i in range(seq_length, t_len):
                arr_t = arr[i - seq_length:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0, 1))
        self.feature_std = self.feature.std(axis=(0, 1))
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _process_feature(self, config):
        feature_var = config.feature_params
        feature_use = config.used_feature_params
        norm_feature = [feature_use[i] for i in range(len(config.feature_process)) if config.feature_process[i] == 1]
        em_feature = [feature_use[i] for i in range(len(config.feature_process)) if config.feature_process[i] == 2]
        norm_feature_idx = [feature_var.index(var) for var in norm_feature]
        em_feature_idx = [feature_var.index(var) for var in em_feature]
        self.embedding_feature = np.unique(self.feature[:, :, em_feature_idx], axis=1).squeeze()
        self.feature = self.feature[:, :, norm_feature_idx]

    def _process_time(self, config, mode):
        self.start_time = get_time(config.__getattribute__(f'{mode}_start'))
        self.end_time = get_time(config.__getattribute__(f'{mode}_end'))
        self.data_start = get_time(config.data_start)
        self.data_end = get_time(config.data_end)
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx + 1, :]
        self.feature = self.feature[start_idx: end_idx + 1, :]

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * 3))

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.embedding_feature[index]


class SimParser(data.Dataset):
    def __init__(self, config: Config, mode='train'):
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        self.k = config.k
        self.nodes = np.load(os.path.join(config.dataset_dir, 'KnowAir_loc_filled.npy'))
        self.feature = np.load(os.path.join(config.dataset_dir, 'KnowAir_feature.npy'))
        self.pm25 = np.load(os.path.join(config.dataset_dir, 'KnowAir_pm25.npy'))
        self.adj_weighted = np.load(os.path.join(config.dataset_dir, 'KnowAir_weighted_adj.npy'))
        self.node_num = self.adj_weighted.shape[0]
        self._process_time(config, mode)
        self._process_feature(config)
        seq_len = config.hist_len + config.pred_len
        self._add_time_dim(seq_len)
        self._calc_mean_std()
        # data preprocess
        self.pm25_scaler = StandardScaler(mean=self.pm25_mean, std=self.pm25_std)
        self.feature_scaler = StandardScaler(mean=self.feature_mean, std=self.feature_std)
        self.loc_scaler = StandardScaler(mean=self.loc_mean, std=self.loc_std)
        self.pm25 = self.pm25_scaler.normalize(self.pm25)
        self.feature = self.feature_scaler.normalize(self.feature)
        self.loc = self.loc_scaler.normalize(self.nodes)

        self.idx = [np.arange(self.node_num) for _ in range(self.pm25.shape[0])]
        self.idx = np.stack(self.idx, axis=0)
        self.idx = np.reshape(self.idx, (-1, 1))

        self.locs = [self.loc for _ in range(self.pm25.shape[0])]
        self.locs = np.stack(self.locs, axis=0)
        self.locs = np.reshape(self.locs, (-1, self.locs.shape[-1]))

        self.embedding_feature = [self.embedding_feature for _ in range(self.node_num)]
        self.embedding_feature = np.stack(self.embedding_feature, axis=1)
        self.embedding_feature = self.embedding_feature.reshape((-1, seq_len, self.embedding_feature.shape[-1]))

        self.pm25 = np.transpose(self.pm25, axes=(0, 2, 1, 3)).reshape((-1, seq_len, self.pm25.shape[-1]))
        self.feature = np.transpose(self.feature, axes=(0, 2, 1, 3)).reshape((-1, seq_len, self.feature.shape[-1]))

    def _process_time(self, config, mode):
        self.start_time = get_time(config.__getattribute__(f'{mode}_start'))
        self.end_time = get_time(config.__getattribute__(f'{mode}_end'))
        self.data_start = get_time(config.data_start)
        self.data_end = get_time(config.data_end)
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx + 1, :]
        self.feature = self.feature[start_idx: end_idx + 1, :]

    def _process_feature(self, config):
        feature_var = config.feature_params
        feature_use = config.used_feature_params
        norm_feature = [feature_use[i] for i in range(len(config.feature_process)) if config.feature_process[i] == 1]
        em_feature = [feature_use[i] for i in range(len(config.feature_process)) if config.feature_process[i] == 2]
        norm_feature_idx = [feature_var.index(var) for var in norm_feature]
        em_feature_idx = [feature_var.index(var) for var in em_feature]
        self.embedding_feature = np.unique(self.feature[:, :, em_feature_idx], axis=1).squeeze()
        self.feature = self.feature[:, :, norm_feature_idx]

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * 3))

    def _add_time_dim(self, seq_len):
        def _add_t(arr, seq_length):
            t_len = arr.shape[0]
            assert t_len > seq_length
            arr_ts = []
            for i in range(seq_length, t_len):
                arr_t = arr[i - seq_length:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.embedding_feature = _add_t(self.embedding_feature, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0, 1))
        self.feature_std = self.feature.std(axis=(0, 1))
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()
        self.loc_mean = self.nodes.mean(axis=0)
        self.loc_std = self.nodes.std(axis=0)

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        loc_idx = self.idx[index]
        batch = index // self.node_num
        one_hop_loc = self.adj_weighted[loc_idx].squeeze()
        max_k_indices = np.argsort(one_hop_loc)[-self.k:][::-1]
        one_hop_loc = np.nonzero(one_hop_loc)

        feature = [self.feature[index]]
        feature_batch = self.feature[batch*self.node_num:(batch+1)*self.node_num, :]
        feature_close = feature_batch[max_k_indices]
        feature_close = [feature_close[i] for i in range(feature_close.shape[0])]
        feature_aug = feature_batch[one_hop_loc].mean(axis=0)
        feature += feature_close
        feature.append(feature_aug)
        feature = np.stack(feature, axis=0)
        return self.pm25[index], feature, self.locs[index], self.embedding_feature[index]


if __name__ == '__main__':
    know_air = SimParser(Config())
