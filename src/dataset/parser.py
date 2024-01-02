import numpy as np
import os

import torch

from torch.utils import data
from torch_geometric.utils import to_dense_adj

from src.utils.config import Config, get_time


class GAGNNDataset(data.Dataset):
    # TODO 尚未针对配置文件进行GAGNN数据读取适配
    def __init__(self, config: Config, mode: str = 'train'):
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        self.x = np.load(os.path.join(config.dataset_dir, f'GAGNN_{mode}_x.npy'), allow_pickle=True)
        self.u = np.load(os.path.join(config.dataset_dir, f'GAGNN_{mode}_u.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(config.dataset_dir, f'GAGNN_{mode}_y.npy'), allow_pickle=True)
        self.edge_w = np.load(os.path.join(config.dataset_dir, 'GAGNN_edge_w.npy'), allow_pickle=True)
        self.edge_index = np.load(os.path.join(config.dataset_dir, 'GAGNN_edge_index.npy'), allow_pickle=True)
        self.loc = np.load(os.path.join(config.dataset_dir, 'GAGNN_loc_filled.npy'), allow_pickle=True)
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
    def __init__(self, config: Config, hist_len=1, pred_len=24, mode='train'):
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        # process graph
        self.nodes = np.load(os.path.join(config.dataset_dir, 'KnowAir_loc_filled.npy'))
        self.edge_index = np.load(os.path.join(config.dataset_dir, 'KnowAir_edge_index.npy'))
        self.edge_attr = np.load(os.path.join(config.dataset_dir, 'KnowAir_edge_attr.npy'))
        self.node_num = self.nodes.shape[0]
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]
        # process pm25 and meteo feature
        self.feature = np.load(os.path.join(config.dataset_dir, 'KnowAir_feature.npy'))
        self.pm25 = np.load(os.path.join(config.dataset_dir, 'KnowAir_pm25.npy'))
        self._process_time(config, mode)
        self._process_feature(config)
        # sequence data general preprocess, consider the history data
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

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
        meteo_var = config.meteo_params
        meteo_use = config.used_meteo_params
        meteo_idx = [meteo_var.index(var) for var in meteo_use]
        self.feature = self.feature[:, :, meteo_idx]

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
        return self.pm25[index], self.feature[index]


if __name__ == '__main__':
    know_air = KnowAirDataset(Config())
