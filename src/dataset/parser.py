import numpy as np
import os

import torch

from torch.utils import data

from src.utils.config import Config, get_time
from src.utils.scaler import StandardScaler
from src.utils.utils import np_relu


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
        self.nodes = np.load(os.path.join(config.dataset_dir, 'KnowAir_loc_filled.npy'))
        self.node_attr = np.load(os.path.join(config.dataset_dir, 'KnowAir_node_attr.npy'))
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
        self.wind_mean = self.feature_mean[:, -2:]
        self.wind_std = self.feature_std[:, -2:]
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
        one_hop_loc = np.nonzero(one_hop_loc)[0]
        feature_batch = self.feature[batch*self.node_num:(batch+1)*self.node_num, :]
        feature_one_hop_loc = feature_batch[one_hop_loc]
        # cal static feature
        feature_self = self.feature[index].reshape(1, feature_one_hop_loc.shape[1], -1)
        # max_k_indices = np.argsort(one_hop_loc)[-self.k:][::-1]
        # feature_close = feature_batch[max_k_indices]
        feature_aug = feature_batch[one_hop_loc].mean(axis=0).reshape(1, feature_one_hop_loc.shape[1], -1)
        # static_feature = np.concatenate((feature_self, feature_close, feature_aug), axis=0)
        static_feature = np.concatenate((feature_self, feature_aug), axis=0)
        # cal dynamic feature
        # first self -> one-hop
        out_node_attr = self.node_attr[loc_idx][:, one_hop_loc]
        out_src_node = feature_self.squeeze()
        out_src_wind = out_src_node[:, -2:] * self.wind_std[loc_idx, :] + self.wind_mean[loc_idx, :]
        out_src_wind_speed = out_src_wind[:, 0].reshape((-1, 1))
        out_src_wind_dir = out_src_wind[:, 1].reshape((-1, 1))
        out_node_attr = out_node_attr.repeat(out_src_node.shape[0], axis=0)
        out_tar_dist = out_node_attr[:, :, 0]
        out_tar_dir = out_node_attr[:, :, 1]
        theta = np.abs(out_tar_dir - out_src_wind_dir)
        # out_weight 计划还可以使用在h0初始化中
        out_weight = np_relu(np.tanh(3 * out_src_wind_speed * np.cos(theta) / out_tar_dist))
        out_weight = np.mean(out_weight, axis=-1).reshape((-1, 1))
        out_weight = -out_weight
        # out_feature = []
        # for i in range(out_src_node.shape[0]):
        #     out_node_idx = one_hop_loc[np.nonzero(out_weight[i])[0]]
        #     if out_node_idx.size > 0:
        #         out_feature.append(feature_batch[out_node_idx].mean(axis=0))
        # out_feature = np.stack(out_feature, axis=0)
        # out_feature = out_feature.mean(axis=0).reshape(1, feature_one_hop_loc.shape[1], -1)
        # second one-hop -> self
        in_node_attr = np.transpose(self.node_attr, (1, 0, 2))[loc_idx][:, one_hop_loc]
        in_src_nodes = feature_one_hop_loc.transpose((1, 0, 2))
        in_src_wind = in_src_nodes[:, :, -2:] * self.wind_std[one_hop_loc, :] + self.wind_mean[one_hop_loc, :]
        in_src_wind_speed = in_src_wind[:, :, 0]
        in_src_wind_dir = in_src_wind[:, :, 1]
        in_node_attr = in_node_attr.repeat(in_src_nodes.shape[0], axis=0)
        in_tar_dist = in_node_attr[:, :, 0]
        in_tar_dir = in_node_attr[:, :, 1]
        theta = np.abs(in_tar_dir - in_src_wind_dir)
        # in_weight 计划还可以使用在h0初始化中
        in_weight = np_relu(np.tanh(3 * in_src_wind_speed * np.cos(theta) / in_tar_dist))
        in_weight = np.mean(in_weight, axis=-1).reshape((-1, 1))
        # in_feature = []
        # for i in range(in_src_nodes.shape[0]):
        #     in_node_idx = one_hop_loc[np.nonzero(in_weight[i])[0]]
        #     if in_node_idx.size > 0:
        #         in_feature.append(feature_batch[in_node_idx].mean(axis=0))
        # in_feature = np.stack(in_feature, axis=0)
        # in_feature = in_feature.mean(axis=0).reshape(1, feature_one_hop_loc.shape[1], -1)

        # feature = np.concatenate((static_feature, in_feature, out_feature), axis=0)
        in_out_weight = np.concatenate((in_weight, out_weight), axis=-1)
        return self.pm25[index], static_feature, self.locs[index], self.embedding_feature[index], in_out_weight


if __name__ == '__main__':
    know_air = SimParser(Config())
    know_air.__getitem__(1)
