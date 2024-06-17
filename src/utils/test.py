import os
import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from netCDF4 import Dataset

from src.utils.config import Config
import scipy.sparse as sp

from src.utils.utils import load_pickle


def load_air_former_data():
    a = np.load(os.path.join(Config().dataset_dir, 'KnowAir_assignment.npy'))
    a = torch.from_numpy(a[0])
    row = a.transpose(1, 0)
    row_sums = row.sum(axis=1)
    print(row_sums)


def load_nc():
    nc = Dataset(os.path.join(Config().dataset_dir, 'cams.eaq.vra.ENSa.pm2p5.l0.2016-01.nc'))
    for var in nc.variables.keys():
        data = nc.variables[var][:].data
        print(var, data.shape)
    time = nc.variables['time'][:].data
    lat = nc.variables['lat'][:].data
    lon = nc.variables['lon'][:].data
    pm25 = nc.variables['pm2p5'][:].data
    np.save(os.path.join(Config().dataset_dir, 'europe_pm25.npy'), pm25)


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj():
    nc = os.path.join(Config().dataset_dir, 'adj_mx.pkl')
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(nc)
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]


def detect_change_points(threshold=1.0):
    data = np.load(os.path.join(Config().dataset_dir, 'KnowAir_pm25.npy'))
    # 计算相邻数据点之间的差分
    data = data.squeeze(-1).transpose(1, 0)
    diff = np.diff(data)

    # 检测大于阈值的差分
    sub_points = np.where(np.abs(diff) > threshold, 1, 0)
    large_points = np.where(data[:, 1:] > 75, 1, 0)
    points = sub_points & large_points
    points = np.concatenate([np.zeros((data.shape[0], 1)), points], axis=1)

    change_points = points + 1

    return change_points


def load_pm25():
    data = np.load(os.path.join(Config().dataset_dir, 'KnowAir_pm25.npy'))
    # data = data.squeeze()
    # 绘制统计量线
    x = np.arange(data.shape[1])

    # 计算统计量
    median = np.median(data, axis=0)
    percentile_25 = np.percentile(data, 25, axis=0)
    percentile_75 = np.percentile(data, 75, axis=0)
    percentile_95 = np.percentile(data, 95, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(x, median, label='Median', color='blue')
    plt.plot(x, percentile_25, label='25th Percentile', color='green', linestyle='--')
    plt.plot(x, percentile_75, label='75th Percentile', color='orange', linestyle='--')
    plt.plot(x, percentile_95, label='95th Percentile', color='red', linestyle='--')
    plt.legend()
    plt.title('Statistics of Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    detect_change_points(20)
