import os
import datetime

import numpy as np
import torch
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


if __name__ == '__main__':
    load_air_former_data()
