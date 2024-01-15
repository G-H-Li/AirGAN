import os
import datetime

import numpy as np
import torch
from netCDF4 import Dataset

from src.utils.config import Config


def load_air_former_data():
    a = np.load(os.path.join(Config().dataset_dir, 'assignment.npy'))
    a = torch.from_numpy(a[:, 0])
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


if __name__ == '__main__':
    load_nc()
