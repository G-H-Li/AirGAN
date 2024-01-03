import os

import numpy as np

from src.utils.config import Config

if __name__ == '__main__':
    a = np.load(os.path.join(Config().dataset_dir, 'KnowAir_feature.npy'))
    print(a[:, -1])
