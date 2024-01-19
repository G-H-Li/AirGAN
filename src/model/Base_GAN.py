import torch

from src.utils.config import Config


class Base_GAN(object):
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.generator = None
        self.discriminator = None

    def batch_train(self, pm25_hist, feature_hist, pm25_labels):
        raise NotImplementedError

    def batch_test(self, pm25_hist, feature_hist, pm25_labels):
        raise NotImplementedError

    def batch_valid(self, pm25_hist, feature_hist, pm25_labels):
        raise NotImplementedError
