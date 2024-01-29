import os

import torch
import arrow

from src.dataset.parser import KnowAirDataset
from src.utils.config import Config
from src.utils.logger import TrainLogger


class Trainer(object):
    def __init__(self):
        # read config
        self.config = Config()
        # log setting
        self._create_records()
        self.logger = TrainLogger(os.path.join(self.record_dir, 'progress.log')).logger
        # cuda setting
        self._set_seed()
        self.device = self._choose_device()
        # data setting
        self._read_data()

        # result save list
        self.train_loss_list = []
        self.test_loss_list = []
        self.valid_loss_list = []
        self.rmse_list = []
        self.mae_list = []
        self.csi_list = []
        self.pod_list = []
        self.far_list = []

        self.exp_train_loss_list = []
        self.exp_test_loss_list = []
        self.exp_valid_loss_list = []
        self.exp_rmse_list = []
        self.exp_mae_list = []
        self.exp_csi_list = []
        self.exp_pod_list = []
        self.exp_far_list = []

    def _set_seed(self):
        if self.config.seed != 0:
            torch.manual_seed(self.config.seed)

    def _create_records(self):
        """
        Create the records directory for experiments
        :return:
        """
        exp_datetime = arrow.now().format('YYYYMMDDHHmmss')
        self.record_dir = os.path.join(self.config.records_dir, f'{self.config.model_name}_{exp_datetime}')
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def _read_data(self):
        """
        construct train, valid, and test dataset
        :return: dataset loader
        """
        if self.config.dataset_name == 'KnowAir':
            self.train_dataset = KnowAirDataset(config=self.config, mode='train')
            self.valid_dataset = KnowAirDataset(config=self.config, mode='valid')
            self.test_dataset = KnowAirDataset(config=self.config, mode='test')
            self.city_num = self.train_dataset.node_num
            self.edge_index = self.train_dataset.edge_index
            self.edge_attr = self.train_dataset.edge_attr
            self.city_loc = self.train_dataset.nodes
            self.wind_mean, self.wind_std = self.train_dataset.wind_mean, self.train_dataset.wind_std
            self.pm25_mean, self.pm25_std = self.test_dataset.pm25_mean, self.test_dataset.pm25_std
        else:
            self.logger.error("Unsupported dataset type")
            raise ValueError('Unknown dataset')

    def _choose_device(self):
        """
        Choose train device
        :return: torch device
        """
        return torch.device("cuda" if torch.cuda.is_available() and self.config.device == 'cuda' else "cpu")

    def _train(self, train_loader):
        """
        Train model
        :return: train loss
        """
        raise NotImplementedError('Needs implementation by child class.')

    def _valid(self, valid_loader):
        """
        Validate model
        :return: validation loss
        """
        raise NotImplementedError('Needs implementation by child class.')

    def _test(self, test_loader):
        """
        Test model
        :return: test loss
        """
        raise NotImplementedError('Needs implementation by child class.')

    def run(self):
        """
        do experiment training
        :return:
        """
        raise NotImplementedError('Needs implementation by child class.')
