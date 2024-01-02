import os
import sys
from datetime import datetime

import yaml
import arrow

# current file path
current_directory = os.path.dirname(__file__)
# config file's relative path
relative_path = '..\\..\\config.yaml'
# construct the config file complete path
config_complete_path = os.path.join(current_directory, relative_path)


def get_time(time_yaml):
    arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
    return arrow_time


class Config(object):
    """
    Configuration class
    Use to read the config
    """
    def __init__(self, config_path: str = config_complete_path):
        self.config_path = config_path
        with open(self.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        if self.config is None:
            sys.exit("No config file found")
        self._read_experiment_config()
        self._read_filepath_config()
        self._read_data_info_config()
        self._read_hyper_params_config()
        self._read_model_params_config()

    def _read_experiment_config(self):
        """
        Read experiment config
        :return:
        """
        experiment_config = self.config['experiment']
        self.device = experiment_config['device']
        # dataset setting
        self.dataset_name = experiment_config['dataset_name']
        self.used_meteo_params = experiment_config['used_meteo_params']
        self.train_start = experiment_config['train_start']
        self.train_end = experiment_config['train_end']
        self.test_start = experiment_config['test_start']
        self.test_end = experiment_config['test_end']
        self.valid_start = experiment_config['valid_start']
        self.valid_end = experiment_config['valid_end']
        # progress setting
        self.save_npy = experiment_config['save_npy']
        # model setting
        self.model_name = experiment_config['model_name']

    def _read_hyper_params_config(self):
        """
        Read general hyperparameter config
        :return:
        """
        hyper_params_config = self.config['general_hyper_params']
        self.hist_len = hyper_params_config['hist_len']
        self.pred_len = hyper_params_config['pred_len']
        self.batch_size = hyper_params_config['batch_size']
        self.epochs = hyper_params_config['epochs']
        self.exp_times = hyper_params_config['exp_times']
        self.weight_decay = hyper_params_config['weight_decay']
        self.weight_rate = hyper_params_config['weight_rate']
        self.lr = hyper_params_config['lr']
        self.early_stop = hyper_params_config['early_stop']

    def _read_model_params_config(self):
        """
        Read model hyperparameter config
        :return:
        """
        if self.model_name is None:
            raise ValueError('Config model_name not exists')
        model_params_config = self.config['model_hyper_params']
        if self.model_name == 'GAGNN':
            self.group_num = model_params_config['group_num']
            self.gnn_layers = model_params_config['gnn_layers']
            self.gnn_hidden = model_params_config['gnn_hidden']

    def _read_data_info_config(self):
        """
        Read dataset info
        :return:
        """
        if self.dataset_name is None:
            raise ValueError('Config dataset_name not exists')
        dataset_info = self.config[self.dataset_name]
        if dataset_info is None:
            raise ValueError(f'Dataset {self.dataset_name} info not exists')
        self.data_start = dataset_info['data_start']
        self.data_end = dataset_info['data_end']
        self.city_num = dataset_info['city_num']
        self.meteo_params = dataset_info['meteo_params']

    def _read_filepath_config(self):
        """
        Read filepath config
        :return:
        """
        if sys.platform == 'win32':
            node_name = 'Local'
        else:
            node_name = os.uname().nodename
        file_dir = self.config['filepath'][node_name]
        self.dataset_dir = file_dir['dataset_dir']
        self.records_dir = file_dir['records_dir']
        self.results_dir = file_dir['results_dir']
