import os
import sys
from datetime import datetime

import yaml
import arrow

# current file path
current_directory = os.path.dirname(__file__)
# config file's relative path
relative_path = '../../config/'
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
    def __init__(self, config_path: str = config_complete_path, config_filename: str = 'base_config.yaml'):
        self.config_path = os.path.join(config_path, config_filename)
        # read general config
        with open(self.config_path) as f:
            self.base_config = yaml.load(f, Loader=yaml.FullLoader)
        if self.base_config is None:
            sys.exit("No config file found")
        self._read_experiment_config()
        # read model hyperparameters
        if self.model_name in ['MLP', 'GRU', 'LSTM', 'GC_LSTM', 'GAGNN', 'PM25_GNN', 'CW_GAN', 'SimST', 'ADAIN']:
            self.model_config_path = os.path.join(config_path, f'{self.model_name}_config.yaml')
            with open(self.model_config_path) as f:
                self.hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
            if self.hyperparameters is None:
                sys.exit("No model config file found")
        else:
            raise ValueError('Unknown model')
        self._read_filepath_config()
        self._read_data_info_config()
        self._read_hyper_params_config()
        self._read_model_params_config()

    def _read_experiment_config(self):
        """
        Read experiment config
        :return:
        """
        experiment_config = self.base_config['experiment']
        self.device = experiment_config['device']
        # dataset setting
        self.dataset_name = experiment_config['dataset_name']
        self.used_feature_params = experiment_config['used_feature_params']
        self.feature_process = experiment_config['feature_process']
        if len(self.used_feature_params) != len(self.feature_process):
            raise ValueError('feature_process and used_feature_params length do not match')
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
        # train seed
        self.seed = experiment_config['seed']
        # data loader worker num
        self.num_workers = experiment_config['num_workers']

    def _read_hyper_params_config(self):
        """
        Read general hyperparameter config
        :return:
        """
        hyper_params_config = self.hyperparameters['general_hyper_params']
        self.is_early_stop = hyper_params_config['is_early_stop']
        self.early_stop = hyper_params_config['early_stop']
        self.hist_len = hyper_params_config['hist_len']
        self.pred_len = hyper_params_config['pred_len']
        self.batch_size = hyper_params_config['batch_size']
        self.epochs = hyper_params_config['epochs']
        self.exp_times = hyper_params_config['exp_times']
        self.weight_decay = hyper_params_config['weight_decay']
        self.lr = hyper_params_config['lr']

    def _read_model_params_config(self):
        """
        Read model hyperparameter config
        :return:
        """
        if 'model_hyper_params' not in self.hyperparameters:
            return None
        model_params_config = self.hyperparameters['model_hyper_params']
        if 'feature_process' in model_params_config:
            # if model config exist feature process, use model config
            self.feature_process = model_params_config['feature_process']
            if len(self.used_feature_params) != len(self.feature_process):
                raise ValueError('feature_process and used_feature_params length do not match')
        if self.model_name == 'GAGNN':
            config = model_params_config['GAGNN']
            self.group_num = config['group_num']
            self.weight_rate = config['weight_rate']
            self.gnn_hidden = config['gnn_hidden']
            self.gnn_layer = config['gnn_layer']
            self.edge_hidden = config['edge_hidden']
            self.head_nums = config['head_nums']
        elif self.model_name == 'AirFormer':
            config = model_params_config['AirFormer']
            self.dropout = config['dropout']
            self.hidden_channels = config['hidden_channels']
            self.head_nums = config['head_nums']
            self.steps = config['steps']
            self.blocks = config['blocks']
        elif self.model_name == 'CW_GAN':
            config = model_params_config['CW_GAN']
            self.critic_iters = config['critic_iters']
            self.hidden_dim = config['hidden_dim']
        elif self.model_name == 'DGCRN':
            config = model_params_config['DGCRN']
            self.rnn_size = config['rnn_size']
            self.dropout = config['dropout']
            self.gnn_dim = config['gnn_dim']
            self.prop_alpha = config['prop_alpha']
            self.gcn_depth = config['gcn_depth']
            self.clip = config['clip']
            self.tanh_alpha = config['tanh_alpha']
            self.step_size = config['step_size']
            self.node_dim = config['node_dim']
        elif self.model_name == 'SimST':
            config = model_params_config['SimST']
            self.hidden_dim = config['hidden_dim']
            self.dropout = config['dropout']
            self.clip = config['clip']
            self.gru_layers = config['gru_layers']
            self.use_dynamic = config['use_dynamic']
        elif self.model_name == 'ADAIN':
            config = model_params_config['ADAIN']
            self.dropout = config['dropout']

    def _read_data_info_config(self):
        """
        Read dataset info
        :return:
        """
        if self.dataset_name is None:
            raise ValueError('Config dataset_name not exists')
        dataset_info = self.base_config[self.dataset_name]
        if dataset_info is None:
            raise ValueError(f'Dataset {self.dataset_name} info not exists')
        self.data_start = dataset_info['data_start']
        self.data_end = dataset_info['data_end']
        self.city_num = dataset_info['city_num']
        self.feature_params = dataset_info['feature_params']

    def _read_filepath_config(self):
        """
        Read filepath config
        :return:
        """
        if sys.platform == 'win32':
            node_name = 'Local'
        else:
            node_name = os.uname().nodename
        file_dir = self.base_config['filepath'][node_name]
        self.dataset_dir = file_dir['dataset_dir']
        self.records_dir = file_dir['records_dir']
        self.results_dir = file_dir['results_dir']


class ReferConfig:
    def __init__(self, config_path: str = config_complete_path, config_filename: str = 'refer_base_config.yaml'):
        self.config_path = os.path.join(config_path, config_filename)
        # read general config
        with open(self.config_path) as f:
            self.base_config = yaml.load(f, Loader=yaml.FullLoader)
        if self.base_config is None:
            sys.exit("No config file found")
        self._read_experiment_config()
        # read model hyperparameters
        if self.model_name in ['ADAIN', 'NBST']:
            self.model_config_path = os.path.join(config_path, f'{self.model_name}_config.yaml')
            with open(self.model_config_path) as f:
                self.hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
            if self.hyperparameters is None:
                sys.exit("No model config file found")
        else:
            raise ValueError('Unknown model')
        self._read_filepath_config()
        self._read_data_info_config()
        self._read_hyper_params_config()
        self._read_model_params_config()

    def _read_filepath_config(self):
        """
        Read filepath config
        :return:
        """
        if sys.platform == 'win32':
            node_name = 'Local'
        else:
            node_name = os.uname().nodename
        file_dir = self.base_config['filepath'][node_name]
        self.dataset_dir = file_dir['dataset_dir']
        self.records_dir = file_dir['records_dir']
        self.results_dir = file_dir['results_dir']

    def _read_experiment_config(self):
        """
        Read experiment config
        :return:
        """
        experiment_config = self.base_config['experiment']
        self.device = experiment_config['device']
        # dataset setting
        self.dataset_name = experiment_config['dataset_name']
        self.used_feature_params = experiment_config['used_feature_params']
        self.feature_process = experiment_config['feature_process']
        if len(self.used_feature_params) != len(self.feature_process):
            raise ValueError('feature_process and used_feature_params length do not match')
        # progress setting
        self.save_npy = experiment_config['save_npy']
        # model setting
        self.model_name = experiment_config['model_name']
        # train seed
        self.seed = experiment_config['seed']
        # data loader worker num
        self.num_workers = experiment_config['num_workers']

    def _read_hyper_params_config(self):
        """
        Read general hyperparameter config
        :return:
        """
        hyper_params_config = self.hyperparameters['general_hyper_params']
        self.is_early_stop = hyper_params_config['is_early_stop']
        self.early_stop = hyper_params_config['early_stop']
        self.seq_len = hyper_params_config['seq_len']
        self.batch_size = hyper_params_config['batch_size']
        self.epochs = hyper_params_config['epochs']
        self.exp_times = hyper_params_config['exp_times']
        self.weight_decay = hyper_params_config['weight_decay']
        self.lr = hyper_params_config['lr']

    def _read_model_params_config(self):
        """
        Read model hyperparameter config
        :return:
        """
        if 'model_hyper_params' not in self.hyperparameters:
            return None
        model_params_config = self.hyperparameters['model_hyper_params']
        if 'feature_process' in model_params_config:
            # if model config exist feature process, use model config
            self.feature_process = model_params_config['feature_process']
            if len(self.used_feature_params) != len(self.feature_process):
                raise ValueError('feature_process and used_feature_params length do not match')
        if self.model_name == 'ADAIN':
            config = model_params_config['ADAIN']
            self.dropout = config['dropout']
        elif self.model_name == 'NBST':
            config = model_params_config['NBST']
            self.dropout = config['dropout']
            self.hidden_dim = config['hidden_dim']
            # self.gru_layers = config['gru_layers']
            self.clip = config['clip']
            self.alpha = config['alpha']
            self.head_num = config['head_num']
            self.attn_layer = config['attn_layer']

    def _read_data_info_config(self):
        """
        Read dataset info
        :return:
        """
        if self.dataset_name is None:
            raise ValueError('Config dataset_name not exists')
        dataset_info = self.base_config[self.dataset_name]
        if dataset_info is None:
            raise ValueError(f'Dataset {self.dataset_name} info not exists')
        self.data_start = dataset_info['data_start']
        self.data_end = dataset_info['data_end']
        self.city_num = dataset_info['city_num']
        self.feature_params = dataset_info['feature_params']
