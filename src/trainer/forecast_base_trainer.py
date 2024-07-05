import os
import shutil
import random

import numpy as np
import torch
import arrow
from torch.utils.data import DataLoader

from src.dataset.forecast_parser import KnowAirDataset
from src.utils.config import Config
from src.utils.logger import TrainLogger
from src.utils.metrics import get_metrics
from src.utils.utils import get_mean_std


class ForecastBaseTrainer(object):
    def __init__(self, mode: str = "train"):
        # read config
        self.config = Config()
        # log setting
        self.mode = mode
        self._create_records()
        self.logger = TrainLogger(os.path.join(self.record_dir, 'progress.log')).logger
        # cuda setting
        self._set_seed()
        self.device = self._choose_device()
        # data setting
        self._read_data()
        self.predict_mode = 'group'
        self.model = None

        # result save list
        self.train_loss_list = []
        self.test_loss_list = []
        self.valid_loss_list = []
        self.rmse_list = []
        self.mae_list = []
        self.csi_list = []
        self.pod_list = []
        self.far_list = []
        self.rmse_sud_list = []
        self.mae_sud_list = []
        self.train_time_list = []
        self.test_time_list = []
        self.valid_time_list = []

        self.exp_train_loss_list = []
        self.exp_test_loss_list = []
        self.exp_valid_loss_list = []
        self.exp_rmse_list = []
        self.exp_mae_list = []
        self.exp_csi_list = []
        self.exp_pod_list = []
        self.exp_far_list = []
        self.exp_rmse_sud_list = []
        self.exp_mae_sud_list = []
        self.exp_train_time_list = []
        self.exp_test_time_list = []
        self.exp_valid_time_list = []

    def _set_seed(self):
        if self.config.seed != 0:
            seed = self.config.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _create_records(self):
        """
        Create the records directory for experiments
        :return:
        """
        exp_datetime = arrow.now().format('YYYYMMDDHHmmss')
        self.record_dir = os.path.join(self.config.records_dir, f'{self.config.model_name}_{self.mode}_{exp_datetime}')
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
        elif self.config.dataset_name == 'UrbanAir':
            pass
        else:
            self.logger.error("Unsupported dataset type")
            raise ValueError('Unknown dataset')

    def _choose_device(self):
        """
        Choose train device
        :return: torch device
        """
        return torch.device("cuda" if torch.cuda.is_available() and self.config.device == 'cuda' else "cpu")

    def get_model_info(self):
        """
        statistic model information
        :return:
        """
        raise NotImplementedError('Needs implementation by child class.')

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

    def run_test(self, model_path: str, test_hist_len: int, test_pred_len: int):
        """
        Run the test
        :return:
        """
        raise NotImplementedError('Needs implementation by child class.')

    def run(self):
        """
        do experiment training
        :return:
        """
        # save config file
        try:
            shutil.copy(self.config.config_path, os.path.join(self.record_dir, 'base_config.yaml'))
            shutil.copy(self.config.model_config_path, os.path.join(self.record_dir,
                                                                    f'{self.config.model_name}_config.yaml'))
            self.logger.debug('base_config.yaml copied')
        except IOError as e:
            self.logger.error(f'Error copying config file: {e}')
        # self.get_model_info()
        self.logger.debug('Start experiment...')
        for exp in range(self.config.exp_times):
            self.logger.info(f'Current experiment : {exp}')
            exp_dir = os.path.join(self.record_dir, f'exp_{exp}')
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            # create data loader
            train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                      drop_last=True, pin_memory=True, num_workers=self.config.num_workers)
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      drop_last=True, pin_memory=True, num_workers=self.config.num_workers)
            test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                     drop_last=True, pin_memory=True, num_workers=self.config.num_workers)
            # epoch variants
            best_epoch = 0
            self.train_loss_list = []
            self.test_loss_list = []
            self.valid_loss_list = []
            self.train_time_list = []
            self.test_time_list = []
            self.valid_time_list = []
            self.rmse_list = []
            self.mae_list = []
            self.csi_list = []
            self.pod_list = []
            self.far_list = []
            self.rmse_sud_list = []
            self.mae_sud_list = []

            for epoch in range(self.config.epochs):
                self.logger.debug(f'Experiment time :{exp}, Epoch time : {epoch}')
                train_loss, train_time = self._train(train_loader)
                val_loss, valid_time = self._valid(valid_loader)
                self.logger.info('\n train_loss: %.4f, val_loss: %.4f \n'
                                 'train_city_time: %f, val_city_time: %f,'
                                 % (train_loss, val_loss, train_time, valid_time))
                # End train without the best result in consecutive early_stop epochs
                if epoch - best_epoch > self.config.early_stop and self.config.is_early_stop:
                    self.logger.info('Early stop at epoch {}, best loss = {:.6f}'
                                     .format(epoch, np.min(self.valid_loss_list)))
                    break
                # update val loss
                best_val_loss = np.min(self.valid_loss_list) if len(self.valid_loss_list) > 0 else np.inf
                if val_loss < best_val_loss:
                    best_epoch = epoch
                    # test model
                    test_loss, predict_epoch, label_epoch, test_time = self._test(test_loader)
                    rmse, mae, csi, pod, far, mae_sudden, rmse_sudden = get_metrics(predict_epoch, label_epoch, predict_mode=self.predict_mode)

                    self.logger.info('\n Epoch time: %d, test results: \n'
                                     'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f \n'
                                     'Train time: %f, Val time: %f, Test time: %f \n'
                                     'RMSE: %0.2f, MAE: %0.2f \n'
                                     'CSI: %0.4f, POD: %0.4f, FAR: %0.4f \n'
                                     'RMSE_SUD: %0.2f, MAE_SUD: %0.2f'
                                     % (epoch, train_loss, val_loss, test_loss,
                                        train_time, valid_time, test_time,
                                        rmse, mae, csi, pod, far, rmse_sudden, mae_sudden))
                    self.train_loss_list.append(train_loss)
                    self.valid_loss_list.append(val_loss)
                    self.test_loss_list.append(test_loss)
                    self.test_time_list.append(test_time)
                    self.train_time_list.append(train_time)
                    self.valid_time_list.append(valid_time)
                    self.rmse_list.append(rmse)
                    self.mae_list.append(mae)
                    self.csi_list.append(csi)
                    self.pod_list.append(pod)
                    self.far_list.append(far)
                    self.rmse_sud_list.append(rmse_sudden)
                    self.mae_sud_list.append(mae_sudden)
                    # save model
                    torch.save(self.model.state_dict(),
                               os.path.join(exp_dir, f'model_{self.config.model_name}.pth'))
                    # save prediction and label
                    if self.config.save_npy:
                        np.save(os.path.join(exp_dir, f'predict.npy'), predict_epoch)
                        np.save(os.path.join(exp_dir, f'label.npy'), label_epoch)
                        self.logger.info(f'Save model and results at epoch {epoch}')
                    else:
                        self.logger.info(f'Save model at epoch {epoch}')

            self.logger.info('\n Experiment time: %d, test results: \n'
                             'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f \n'
                             'Train time: %f, Val time: %f, Test time: %f \n'
                             'RMSE: %0.2f, MAE: %0.2f \n'
                             'CSI: %0.4f, POD: %0.4f, FAR: %0.4f \n'
                             'RMSE_SUD: %0.2f, MAE_SUD: %0.2f'
                             % (exp, self.train_loss_list[-1], self.valid_loss_list[-1], self.test_loss_list[-1],
                                self.train_time_list[-1], self.valid_time_list[-1], self.test_time_list[-1],
                                self.rmse_list[-1], self.mae_list[-1], self.csi_list[-1],
                                self.pod_list[-1], self.far_list[-1], self.rmse_sud_list[-1], self.mae_sud_list[-1]))
            self.exp_train_loss_list.append(self.train_loss_list[-1])
            self.exp_test_loss_list.append(self.test_loss_list[-1])
            self.exp_valid_loss_list.append(self.valid_loss_list[-1])
            self.exp_test_time_list.append(self.test_time_list[-1])
            self.exp_train_time_list.append(self.train_time_list[-1])
            self.exp_valid_time_list.append(self.valid_time_list[-1])
            self.exp_rmse_list.append(self.rmse_list[-1])
            self.exp_mae_list.append(self.mae_list[-1])
            self.exp_csi_list.append(self.csi_list[-1])
            self.exp_pod_list.append(self.pod_list[-1])
            self.exp_far_list.append(self.far_list[-1])
            self.exp_rmse_sud_list.append(self.rmse_sud_list[-1])
            self.exp_mae_sud_list.append(self.mae_sud_list[-1])

            # save metrics
            metrics_data = np.concatenate((np.array(self.train_loss_list), np.array(self.valid_loss_list),
                                           np.array(self.test_loss_list), np.array(self.train_time_list),
                                           np.array(self.valid_time_list), np.array(self.test_time_list),
                                           np.array(self.rmse_list),
                                           np.array(self.mae_list), np.array(self.csi_list),
                                           np.array(self.pod_list), np.array(self.far_list),
                                           np.array(self.rmse_sud_list), np.array(self.mae_sud_list)), axis=0)
            np.save(os.path.join(exp_dir, f'exp_{exp}_res.npy'), metrics_data)

        self.logger.info("\n Finished all experiments: \n"
                         'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_train_loss_list)) +
                         'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_valid_loss_list)) +
                         'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_test_loss_list)) +
                         'train_time | mean: %f std: %f\n' % (get_mean_std(self.exp_train_time_list)) +
                         'val_time   | mean: %f std: %f\n' % (get_mean_std(self.exp_valid_time_list)) +
                         'test_time  | mean: %f std: %f\n' % (get_mean_std(self.exp_test_time_list)) +
                         'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_rmse_list)) +
                         'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_mae_list)) +
                         'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_csi_list)) +
                         'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_pod_list)) +
                         'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_far_list)) +
                         'RMSE_SUD   | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_rmse_sud_list)) +
                         'MAE_SUD    | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_mae_sud_list)))
        metrics_data = np.concatenate((np.array(self.exp_train_loss_list), np.array(self.exp_valid_loss_list),
                                       np.array(self.exp_test_loss_list), np.array(self.exp_train_time_list),
                                       np.array(self.exp_valid_time_list), np.array(self.exp_test_time_list),
                                       np.array(self.exp_rmse_list),
                                       np.array(self.exp_mae_list), np.array(self.exp_csi_list),
                                       np.array(self.exp_pod_list), np.array(self.exp_far_list),
                                       np.array(self.exp_rmse_sud_list), np.array(self.exp_mae_sud_list)), axis=0)
        np.save(os.path.join(self.record_dir, 'all_exp_res.npy'), metrics_data)
        self.logger.debug('Experiments finished.')
