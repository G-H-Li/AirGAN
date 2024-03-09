import os
import shutil

import arrow
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.dataset.reference_parser import NBSTParser
from src.utils.config import ReferConfig
from src.utils.logger import TrainLogger
from src.utils.metrics import get_metrics, get_class_metrics
from src.utils.utils import get_mean_std


class ReferenceBaseTrainer:
    def __init__(self, mode: str = 'train'):
        self.pm25_scaler = None
        self.config = ReferConfig()
        self.mode = mode
        self._create_records()
        self.logger = TrainLogger(os.path.join(self.record_dir, 'progress.log')).logger
        # cuda setting
        self._set_seed()
        self.device = self._choose_device()

        self.model = None

        # result save list
        self.train_loss_list = []
        self.test_loss_list = []
        self.rmse_list = []
        self.mae_list = []
        self.csi_list = []
        self.pod_list = []
        self.far_list = []

        self.exp_train_loss_list = []
        self.exp_test_loss_list = []
        self.exp_rmse_list = []
        self.exp_mae_list = []
        self.exp_csi_list = []
        self.exp_pod_list = []
        self.exp_far_list = []

    def _create_records(self):
        """
        Create the records directory for experiments
        :return:
        """
        exp_datetime = arrow.now().format('YYYYMMDDHHmmss')
        self.record_dir = os.path.join(self.config.records_dir, f'{self.config.model_name}_{self.mode}_{exp_datetime}')
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def _set_seed(self):
        if self.config.seed != 0:
            torch.manual_seed(self.config.seed)

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

    def _test(self, test_loader):
        """
        Test model
        :return: validation loss
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
        splitter = KFold(n_splits=self.config.exp_times, shuffle=True, random_state=self.config.seed)
        all_stations = list(range(30))
        for exp, (train_ids, test_ids) in enumerate(splitter.split(all_stations)):
            self.logger.info(f'Current experiment : {exp}')
            exp_dir = os.path.join(self.record_dir, f'exp_{exp}')
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            if exp > 0:
                last_exp_filepath = os.path.join(self.record_dir, f'exp_{exp-1}', f'model_{self.config.model_name}.pth')
                self.model.load_state_dict(torch.load(last_exp_filepath))
            # create data loader
            if self.config.dataset_name == 'UrbanAir':
                train_dataset = NBSTParser(config=self.config, node_ids=train_ids, mode='train')
                test_dataset = NBSTParser(config=self.config, node_ids=test_ids, mode='valid')
                self.pm25_scaler = test_dataset.pm25_scaler
                # pm25_mean, pm25_std = test_dataset.pm25_mean, test_dataset.pm25_std
            else:
                self.logger.error("Unsupported dataset type")
                raise ValueError('Unknown dataset')

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                      drop_last=True, num_workers=self.config.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                     drop_last=True, num_workers=self.config.num_workers)
            # epoch variants
            best_epoch = 0
            best_loss = float('inf')
            self.train_loss_list = []
            self.test_loss_list = []
            self.rmse_list = []
            self.mae_list = []
            self.csi_list = []
            self.pod_list = []
            self.far_list = []

            for epoch in range(self.config.epochs):
                self.logger.debug(f'Experiment time :{exp}, Epoch time : {epoch}')
                train_loss = self._train(train_loader)
                self.logger.info('\n train_loss: %.4f' % train_loss)
                # End train without the best result in consecutive early_stop epochs
                if epoch - best_epoch > self.config.early_stop and self.config.is_early_stop:
                    self.logger.info('Early stop at epoch {}, best loss = {:.6f}'
                                     .format(epoch, np.min(self.test_loss_list)))
                    break
                # test model
                test_loss, predict_epoch, label_epoch = self._test(test_loader)
                rmse, mae, csi, pod, far = get_metrics(predict_epoch, label_epoch, predict_mode='city')
                # rmse, mae, accuracy, f1, recall, precision = get_class_metrics(predict_epoch, label_epoch)

                self.logger.info('\n Epoch time: %d, test results: \n'
                                 'Train loss: %0.4f, Test loss: %0.4f \n'
                                 'RMSE: %0.2f, MAE: %0.2f  \n'
                                 'CSI: %0.4f, POD: %0.4f, FAR: %0.4f'
                                 % (epoch, train_loss, test_loss,
                                    rmse, mae, csi, pod, far))
                self.train_loss_list.append(train_loss)
                self.test_loss_list.append(test_loss)
                self.rmse_list.append(rmse)
                self.mae_list.append(mae)
                self.csi_list.append(csi)
                self.pod_list.append(pod)
                self.far_list.append(far)
                # self.acc_list.append(accuracy)
                # self.f1_list.append(f1)
                # self.recall_list.append(recall)
                # self.precision_list.append(precision)
                # save model
                if test_loss < best_loss:
                    # update val loss
                    best_loss = test_loss
                    best_epoch = epoch
                    torch.save(self.model.state_dict(),
                               os.path.join(exp_dir, f'model_{self.config.model_name}.pth'))
                    # save prediction and label
                    if self.config.save_npy:
                        np.save(os.path.join(exp_dir, f'predict.npy'), predict_epoch)
                        np.save(os.path.join(exp_dir, f'label.npy'), label_epoch)
                        self.logger.info(f'Save model and results at epoch {epoch}')
                    else:
                        self.logger.info(f'Save model at epoch {epoch}')

            # self.logger.info('\n Experiment time: %d, test results: \n'
            #                  'Train loss: %0.4f, Test loss: %0.4f \n'
            #                  'RMSE: %0.2f, MAE: %0.2f, ACC: %0.4f\n'
            #                  'F1: %0.4f, RECALL: %0.4f, PRE: %0.4f'
            #                  % (exp, self.train_loss_list[-1], self.test_loss_list[-1],
            #                     self.rmse_list[-1], self.mae_list[-1], self.acc_list[-1],
            #                     self.f1_list[-1], self.recall_list[-1], self.precision_list[-1]))
            self.logger.info('\n Experiment time: %d, test results: \n'
                             'Train loss: %0.4f, Test loss: %0.4f \n'
                             'RMSE: %0.2f, MAE: %0.2f \n'
                             'CSI: %0.4f, POD: %0.4f, FAR: %0.4f'
                             % (exp, self.train_loss_list[-1], self.test_loss_list[-1],
                                self.rmse_list[-1], self.mae_list[-1],
                                self.csi_list[-1], self.pod_list[-1], self.far_list[-1]))
            self.exp_train_loss_list.append(self.train_loss_list[-1])
            self.exp_test_loss_list.append(self.test_loss_list[-1])
            self.exp_rmse_list.append(self.rmse_list[-1])
            self.exp_mae_list.append(self.mae_list[-1])
            self.exp_csi_list.append(self.csi_list[-1])
            self.exp_pod_list.append(self.pod_list[-1])
            self.exp_far_list.append(self.far_list[-1])
            # self.exp_acc_list.append(self.acc_list[-1])
            # self.exp_f1_list.append(self.f1_list[-1])
            # self.exp_recall_list.append(self.recall_list[-1])
            # self.exp_precision_list.append(self.precision_list[-1])

            # save metrics
            metrics_data = np.concatenate((np.array(self.train_loss_list),
                                           np.array(self.test_loss_list),
                                           np.array(self.rmse_list),
                                           np.array(self.mae_list),
                                           np.array(self.csi_list), np.array(self.pod_list),
                                           np.array(self.far_list)), axis=0)
            np.save(os.path.join(exp_dir, f'exp_{exp}_res.npy'), metrics_data)
        #
        # self.logger.info("\n Finished all experiments: \n"
        #                  'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_train_loss_list)) +
        #                  'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_test_loss_list)) +
        #                  'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_rmse_list)) +
        #                  'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_mae_list)) +
        #                  'ACC        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_acc_list)) +
        #                  'F1         | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_f1_list)) +
        #                  'RECALL     | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_recall_list)) +
        #                  'PRE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_precision_list)))
        self.logger.info("\n Finished all experiments: \n"
                         'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_train_loss_list)) +
                         'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_test_loss_list)) +
                         'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_rmse_list)) +
                         'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_mae_list)) +
                         'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_csi_list)) +
                         'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_pod_list)) +
                         'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_far_list)))
        metrics_data = np.concatenate((np.array(self.exp_train_loss_list),
                                       np.array(self.exp_test_loss_list),
                                       np.array(self.exp_rmse_list),
                                       np.array(self.exp_mae_list),
                                       np.array(self.exp_csi_list), np.array(self.exp_pod_list),
                                       np.array(self.exp_far_list)), axis=0)
        np.save(os.path.join(self.record_dir, 'all_exp_res.npy'), metrics_data)
        self.logger.debug('Experiments finished.')
