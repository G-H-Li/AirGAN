import os.path
import shutil

import arrow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.parser import KnowAirDataset
from src.model.GC_LSTM import GC_LSTM
from src.model.GRU import GRU
from src.model.LSTM import LSTM
from src.model.MLP import MLP
from src.model.PM25_GNN import PM25_GNN
from src.utils.config import Config
from src.utils.logger import TrainLogger


def get_metrics(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:, :, :, 0].transpose((0, 2, 1))
    label = label_epoch[:, :, :, 0].transpose((0, 2, 1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


class Trainer:
    """
    Trainer class
    General purpose training:
    1. load setting
    2. load dataset
    3. train model
    4. test model
    """

    def __init__(self):
        # read config
        self.config = Config()
        # log setting
        self._create_records()
        self.logger = TrainLogger(os.path.join(self.record_dir, 'progress.log')).logger
        # cuda setting
        torch.set_num_threads(1)
        self.device = self._choose_device()
        # data setting
        self._read_data()
        # model setting
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        # train setting
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                             lr=self.config.lr,
                                             weight_decay=self.config.weight_decay)
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

    def _choose_device(self):
        """
        Choose train device
        :return: torch device
        """
        return torch.device("cuda" if torch.cuda.is_available() and self.config.device == 'cuda' else "cpu")

    def _read_data(self):
        """
        construct train, valid, and test dataset
        :return: dataset loader
        """
        if self.config.dataset_name == 'KnowAir':
            self.train_dataset = KnowAirDataset(config=self.config)
            self.valid_dataset = KnowAirDataset(config=self.config)
            self.test_dataset = KnowAirDataset(config=self.config)
            self.in_dim = self.train_dataset.feature.shape[-1] + self.train_dataset.pm25.shape[-1]
            self.city_num = self.train_dataset.node_num
            self.edge_index = self.train_dataset.edge_index
            self.edge_attr = self.train_dataset.edge_attr
            self.wind_mean, self.wind_std = self.train_dataset.wind_mean, self.train_dataset.wind_std
            self.pm25_mean, self.pm25_std = self.test_dataset.pm25_mean, self.test_dataset.pm25_std
        else:
            self.logger.error("Unsupported dataset type")
            raise ValueError('Unknown dataset')

    def _get_model(self):
        """
        construct model
        :return: model object
        """
        if self.config.model_name == 'MLP':
            return MLP(self.config.hist_len,
                       self.config.pred_len,
                       self.in_dim)
        elif self.config.model_name == 'LSTM':
            return LSTM(self.config.hist_len,
                        self.config.pred_len,
                        self.in_dim,
                        self.city_num,
                        self.config.batch_size,
                        self.device)
        elif self.config.model_name == 'GRU':
            return GRU(self.config.hist_len,
                       self.config.pred_len,
                       self.in_dim,
                       self.city_num,
                       self.config.batch_size,
                       self.device)
        elif self.config.model_name == 'GC_LSTM':
            return GC_LSTM(self.config.hist_len,
                           self.config.pred_len,
                           self.in_dim,
                           self.city_num,
                           self.config.batch_size,
                           self.device,
                           self.edge_index)
        elif self.config.model_name == 'PM25_GNN':
            return PM25_GNN(self.config.hist_len,
                            self.config.pred_len,
                            self.in_dim,
                            self.city_num,
                            self.config.batch_size,
                            self.device,
                            self.edge_index,
                            self.edge_attr,
                            self.wind_mean,
                            self.wind_std)
        else:
            self.logger.error('Unsupported model name')
            raise Exception('Wrong model name')

    def _train(self, train_loader):
        """
        Train model
        :return: train loss
        """
        self.model.train()
        train_loss = 0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            self.optimizer.zero_grad()
            pm25, feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            pm25_pred = self.model(pm25_hist, feature)
            loss = self.criterion(pm25_pred, pm25_label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= batch_idx + 1
        return train_loss

    def _valid(self, val_loader):
        """
        Validate model
        :return: validation loss
        """
        self.model.eval()
        val_loss = 0
        for batch_idx, data in tqdm(enumerate(val_loader)):
            pm25, feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            pm25_pred = self.model(pm25_hist, feature)
            loss = self.criterion(pm25_pred, pm25_label)
            val_loss += loss.item()

        val_loss /= batch_idx + 1
        return val_loss

    def _test(self, test_loader):
        """
        Test model
        :return: test loss
        """
        self.model.eval()
        predict_list = []
        label_list = []
        test_loss = 0
        for batch_idx, data in enumerate(test_loader):
            pm25, feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            pm25_pred = self.model(pm25_hist, feature)
            loss = self.criterion(pm25_pred, pm25_label)
            test_loss += loss.item()

            pm25_pred_val = (
                        np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1)
                        * self.pm25_std + self.pm25_mean)
            pm25_label_val = pm25.cpu().detach().numpy() * self.pm25_std + self.pm25_mean
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= batch_idx + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch

    def _create_records(self):
        exp_datetime = arrow.now().format('YYYYMMDDHHmmss')
        self.record_dir = os.path.join(self.config.records_dir, f'{self.config.model_name}_{exp_datetime}')
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def run(self):
        # save config file
        try:
            shutil.copy(self.config.config_path, os.path.join(self.record_dir, 'config.yaml'))
            self.logger.debug('config.yaml copied')
        except IOError as e:
            self.logger.error(f'Error copying config file: {e}')
        self.logger.debug('Start experiment...')
        for exp in range(self.config.exp_times):
            self.logger.info(f'Current experiment : {exp}')
            exp_dir = os.path.join(self.record_dir, f'exp_{exp}')
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            # create data loader
            train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                      drop_last=True)
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      drop_last=True)
            test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                     drop_last=True)
            # epoch variants
            val_loss_min = 1.0e5
            best_epoch = 0
            self.train_loss_list = []
            self.test_loss_list = []
            self.valid_loss_list = []
            self.rmse_list = []
            self.mae_list = []
            self.csi_list = []
            self.pod_list = []
            self.far_list = []

            for epoch in range(self.config.epochs):
                self.logger.debug(f'Experiment time :{exp}, Epoch time : {epoch}')
                train_loss = self._train(train_loader)
                val_loss = self._valid(valid_loader)
                self.logger.info('train_loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
                # End train without the best result in consecutive early_stop epochs
                if epoch - best_epoch > self.config.early_stop:
                    break
                # update val loss
                if val_loss < val_loss_min:
                    val_loss_min = val_loss
                    best_epoch = epoch
                    torch.save(self.model.state_dict(),
                               os.path.join(exp_dir, f'model_{self.config.model_name}.pth'))
                    self.logger.info(f'Save best model at epoch {epoch}, val_loss: {val_loss}')

                    # test model
                    test_loss, predict_epoch, label_epoch = self._test(test_loader)
                    rmse, mae, csi, pod, far = get_metrics(predict_epoch, label_epoch)

                    self.logger.info('Epoch time: %d, test results: \n'
                                     'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f \n'
                                     'RMSE: %0.2f, MAE: %0.2f \n'
                                     'CSI: %0.4f, POD: %0.4f, FAR: %0.4f'
                                     % (epoch, train_loss, val_loss, test_loss, rmse, mae, csi, pod, far))
                    self.train_loss_list.append(train_loss)
                    self.valid_loss_list.append(val_loss)
                    self.test_loss_list.append(test_loss)
                    self.rmse_list.append(rmse)
                    self.mae_list.append(mae)
                    self.csi_list.append(csi)
                    self.pod_list.append(pod)
                    self.far_list.append(far)

            self.logger.info('Experiment time: %d, test results: \n'
                             'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f \n'
                             'RMSE: %0.2f, MAE: %0.2f \n'
                             'CSI: %0.4f, POD: %0.4f, FAR: %0.4f'
                             % (exp, self.train_loss_list[-1], self.valid_loss_list[-1], self.test_loss_list[-1],
                                self.rmse_list[-1], self.mae_list[-1], self.csi_list[-1],
                                self.pod_list[-1], self.far_list[-1]))
            self.exp_train_loss_list.append(self.train_loss_list[-1])
            self.exp_test_loss_list.append(self.test_loss_list[-1])
            self.exp_valid_loss_list.append(self.valid_loss_list[-1])
            self.exp_rmse_list.append(self.rmse_list[-1])
            self.exp_mae_list.append(self.mae_list[-1])
            self.exp_csi_list.append(self.csi_list[-1])
            self.exp_pod_list.append(self.pod_list[-1])
            self.exp_far_list.append(self.far_list[-1])

            # save metrics
            metrics_data = np.concatenate((np.array(self.train_loss_list), np.array(self.valid_loss_list),
                                           np.array(self.test_loss_list), np.array(self.rmse_list),
                                           np.array(self.mae_list), np.array(self.csi_list),
                                           np.array(self.pod_list), np.array(self.far_list)), axis=0)
            np.save(os.path.join(exp_dir, f'exp_{exp}_res.npy'), metrics_data)

        self.logger.info("Finished all experiments: \n"
                         'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_train_loss_list)) +
                         'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_valid_loss_list)) +
                         'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_test_loss_list)) +
                         'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_rmse_list)) +
                         'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_mae_list)) +
                         'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_csi_list)) +
                         'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_pod_list)) +
                         'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_far_list)))
        metrics_data = np.concatenate((np.array(self.exp_train_loss_list), np.array(self.exp_valid_loss_list),
                                       np.array(self.exp_test_loss_list), np.array(self.exp_rmse_list),
                                       np.array(self.exp_mae_list), np.array(self.exp_csi_list),
                                       np.array(self.exp_pod_list), np.array(self.exp_far_list)), axis=0)
        np.save(os.path.join(self.record_dir, 'all_exp_res.npy'), metrics_data)
        self.logger.debug('Experiments finished.')
