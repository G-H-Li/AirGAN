import os.path
import shutil

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.parser import KnowAirDataset
from src.model.AirFormer import AirFormer
from src.model.DGCRN import DGCRN
from src.model.GAGNN import GAGNN
from src.model.GC_LSTM import GC_LSTM
from src.model.GRU import GRU
from src.model.LSTM import LSTM
from src.model.MLP import MLP
from src.model.PM25_GNN import PM25_GNN
from src.trainer.trainer import Trainer
from src.utils.metrics import get_metrics
from src.utils.utils import get_mean_std


class STTrainer(Trainer):
    """
    Trainer class
    General purpose training:
    1. load setting
    2. load dataset
    3. train model
    4. test model
    """
    def __init__(self):
        super().__init__()
        # model setting
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        # train setting
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_criterion(self):
        """
        define the loss function
        :return:
        """
        if self.config.model_name == "GAGNN":
            return nn.L1Loss(reduction='sum')
        elif self.config.model_name == "AirFormer":
            return nn.L1Loss()
        else:
            return nn.MSELoss()

    def _get_optimizer(self):
        """
        define the optimizer function
        :return:
        """
        if self.config.model_name == "GAGNN":
            all_params = self.model.parameters()
            w_params = []
            for pname, p in self.model.named_parameters():
                if pname == 'w':
                    w_params += [p]
            params_id = list(map(id, w_params))
            other_params = list(filter(lambda i: id(i) not in params_id, all_params))
            return torch.optim.Adam([
                {'params': other_params},
                {'params': w_params, 'lr': self.config.lr * self.config.weight_rate}
            ], lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.model_name == "AirFormer":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        else:
            return torch.optim.RMSprop(self.model.parameters(),
                                       lr=self.config.lr,
                                       weight_decay=self.config.weight_decay)

    def _get_scheduler(self):
        """
        define the scheduler
        :return:
        """
        if self.config.model_name == "AirFormer":
            return MultiStepLR(optimizer=self.optimizer,
                               milestones=self.config.steps,
                               gamma=self.config.weight_decay)
        else:
            return None

    def _get_model(self):
        """
        construct model
        :return: model object
        """
        self.in_dim = (self.train_dataset.feature.shape[-1] +
                       self.train_dataset.pm25.shape[-1] * self.config.hist_len)
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
        elif self.config.model_name == 'GAGNN':
            self.in_dim = self.train_dataset.feature.shape[-1]
            return GAGNN(self.config.hist_len,
                         self.config.pred_len,
                         self.in_dim,
                         self.city_num,
                         self.config.batch_size,
                         self.device,
                         self.edge_index,
                         self.edge_attr,
                         self.city_loc,
                         self.config.group_num,
                         self.config.gnn_hidden,
                         self.config.gnn_layer,
                         self.config.edge_hidden,
                         self.config.head_nums)
        elif self.config.model_name == 'AirFormer':
            return AirFormer(self.config.hist_len,
                             self.config.pred_len,
                             self.in_dim,
                             self.city_num,
                             self.config.batch_size,
                             self.device,
                             self.config.dropout,
                             self.config.head_nums,
                             self.config.hidden_channels,
                             self.config.blocks)
        elif self.config.model_name == 'DGCRN':
            return DGCRN(self.config.gcn_pred_len,
                         self.city_num,
                         self.device,
                         self.static_graph,  # TODO
                         self.config.dropout,
                         self.config.node_dim,
                         2,
                         self.config.hist_len,
                         self.in_dim,
                         self.config.pred_len,
                         [0.05, 0.95, 0.95],
                         self.config.tanh_alpha,
                         4000,
                         self.config.rnn_size,
                         self.config.gnn_dim)
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
            pm25, feature, em_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            em_feature = em_feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            pm25_pred = self.model(pm25_hist, feature, em_feature)
            loss = self.criterion(pm25_pred, pm25_label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader) + 1
        return train_loss

    def _valid(self, valid_loader):
        """
        Validate model
        :return: validation loss
        """
        self.model.eval()
        val_loss = 0
        for batch_idx, data in tqdm(enumerate(valid_loader)):
            pm25, feature, time_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            time_feature = time_feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            pm25_pred = self.model(pm25_hist, feature, time_feature)
            loss = self.criterion(pm25_pred, pm25_label)
            val_loss += loss.item()

        val_loss /= len(valid_loader) + 1
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
            pm25, feature, time_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            time_feature = time_feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            pm25_pred = self.model(pm25_hist, feature, time_feature)
            loss = self.criterion(pm25_pred, pm25_label)
            test_loss += loss.item()

            pm25_pred_val = self.test_dataset.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
            pm25_label_val = self.test_dataset.pm25_scaler.denormalize(pm25_label.cpu().detach().numpy())
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= len(test_loader) + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch

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
                if epoch - best_epoch > self.config.early_stop and self.config.is_early_stop:
                    self.logger.info('Early stop at epoch {}, best loss = {:.6f}'
                                     .format(epoch, np.min(self.valid_loss_list)))
                    break
                # update val loss
                best_val_loss = np.min(self.valid_loss_list) if len(self.valid_loss_list) > 0 else np.inf
                if val_loss < best_val_loss:
                    best_epoch = epoch
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
