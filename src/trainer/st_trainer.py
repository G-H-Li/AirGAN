import os
import shutil
from time import time

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from src.dataset.forecast_parser import KnowAirDataset
from src.forecast_model.AirFormer import AirFormer
from src.forecast_model.DGCRN import DGCRN
from src.forecast_model.GAGNN import GAGNN
from src.forecast_model.GC_LSTM import GC_LSTM
from src.forecast_model.GRU import GRU
from src.forecast_model.LSTM import LSTM
from src.forecast_model.MLP import MLP
from src.forecast_model.PM25_GNN import PM25_GNN
from src.trainer.forecast_base_trainer import ForecastBaseTrainer


class STTrainer(ForecastBaseTrainer):
    """
    Trainer class
    General purpose training:
    1. load setting
    2. load dataset
    3. train model
    4. test model
    """
    def __init__(self, mode):
        super().__init__(mode)
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

    def get_model_info(self):
        data_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                 drop_last=True, pin_memory=True, num_workers=self.config.num_workers)
        for data in data_loader:
            pm25, feature, em_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            em_feature = em_feature.to(self.device)
            pm25_hist = pm25[:, :self.config.hist_len]
            model_stat = summary(self.model, input_data=[pm25_hist, feature, em_feature], verbose=0,
                                 batch_dim=self.config.batch_size,
                                 col_names=["input_size", "output_size", "num_params", "params_percent",
                                            "kernel_size", "mult_adds", "trainable"])

            self.logger.info(model_stat)
            break

    def _train(self, train_loader):
        """
        Train model
        :return: train loss
        """
        self.model.train()
        train_loss = 0
        cost_time = 0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            self.optimizer.zero_grad()
            pm25, feature, em_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            em_feature = em_feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]

            start_time = time()
            pm25_pred = self.model(pm25_hist, feature, em_feature)
            end_time = time()

            loss = self.criterion(pm25_pred, pm25_label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            cost_time += ((end_time - start_time) / self.config.batch_size / self.city_num)

        train_loss /= len(train_loader) + 1
        cost_time /= len(train_loader) + 1
        return train_loss, cost_time

    def _valid(self, valid_loader):
        """
        Validate model
        :return: validation loss
        """
        self.model.eval()
        val_loss = 0
        cost_time = 0
        for batch_idx, data in tqdm(enumerate(valid_loader)):
            pm25, feature, time_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            time_feature = time_feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]

            start_time = time()
            pm25_pred = self.model(pm25_hist, feature, time_feature)
            end_time = time()

            loss = self.criterion(pm25_pred, pm25_label)
            val_loss += loss.item()
            cost_time += ((end_time - start_time) / self.config.batch_size / self.city_num)

        val_loss /= len(valid_loader) + 1
        cost_time /= len(valid_loader) + 1
        return val_loss, cost_time

    def _test(self, test_loader):
        """
        Test model
        :return: test loss
        """
        self.model.eval()
        predict_list = []
        label_list = []
        test_loss = 0
        cost_time = 0
        for batch_idx, data in tqdm(enumerate(test_loader)):
            pm25, feature, time_feature = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            time_feature = time_feature.to(self.device)
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]

            start_time = time()
            pm25_pred = self.model(pm25_hist, feature, time_feature)
            end_time = time()

            loss = self.criterion(pm25_pred, pm25_label)
            test_loss += loss.item()
            cost_time += ((end_time - start_time) / self.config.batch_size / self.city_num)

            pm25_pred_val = self.test_dataset.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
            pm25_label_val = self.test_dataset.pm25_scaler.denormalize(pm25_label.cpu().detach().numpy())
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= len(test_loader) + 1
        cost_time /= len(test_loader) + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch, cost_time

    def run_test(self, model_path: str, test_hist_len: int, test_pred_len: int):
        try:
            shutil.copy(model_path, os.path.join(self.record_dir, f'model_{self.config.model_name}.pth'))
            self.logger.debug('model file copied')
        except IOError as e:
            self.logger.error(f'Error copying config file: {e}')
        # prepare dataset
        config = self.config
        config.hist_len = test_hist_len
        config.pred_len = test_pred_len
        dataset = KnowAirDataset(config, mode='test')
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False,
                                drop_last=True, pin_memory=True, num_workers=self.config.num_workers)

        self.model.load_state_dict(torch.load(model_path))
        self.model.hist_len = test_hist_len
        self.model.pred_len = test_pred_len
        self.model.eval()
        predict_list = []
        label_list = []
        self.logger.info("Start Test:")
        start_time = time()
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(dataloader)):
                pm25, feature, time_feature = data
                pm25 = pm25.to(self.device)
                feature = feature.to(self.device)
                time_feature = time_feature.to(self.device)
                pm25_label = pm25[:, test_hist_len:]
                pm25_hist = pm25[:, :test_hist_len]
                pm25_pred = self.model(pm25_hist, feature, time_feature)

                pm25_pred_val = self.test_dataset.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
                pm25_label_val = self.test_dataset.pm25_scaler.denormalize(pm25_label.cpu().detach().numpy())
                predict_list.append(pm25_pred_val)
                label_list.append(pm25_label_val)

        end_time = time()

        self.logger.info(f'Test end. Time taken: {end_time - start_time} s')
        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        np.save(os.path.join(self.record_dir,
                             f'{self.config.model_name}_predict_{test_hist_len}_{test_pred_len}.npy'), predict_epoch)
        np.save(os.path.join(self.record_dir,
                             f'{self.config.model_name}_label_{test_hist_len}_{test_pred_len}.npy'), label_epoch)