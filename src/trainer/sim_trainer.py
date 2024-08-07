import os
import shutil
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from src.dataset.forecast_parser import SimParser
from src.forecast_model.SimST import SimST, MultiLoss
from src.trainer.forecast_base_trainer import ForecastBaseTrainer


class SimTrainer(ForecastBaseTrainer):
    def __init__(self, mode):
        super(SimTrainer, self).__init__(mode)
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        self.predict_mode = "city"
        # train setting
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()

    def _read_data(self):
        """
        construct train, valid, and test dataset
        :return: dataset loader
        """
        if self.config.dataset_name == 'KnowAir':
            self.train_dataset = SimParser(config=self.config, mode='train')
            self.valid_dataset = SimParser(config=self.config, mode='valid')
            self.test_dataset = SimParser(config=self.config, mode='test')
            self.city_num = self.train_dataset.node_num
        elif self.config.dataset_name == 'UrbanAir':
            self.train_dataset = SimParser(config=self.config, mode='train')
            self.valid_dataset = SimParser(config=self.config, mode='valid')
            self.test_dataset = SimParser(config=self.config, mode='test')
            self.city_num = self.train_dataset.node_num
        else:
            self.logger.error("Unsupported dataset type")
            raise ValueError('Unknown dataset')

    def _get_model(self):
        self.in_dim = self.train_dataset.feature.shape[-1]
        if self.config.model_name == 'SimST':
            return SimST(self.config.hist_len,
                         self.config.pred_len,
                         self.device,
                         self.config.batch_size,
                         self.in_dim,
                         self.config.hidden_dim,
                         self.city_num,
                         self.config.dropout,
                         self.config.gru_layers,
                         self.config.use_dynamic,
                         self.config.K)
        else:
            self.logger.error('Unsupported model name')
            raise NotImplementedError

    def get_model_info(self):
        data_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                 drop_last=True, pin_memory=True, num_workers=self.config.num_workers)
        for data in data_loader:
            pm25, feature, locs, emb_feature, in_out_weight = data
            pm25 = pm25.to(self.device)
            feature = feature.to(self.device)
            emb_feature = emb_feature.int().to(self.device)
            locs = locs.to(self.device)
            in_out_weight = in_out_weight.to(self.device)
            pm25_hist = pm25[:, :self.config.hist_len]
            model_stat = summary(self.model, input_data=[pm25_hist, feature, locs, emb_feature, in_out_weight], verbose=0,
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
            pm25, feature, locs, emb_feature = data
            pm25 = pm25.to(self.device).float()
            feature = feature.to(self.device).float()
            emb_feature = emb_feature.int().to(self.device)
            locs = locs.to(self.device).float()
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]

            start_time = time()
            pm25_pred = self.model(pm25_hist, feature, locs, emb_feature)
            end_time = time()

            loss = self.criterion(pm25_pred, pm25_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()
            train_loss += loss.item()
            cost_time += ((end_time - start_time) / self.config.batch_size)

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
            pm25, feature, locs, emb_feature = data
            pm25 = pm25.to(self.device).float()
            feature = feature.to(self.device).float()
            emb_feature = emb_feature.int().to(self.device)
            locs = locs.to(self.device).float()
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]

            start_time = time()
            pm25_pred = self.model(pm25_hist, feature, locs, emb_feature)
            end_time = time()

            loss = self.criterion(pm25_pred, pm25_label)
            val_loss += loss.item()
            cost_time += ((end_time - start_time) / self.config.batch_size)

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
            pm25, feature, locs, emb_feature = data
            pm25 = pm25.to(self.device).float()
            feature = feature.to(self.device).float()
            emb_feature = emb_feature.int().to(self.device)
            locs = locs.to(self.device).float()
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]

            start_time = time()
            pm25_pred = self.model(pm25_hist, feature, locs, emb_feature)
            end_time = time()

            loss = self.criterion(pm25_pred, pm25_label)
            test_loss += loss.item()
            cost_time += ((end_time - start_time) / self.config.batch_size)

            pm25_pred_val = self.test_dataset.pm25_scaler.denormalize(pm25_pred[..., [0]].cpu().detach().numpy())
            pm25_label_val = self.test_dataset.pm25_scaler.denormalize(pm25_label[..., [0]].cpu().detach().numpy())
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= len(test_loader) + 1
        cost_time /= len(test_loader) + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch, cost_time

    def _get_criterion(self):
        """
        define the loss function
        :return:
        """
        # return nn.MSELoss()
        # return nn.L1Loss()
        return MultiLoss()

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.config.lr, weight_decay=self.config.weight_decay)

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
        dataset = SimParser(config, mode='test')
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False,
                                drop_last=True, pin_memory=True, num_workers=self.config.num_workers)

        self.model.load_state_dict(torch.load(model_path))
        if test_hist_len != self.model.hist_len:
            raise ValueError('test history_seq_len does not match the model history_seq_len')
        pred_count = test_pred_len // self.model.pred_len
        remainder = test_pred_len % self.model.pred_len
        pred_count = pred_count if remainder == 0 else pred_count + 1
        all_len = self.model.pred_len + self.model.hist_len
        self.model.eval()
        predict_list = []
        label_list = []
        self.logger.info("Start Test:")
        start_time = time()
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(dataloader)):
                pred = []
                pm25, feature, locs, emb_feature = data
                pm25 = pm25.to(self.device)
                pm25_label = pm25[:, test_hist_len:]
                pm25_hist = pm25[:, :test_hist_len]
                feature = feature.to(self.device)
                emb_feature = emb_feature.int().to(self.device)
                locs = locs.to(self.device)
                for i in range(pred_count):
                    features = feature[:, :, i*test_hist_len:i*test_hist_len+all_len]
                    emb_features = emb_feature[:, i * test_hist_len:i * test_hist_len + all_len]
                    pm25_pred = self.model(pm25_hist, features, locs, emb_features)
                    pred.append(pm25_pred)
                    pm25_hist = torch.cat([pm25_hist, pm25_pred], dim=1)[:, -test_hist_len:]

                if remainder != 0:
                    pm25_pred = torch.cat(pred, dim=1)[:, :remainder-self.model.pred_len]
                else:
                    pm25_pred = torch.cat(pred, dim=1)
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

    def run_one_test(self, model_path: str, test_hist_len: int, test_pred_len: int):
        try:
            shutil.copy(model_path, os.path.join(self.record_dir, f'model_{self.config.model_name}.pth'))
            self.logger.debug('model file copied')
        except IOError as e:
            self.logger.error(f'Error copying config file: {e}')
        # prepare dataset
        config = self.config
        config.hist_len = test_hist_len
        config.pred_len = test_pred_len
        dataset = SimParser(config, mode='test')
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.batch_size = 1
        if test_hist_len != self.model.hist_len:
            raise ValueError('test history_seq_len does not match the model history_seq_len')
        pred_count = test_pred_len // self.model.pred_len
        remainder = test_pred_len % self.model.pred_len
        pred_count = pred_count if remainder == 0 else pred_count + 1
        all_len = self.model.pred_len + self.model.hist_len
        self.model.eval()
        predict_list = []
        label_list = []
        self.logger.info("Start Test:")
        start_time = time()
        with torch.no_grad():
            pred = []
            pm25, feature, locs, emb_feature = dataset[0]
            pm25 = torch.from_numpy(pm25).to(self.device).unsqueeze(0)
            pm25_label = pm25[:, test_hist_len:]
            pm25_hist = pm25[:, :test_hist_len]
            feature = torch.from_numpy(feature).to(self.device).unsqueeze(0)
            emb_feature = torch.from_numpy(emb_feature).int().to(self.device).unsqueeze(0)
            locs = torch.from_numpy(locs).to(self.device).unsqueeze(0)
            for i in range(pred_count):
                features = feature[:, :, i*test_hist_len:i*test_hist_len+all_len]
                emb_features = emb_feature[:, i * test_hist_len:i * test_hist_len + all_len]
                pm25_pred = self.model(pm25_hist, features, locs, emb_features)
                pred.append(pm25_pred)
                pm25_hist = torch.cat([pm25_hist, pm25_pred], dim=1)[:, -test_hist_len:]

            if remainder != 0:
                pm25_pred = torch.cat(pred, dim=1)[:, :remainder-self.model.pred_len]
            else:
                pm25_pred = torch.cat(pred, dim=1)
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