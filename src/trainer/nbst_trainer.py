import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.dataset.reference_parser import NBSTParser
from src.reference_model.NBST import NBST
from src.trainer.reference_base_trainer import ReferenceBaseTrainer


class NBSTTrainer(ReferenceBaseTrainer):
    def __init__(self, mode='train'):
        super().__init__(mode=mode)
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        self.optimizer = self._get_optimizer()

    def _get_model(self):
        self.in_dim = 4
        self.node_in_dim = 12
        if self.config.model_name == 'NBST':
            return NBST(self.config.seq_len,
                        self.device,
                        self.config.batch_size,
                        self.in_dim,
                        self.node_in_dim,
                        self.config.hidden_dim,
                        self.config.dropout,
                        self.config.head_num,
                        self.config.attn_layer)
        else:
            self.logger.error('Unsupported model name')
            raise NotImplementedError

    def _get_criterion(self):
        # return nn.L1Loss()
        return nn.MSELoss()
        # return NBSTLoss(self.pm25_std, self.pm25_mean, self.config.alpha, self.device)

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # return torch.optim.NAdam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # return torch.optim.SGD(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _train(self, train_loader):
        """
        Train model
        :return: train loss
        """
        self.model.train()
        train_loss = 0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            self.optimizer.zero_grad()
            local_pm25, station_dist, local_node, local_features, local_emb, station_nodes, station_features, station_emb = data
            local_node = local_node.float().to(self.device)
            local_features = local_features.float().to(self.device)
            local_emb = local_emb.to(self.device)
            station_nodes = station_nodes.float().to(self.device)
            station_emb = station_emb.to(self.device)
            station_features = station_features.float().to(self.device)
            pm25_label = local_pm25.float().to(self.device)
            station_dist = station_dist.float().to(self.device)

            pm25_pred = self.model(station_dist, local_node, local_features,
                                   local_emb, station_nodes, station_features, station_emb)

            loss = self.criterion(pm25_pred, pm25_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader) + 1
        return train_loss

    def _test(self, test_loader):
        """
        Test model
        :return: test loss
        """
        self.model.eval()
        predict_list = []
        label_list = []
        test_loss = 0
        for batch_idx, data in tqdm(enumerate(test_loader)):
            local_pm25, station_dist, local_node, local_features, local_emb, station_nodes, station_features, station_emb = data
            local_node = local_node.float().to(self.device)
            local_features = local_features.float().to(self.device)
            local_emb = local_emb.to(self.device)
            station_nodes = station_nodes.float().to(self.device)
            station_emb = station_emb.to(self.device)
            station_features = station_features.float().to(self.device)
            pm25_label = local_pm25.float().to(self.device)
            station_dist = station_dist.float().to(self.device)

            pm25_pred = self.model(station_dist, local_node, local_features,
                                   local_emb, station_nodes, station_features, station_emb)

            loss = self.criterion(pm25_pred, pm25_label)
            test_loss += loss.item()

            pm25_pred_val = self.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
            pm25_label_val = self.pm25_scaler.denormalize(pm25_label.cpu().detach().numpy())
            # pm25_pred_val = pm25_pred.cpu().detach().numpy()
            # pm25_label_val = pm25_label.cpu().detach().numpy()
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= len(test_loader) + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch

    def run_test(self, model_path: str, idx: int):
        x = list(range(30))
        x.remove(3)
        dataset = NBSTParser(self.config, x, [3], mode='valid')
        self.pm25_scaler = dataset.pm25_scaler

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.batch_size = 1

        local_pm25, station_dist, local_node, local_features, local_emb, station_nodes, station_features, station_emb = dataset[idx]

        self.model.eval()
        predict_list = []
        label_list = []

        with torch.no_grad():
            local_node = torch.from_numpy(local_node).float().to(self.device).unsqueeze(0)
            local_features = torch.from_numpy(local_features).float().to(self.device).unsqueeze(0)
            local_emb = torch.from_numpy(local_emb).to(self.device).unsqueeze(0)
            station_nodes = torch.from_numpy(station_nodes).float().to(self.device).unsqueeze(0)
            station_emb = torch.from_numpy(station_emb).to(self.device).unsqueeze(0)
            station_features = torch.from_numpy(station_features).float().to(self.device).unsqueeze(0)
            station_dist = torch.from_numpy(station_dist).float().to(self.device).unsqueeze(0)

            pm25_pred = self.model(station_dist, local_node, local_features,
                                   local_emb, station_nodes, station_features, station_emb)

            pm25_pred_val = self.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
            pm25_label_val = self.pm25_scaler.denormalize(local_pm25)
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        np.save(os.path.join(self.record_dir,
                             f'{self.config.model_name}_predict_{idx}.npy'), predict_epoch)
        np.save(os.path.join(self.record_dir,
                             f'{self.config.model_name}_label_{idx}.npy'), label_epoch)