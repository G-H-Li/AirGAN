import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from src.reference_model.ADAIN import ADAIN
from src.trainer.reference_base_trainer import ReferenceBaseTrainer


class AdainTrainer(ReferenceBaseTrainer):
    def __init__(self):
        super().__init__()
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()

    def _get_model(self):
        in_dim = 9
        node_in_dim = 12
        if self.config.model_name == 'ADAIN':
            return ADAIN(self.config.seq_len,
                         in_dim,
                         node_in_dim,
                         self.config.dropout)

    def _get_optimizer(self):
        return Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _get_criterion(self):
        return MSELoss()

    def _train(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            self.optimizer.zero_grad()
            label_pm25, local_nodes, local_features, station_nodes, station_features, station_dist = data
            label_pm25 = label_pm25.float().to(self.device)
            local_nodes = local_nodes.float().to(self.device)
            local_features = local_features.float().to(self.device)
            station_features = station_features.float().to(self.device)
            station_dist = station_dist.float().to(self.device)
            station_nodes = station_nodes.float().to(self.device)

            pm25_pred = self.model(local_nodes, local_features, station_nodes, station_features, station_dist)

            loss = self.criterion(pm25_pred, label_pm25)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader) + 1
        return train_loss

    def _test(self, test_loader):
        self.model.eval()
        predict_list = []
        label_list = []
        test_loss = 0
        for batch_idx, data in tqdm(enumerate(test_loader)):
            label_pm25, local_nodes, local_features, station_nodes, station_features, station_dist = data
            label_pm25 = label_pm25.float().to(self.device)
            local_nodes = local_nodes.float().to(self.device)
            local_features = local_features.float().to(self.device)
            station_features = station_features.float().to(self.device)
            station_dist = station_dist.float().to(self.device)
            station_nodes = station_nodes.float().to(self.device)

            pm25_pred = self.model(local_nodes, local_features, station_nodes, station_features, station_dist)

            loss = self.criterion(pm25_pred, label_pm25)
            test_loss += loss.item()

            pm25_pred_val = self.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
            pm25_label_val = self.pm25_scaler.denormalize(label_pm25.cpu().detach().numpy())
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= len(test_loader) + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch
