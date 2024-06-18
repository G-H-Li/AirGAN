import torch
from pytorch_tcn import TCN
from torch import nn
from torch.nn import Sequential, Linear, GRU, Dropout, Tanh


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()

    def forward(self, pred, target):
        target_class = target[..., 1]
        pred_class = pred[..., 1]

        pred_val = pred[..., 0]
        target_val = target[..., 0]

        # abnormal_loss_fn = nn.MSELoss()
        normal_loss_fn = nn.L1Loss()
        class_loss_fn = nn.L1Loss()

        alpha = 1
        w1 = 1
        loss_val = (w1 * normal_loss_fn(pred_val, target_val) + alpha * class_loss_fn(pred_class, target_class))
        return loss_val


class SimST(nn.Module):
    def __init__(self, hist_len, pred_len, device, batch_size, in_dim, hidden_dim, city_num,
                 dropout, gru_layer, use_dynamic, pm25_mean, pm25_std):
        super(SimST, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.city_num = city_num
        self.dropout = dropout
        self.gru_layer = gru_layer
        self.date_emb = 4
        self.loc_emb = 16
        self.use_dynamic = use_dynamic
        self.pm25_mean = pm25_mean
        self.pm25_std = pm25_std

        self.emb_month = nn.Embedding(12, self.date_emb)
        self.emb_weekday = nn.Embedding(7, self.date_emb)
        self.emb_hour = nn.Embedding(24, self.date_emb)

        self.loc_mlp = Sequential(Linear(2, self.loc_emb),
                                  Tanh())

        self.static_mlp = Sequential(Linear(self.in_dim * 2, self.hidden_dim),
                                     Tanh(),
                                     Dropout(self.dropout),
                                     Linear(self.hidden_dim, self.hidden_dim),
                                     Tanh())

        self.dynamic_mlp = Sequential(Linear(2 + 3 * self.date_emb + self.loc_emb, self.hidden_dim),
                                      Tanh(),
                                      Dropout(self.dropout),
                                      Linear(self.hidden_dim, self.hidden_dim),
                                      Tanh())
        self.dynamic_encoder = GRU(1 + 2 * self.hidden_dim, self.hidden_dim,
                                   self.gru_layer, dropout=self.dropout, batch_first=True, bidirectional=True)

        self.dynamic_tcn_encoder = TCN(2 + 2 * self.hidden_dim, [self.hidden_dim], dropout=self.dropout,
                                       input_shape='NLC', use_skip_connections=True, output_projection=self.hidden_dim)

        self.pred_mlp = Sequential(Linear(1 * self.hidden_dim, 1),
                                   Tanh())
        self.class_mlp = Sequential(Linear(1 * self.hidden_dim, 1),
                                    Tanh())

    def forward(self, pm25_hist, features, city_locs, date_emb, in_out_weight):
        # PM25: Batch_size, hist_len, 1
        # features: Batch_size, 2, hist_len+pred_len, feature_nums
        # city_locs: Batch_size, 2
        # date_emb: Batch_size, hist_len+pred_len, 3
        # in_out_weight: Batch_size, hist_len+pred_len, 2
        pred_pm25 = []
        # 消融实验1, 去除额外静态邻近特征
        # features = features[:, [0], :, :]
        features = features.transpose(1, 2)
        batch_size = features.shape[0]
        hist_len = pm25_hist.shape[1]
        pred_len = features.shape[1] - hist_len
        features = features.reshape(batch_size, features.shape[1], -1)

        xn = pm25_hist
        all_month_emb = self.emb_month(date_emb[:, :, 2] - 1)
        all_weekday_emb = self.emb_weekday(date_emb[:, :, 1] - 1)
        all_hour_emb = self.emb_hour(date_emb[:, :, 0])
        city_loc_emb = self.loc_mlp(city_locs)
        for i in range(pred_len):
            static_graph_emb = self.static_mlp(features[:, :hist_len + i])  # 消融实验3，去除时序特征
            dynamic_graph_emb = torch.cat([city_loc_emb.reshape(batch_size, 1, -1).repeat(1, i + hist_len, 1),
                                           in_out_weight[:, :hist_len + i],  # 消融实验1，去除动态传播特征
                                           all_month_emb[:, :hist_len + i],
                                           all_weekday_emb[:, :hist_len + i],
                                           all_hour_emb[:, :hist_len + i]
                                           ], dim=-1)
            dynamic_graph_emb = self.dynamic_mlp(dynamic_graph_emb)
            dynamic_graph_emb = torch.cat((dynamic_graph_emb, static_graph_emb, xn[:, :hist_len + i]), dim=-1)
            # dynamic_out, dynamic_hidden = self.dynamic_encoder(dynamic_graph_emb)
            # pred = torch.cat([dynamic_hidden[-1], dynamic_hidden[-2]], dim=1)
            # tcn
            dynamic_graph_emb = self.dynamic_tcn_encoder(dynamic_graph_emb)
            pred_emb = dynamic_graph_emb[:, -1]
            # pred = dynamic_hidden[-1]  # 消融实验3 去除双向GRU
            cl = self.class_mlp(pred_emb)
            pred = self.pred_mlp(pred_emb)
            x = torch.cat([pred, cl], dim=1)
            # x = pred
            pred_pm25.append(x)
            xn = torch.cat((xn, x.unsqueeze(1)), dim=1)
        pred_pm25 = torch.stack(pred_pm25, dim=1)
        return pred_pm25
