import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, GRU, Tanh, Dropout


class SimST(nn.Module):
    def __init__(self, hist_len, pred_len, device, batch_size, in_dim, hidden_dim, city_num,
                 city_emb_size, dropout, gru_layer, k):
        super(SimST, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.city_num = city_num
        self.city_emb_size = city_emb_size
        self.dropout = dropout
        self.gru_layer = gru_layer
        self.k = k

        self.city_embedding_table = torch.randn(self.city_num, self.city_emb_size).to(self.device)

        self.graph_mlp = Sequential(Linear(self.in_dim * (k + 2), self.hidden_dim),
                                    Tanh(),
                                    Dropout(self.dropout))

        self.temp_encoder = Sequential(GRU(self.hist_len*self.hidden_dim, self.hidden_dim,
                                           self.gru_layer, dropout=self.dropout))
        self.loc_mlp = Sequential(Linear(self.city_emb_size, self.hidden_dim),
                                  Sigmoid(),
                                  Dropout(self.dropout))

        self.pred_mlp = Sequential(Linear(2*self.hidden_dim, self.pred_len),
                                   Sigmoid())

    def forward(self, pm25, features, city_idx):
        # PM25: Batch_size, hist_len, 1
        # features: Batch_size, k+2, hist_len, feature_nums
        # city_idx: Batch_size, 1
        city_idx = city_idx.view(self.batch_size)
        features = features[:, :, :self.hist_len]
        features = features.transpose(1, 2)
        features = features.reshape(self.batch_size, self.hist_len, -1)
        graph_emb = self.graph_mlp(features)
        graph_emb = graph_emb.view(self.batch_size, -1)
        temp_out, temp_hidden = self.temp_encoder(graph_emb)
        city_loc_emb = self.city_embedding_table[city_idx].to(self.device)
        city_loc_emb = self.loc_mlp(city_loc_emb)
        pred = torch.cat([temp_out, city_loc_emb], dim=1)
        pred = self.pred_mlp(pred).view(self.batch_size, self.pred_len, 1)
        return pred
