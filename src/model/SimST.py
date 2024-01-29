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
        self.date_emb = 4

        # self.city_embedding_table = torch.randn(self.city_num, self.city_emb_size).to(self.device)

        self.emb_month = nn.Embedding(12, self.date_emb)
        self.emb_weekday = nn.Embedding(7, self.date_emb)
        self.emb_hour = nn.Embedding(24, self.date_emb)

        self.graph_mlp = Sequential(Linear(self.in_dim * (k + 2), self.hidden_dim),
                                    Tanh(),
                                    Dropout(self.dropout))

        self.temp_encoder = Sequential(GRU(self.hist_len + self.hidden_dim, self.hidden_dim,
                                           self.gru_layer, dropout=self.dropout, batch_first=True))
        self.loc_mlp = Sequential(Linear(2, self.hidden_dim),
                                  Tanh(),
                                  Dropout(self.dropout))

        self.pred_mlp = Sequential(Linear((1+self.gru_layer)*self.hidden_dim+3*self.date_emb, 1),
                                   Tanh())

    def forward(self, pm25_hist, features, city_locs, date_emb):
        # PM25: Batch_size, hist_len, 1
        # features: Batch_size, k+2, hist_len, feature_nums
        # city_idx: Batch_size, 1
        # city_idx = city_idx.view(self.batch_size)
        pred_pm25 = []
        features = features.transpose(1, 2)
        batch_size = features.shape[0]
        features = features.reshape(batch_size, self.hist_len+self.pred_len, -1)
        xn = pm25_hist.view(batch_size, -1)
        city_loc_emb = self.loc_mlp(city_locs.float())
        for i in range(self.pred_len):
            month_emb = self.emb_month(date_emb[:, self.hist_len+i, 2] - 1)
            weekday_emb = self.emb_weekday(date_emb[:, self.hist_len+i, 1] - 1)
            hour_emb = self.emb_hour(date_emb[:, self.hist_len+i, 0])
            graph_emb = self.graph_mlp(features[:, self.hist_len+i].view(batch_size, -1))
            x = torch.cat((graph_emb, xn), dim=-1).unsqueeze(1)
            # graph_emb = graph_emb.view(self.batch_size, -1)
            temp_out, temp_hidden = self.temp_encoder(x)
            # city_loc_emb = self.city_embedding_table[city_idx].to(self.device)
            temp_hidden = temp_hidden.chunk(chunks=self.gru_layer, dim=0)
            temp_hidden = torch.cat(temp_hidden, dim=-1).view(batch_size, -1)
            pred = torch.cat([temp_hidden, city_loc_emb, month_emb, weekday_emb, hour_emb], dim=1)
            pred = self.pred_mlp(pred)
            pred_pm25.append(pred)
            xn = torch.cat((xn[:, 1:], pred), dim=1)
        pred_pm25 = torch.stack(pred_pm25, dim=1)
        return pred_pm25
