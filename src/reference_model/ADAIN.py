import torch
from torch import nn


class ADAIN(nn.Module):
    def __init__(self, seq_len, in_dim, node_in_dim, dropout):
        super(ADAIN, self).__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        self.lstm_hidden = 300
        self.local_linear_hidden = 200
        self.station_linear_hidden = 100
        self.dist_in_dim = 2

        self.local_static_fc = nn.Sequential(nn.Linear(self.node_in_dim, self.station_linear_hidden),
                                             nn.ReLU(),
                                             nn.Dropout(dropout))
        self.local_lstm = nn.LSTM(self.in_dim - 1, self.lstm_hidden, num_layers=2, batch_first=True, bidirectional=False)
        self.local_fc = nn.Sequential(nn.Linear(self.lstm_hidden + self.station_linear_hidden, self.local_linear_hidden),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(self.local_linear_hidden, self.local_linear_hidden),
                                      nn.ReLU(),
                                      nn.Dropout(dropout))

        self.station_static_fc = nn.Sequential(nn.Linear(self.dist_in_dim + self.node_in_dim, self.station_linear_hidden),
                                               nn.ReLU(),
                                               nn.Dropout(dropout))
        self.station_lstm = nn.LSTM(self.in_dim, self.lstm_hidden, num_layers=2, batch_first=True)

        self.station_fc = nn.Sequential(
            nn.Linear(self.station_linear_hidden + self.lstm_hidden, self.local_linear_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.local_linear_hidden, self.local_linear_hidden),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.attention = nn.Sequential(nn.Linear(self.local_linear_hidden * 2, self.local_linear_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.local_linear_hidden, self.local_linear_hidden),
                                       nn.Linear(self.local_linear_hidden, 1),
                                       nn.Softmax())

        self.pred_fc = nn.Sequential(nn.Linear(2 * self.local_linear_hidden, self.local_linear_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(self.local_linear_hidden, self.seq_len))

    def forward(self, local_node, local_features, station_nodes, station_features, station_dist):
        station_num = station_dist.size(-2)

        local_static_out = self.local_static_fc(local_node.squeeze(1))
        _, (local_lstm_out, cn) = self.local_lstm(local_features.squeeze(2))
        local_fc_out = self.local_fc(torch.cat((local_lstm_out[-1], local_static_out), dim=-1))

        mul_list = []
        for i in range(station_num):
            station_dist_i = station_dist[:, i, :]
            station_nodes_i = station_nodes[:, i, :]
            station_feature_i = station_features[:, :, i]

            station_static_out = self.station_static_fc(torch.cat((station_dist_i, station_nodes_i), dim=-1))
            _, (station_lstm_out, cn) = self.station_lstm(station_feature_i)

            station_out = torch.cat((station_static_out, station_lstm_out[-1]), dim=-1)

            station_fc_out = self.station_fc(station_out)

            attention_in = torch.cat((local_fc_out, station_fc_out), dim=-1)
            attention_out = self.attention(attention_in)

            mul_ele = (station_fc_out * attention_out).view(-1, 1, self.local_linear_hidden)
            mul_list.append(mul_ele)

        mul_out = torch.stack(mul_list, dim=1)
        mul_out = torch.sum(mul_out, dim=1)

        pred_in = torch.cat((local_fc_out, mul_out.squeeze(1)), dim=-1)
        pred_out = self.pred_fc(pred_in)
        pred = pred_out.view(-1, self.seq_len, 1)
        return pred
