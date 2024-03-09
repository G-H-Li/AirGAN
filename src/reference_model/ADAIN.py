import torch
from torch import nn


class ADAIN(nn.Module):
    def __init__(self, seq_len, in_dim, dropout):
        super(ADAIN, self).__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.lstm_hidden = 300
        self.local_linear_hidden = 200
        self.station_linear_hidden = 100
        self.dist_in_dim = 2

        self.local_lstm = nn.LSTM(self.in_dim, self.lstm_hidden, num_layers=2, batch_first=True, bidirectional=False)
        self.local_fc = nn.Sequential(nn.Linear(self.lstm_hidden, self.local_linear_hidden),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(self.local_linear_hidden, self.local_linear_hidden),
                                      nn.ReLU(),
                                      nn.Dropout(dropout))

        self.station_fc = nn.Sequential(nn.Linear(self.dist_in_dim, self.station_linear_hidden),
                                        nn.ReLU(),
                                        nn.Dropout(dropout))
        self.station_lstm = nn.LSTM(self.in_dim + 1, self.lstm_hidden, num_layers=2, batch_first=True)

        self.station_fc2 = nn.Sequential(
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
                                     nn.Linear(self.local_linear_hidden, 1))

    def forward(self, local_met, station_dist, station_features):
        station_num = station_dist.size(-2)

        _, (local_lstm_out, cn) = self.local_lstm(local_met)
        local_fc_out = self.local_fc(local_lstm_out)

        mul_list = []
        for i in range(station_num):
            station_dist = station_dist[:, i, :]
            station_feature = station_features[:, :, i]

            station_fc_out = self.station_fc(station_dist)
            _, (station_lstm_out, cn) = self.station_lstm(station_feature)

            station_out = torch.cat((station_fc_out, station_lstm_out), dim=-1)

            station_fc2_out = self.station_fc2(station_out)

            attention_in = torch.cat((local_fc_out, station_fc2_out), dim=-1)
            attention_out = self.attention(attention_in)

            mul_ele = (station_fc2_out @ attention_out).view(-1, 1, self.local_linear_hidden)
            mul_list.append(mul_ele)

        mul_out = torch.stack(mul_list, dim=1)
        mul_out = torch.sum(mul_out, dim=1)

        pred_in = torch.cat((local_fc_out, mul_out), dim=-1)
        pred_out = self.pred_fc(pred_in)
        return pred_out


