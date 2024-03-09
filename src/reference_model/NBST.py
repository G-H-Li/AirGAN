import numpy as np
import torch
from torch import nn
from torch.nn import GRU, Sequential, Linear, Tanh, Dropout, LayerNorm, Softmax, Sigmoid, ReLU


class StaticHalfSelfAttention(nn.Module):
    def __init__(self, input_size, device):
        super(StaticHalfSelfAttention, self).__init__()
        self.input_size = input_size
        self.device = device
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, local, sites):
        u = local @ sites.transpose(-1, -2)
        u = u / np.power(sites.size(-2), 0.5)
        attn = self.softmax(u)
        output = attn @ sites
        return attn, output


class NBST(nn.Module):
    def __init__(self, seq_len, device, batch_size, in_dim, node_in_dim, hidden_dim, dropout, gru_layer):
        super(NBST, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gru_layer = gru_layer
        self.emb_dim = 8

        self.pm25_emb = nn.Embedding(6, self.emb_dim)
        self.weather_emb = nn.Embedding(17, self.emb_dim)
        self.wind_direc_emb = nn.Embedding(25, self.emb_dim)
        self.month_emb = nn.Embedding(12, self.emb_dim)
        self.day_emb = nn.Embedding(7, self.emb_dim)
        self.hour_emb = nn.Embedding(24, self.emb_dim)

        self.static_mlp = nn.Sequential(Linear(self.node_in_dim, self.hidden_dim * 2),
                                        Sigmoid(),
                                        Dropout(self.dropout),
                                        Linear(self.hidden_dim * 2, self.hidden_dim),
                                        Sigmoid(),
                                        nn.LayerNorm(self.hidden_dim))
        self.static_attention = StaticHalfSelfAttention(self.hidden_dim, self.device)
        self.s_attention_batch_norm = nn.BatchNorm1d(self.hidden_dim)

        self.dynamic_mlp = Sequential(Linear(self.in_dim + 2 * self.emb_dim, self.hidden_dim * 2),
                                      Tanh(),
                                      Dropout(self.dropout),
                                      Linear(self.hidden_dim * 2, self.hidden_dim),
                                      Tanh(),
                                      nn.LayerNorm(self.hidden_dim))
        self.dynamic_attention = StaticHalfSelfAttention(self.hidden_dim, self.device)
        self.d_attention_batch_norm = nn.BatchNorm2d(self.hidden_dim)

        self.stations_mlp = nn.Sequential(Linear(self.hidden_dim * 2 + self.emb_dim, self.hidden_dim * 2),
                                          ReLU(),
                                          Dropout(self.dropout),
                                          Linear(self.hidden_dim * 2, self.hidden_dim),
                                          ReLU(),
                                          nn.LayerNorm(self.hidden_dim))

        self.time_encoder = GRU(4 * self.hidden_dim + 3 * self.emb_dim, self.hidden_dim,
                                self.gru_layer, dropout=self.dropout, batch_first=True)

        self.pred_mlp = Sequential(Linear(self.hidden_dim, 1), Tanh())
        # self.pred_mlp = Linear(2 * self.hidden_dim, 6)

    def forward(self, local_node, local_features, local_emb, station_nodes, station_features, station_emb):
        # local_node: batch_size, 1(node_num), 16(node_features_num)
        # local_features: batch_size, seq_len, 1(node_num), 4(features_num)
        # local_emb: batch_size, seq_len,1(node_num), 5(features_num)  station_emb: 6
        batch_size = local_node.size(0)
        seq_len = station_features.size(1)

        month_emb = self.month_emb(station_emb[:, :, :, 3] - 1)
        day_emb = self.day_emb(station_emb[:, :, :, 4])
        hour_emb = self.hour_emb(station_emb[:, :, :, 5])
        station_pm25_emb = self.pm25_emb(station_emb[:, :, :, 0])
        station_weather_emb = self.weather_emb(station_emb[:, :, :, 1])
        station_wind_direc_emb = self.wind_direc_emb(station_emb[:, :, :, 2])
        local_weather_emb = self.weather_emb(local_emb[:, :, :, 0])
        local_wind_direc_emb = self.wind_direc_emb(local_emb[:, :, :, 1])

        local_node_emb = self.static_mlp(local_node.view(-1, local_node.size(-1)))
        station_nodes_emb = self.static_mlp(station_nodes.view(-1, local_node.size(-1)))
        local_node_emb = local_node_emb.view(batch_size, -1, self.hidden_dim)
        station_nodes_emb = station_nodes_emb.view(batch_size, -1, self.hidden_dim)
        static_attention, static_output = self.static_attention(local_node_emb, station_nodes_emb)
        static_output = torch.transpose(static_output, 2, 1)
        static_output_norm = self.s_attention_batch_norm(static_output)
        static_output_norm = torch.transpose(static_output_norm, 2, 1)

        station_features = torch.cat((station_features, station_weather_emb, station_wind_direc_emb), dim=-1)
        station_features_emb = self.dynamic_mlp(station_features.view(-1, station_features.size(-1)))
        local_features = torch.cat((local_features, local_weather_emb, local_wind_direc_emb), dim=-1)
        local_features_emb = self.dynamic_mlp(local_features.view(-1, local_features.size(-1)))
        station_features_emb = station_features_emb.view(batch_size, seq_len, -1, self.hidden_dim)
        local_features_emb = local_features_emb.view(batch_size, seq_len, -1, self.hidden_dim)
        dynamic_attention, dynamic_output = self.dynamic_attention(local_features_emb, station_features_emb)
        dynamic_output = torch.transpose(dynamic_output, -1, 1)
        dynamic_output_norm = self.d_attention_batch_norm(dynamic_output)
        dynamic_output_norm = torch.transpose(dynamic_output_norm, -1, 1)

        station_nodes_emb = station_nodes_emb.view(batch_size, 1, -1, self.hidden_dim).repeat(1, seq_len, 1, 1)
        station_features = torch.cat((station_nodes_emb, station_features_emb, station_pm25_emb), dim=-1)
        stations_features = self.stations_mlp(station_features.view(-1, 2*self.hidden_dim+self.emb_dim))
        stations_features = stations_features.view(batch_size, seq_len, -1, self.hidden_dim)
        static_attention = static_attention.unsqueeze(dim=1).repeat(1, seq_len, 1, 1)
        static_att_stations_features = (static_attention @ stations_features).squeeze(dim=2)
        dynamic_att_stations_features = (dynamic_attention @ stations_features).squeeze(dim=2)

        preds = []
        static_output_norm = static_output_norm.repeat(1, seq_len, 1)
        for i in range(seq_len):
            time_features = torch.cat((static_output_norm[:, :i+1], dynamic_output_norm[:, :i+1, 0],
                                       static_att_stations_features[:, :i+1], dynamic_att_stations_features[:, :i+1],
                                       month_emb[:, :i+1, 0], day_emb[:, :i+1, 0], hour_emb[:, :i+1, 0]), dim=-1)
            station_features_out, station_features_hidden = self.time_encoder(time_features)
            pred = self.pred_mlp(station_features_hidden[-1])
            preds.append(pred.view(batch_size, 1, -1))
        preds = torch.stack(preds, dim=1).view(batch_size, seq_len, -1)
        return preds



