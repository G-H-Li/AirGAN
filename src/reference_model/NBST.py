import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import GRU, Sequential, Linear, Tanh, Dropout


class AttentionSampler(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionSampler, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = Linear(2 * input_size, 1)

    def forward(self, local, sites):
        attn_weights = self.get_attention_weights(local, sites)
        return attn_weights

    def get_attention_weights(self, local, sites):
        attn_energies = torch.zeros(len(sites))
        for i, site in enumerate(sites):
            attn_energies[i] = self.score(local, site)
        attn_weights = F.softmax(attn_energies, dim=0)
        return attn_weights

    def score(self, local, site):
        combined = torch.cat((local, site), -1)
        energy = self.attn(combined)
        return energy


class NBST(nn.Module):
    def __init__(self, device, batch_size, in_dim, node_in_dim, hidden_dim, dropout, gru_layer):
        super(NBST, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gru_layer = gru_layer
        self.emb_dim = 4

        self.weather_emb = nn.Embedding(12, self.emb_dim)
        self.wind_direc_emb = nn.Embedding(10, self.emb_dim)
        self.month_emb = nn.Embedding(12, self.emb_dim)
        self.day_emb = nn.Embedding(7, self.emb_dim)
        self.hour_emb = nn.Embedding(24, self.emb_dim)

        self.nodes_mlp = nn.Sequential(Linear(self.node_in_dim, self.hidden_dim),
                                       Tanh())

        self.attention_sampler = AttentionSampler(self.hidden_dim, self.hidden_dim)

        self.dynamic_mlp = Sequential(Linear(self.in_dim + 5 * self.emb_dim, self.hidden_dim // 2),
                                      Tanh(),
                                      Dropout(self.dropout),
                                      Linear(self.hidden_dim // 2, self.hidden_dim),
                                      Tanh())
        self.dynamic_encoder = GRU(self.hidden_dim, self.hidden_dim,
                                   self.gru_layer, dropout=self.dropout, batch_first=True)

        self.pred_mlp = Sequential(Linear(2 * self.hidden_dim, 1), Tanh())

    def forward(self, local_node, station_nodes, station_features, station_emb):
        batch_size = local_node.size(0)
        seq_len = station_features.size(1)

        local_node_emb = self.nodes_mlp(local_node)
        station_nodes_emb = self.nodes_mlp(station_nodes)
        attention_score = self.attention_sampler(local_node_emb, station_nodes_emb)
        weather_emb = self.weather_emb(station_emb[:, :, :, 0]).reshape(batch_size, seq_len, -1)
        wind_direc_emb = self.wind_direc_emb(station_emb[:, :, :, 1]).reshape(batch_size, seq_len, -1)
        month_emb = self.month_emb(station_emb[:, :, :, 2] - 1).reshape(batch_size, seq_len, -1)
        day_emb = self.day_emb(station_emb[:, :, :, 3] - 1).reshape(batch_size, seq_len, -1)
        hour_emb = self.hour_emb(station_emb[:, :, :, 4] - 1).reshape(batch_size, seq_len, -1)
        station_features = station_features.reshape(batch_size, seq_len, -1)
        station_features = torch.cat((station_features, weather_emb, wind_direc_emb,
                                      month_emb, day_emb, hour_emb), dim=-1)

        station_features_emb = self.dynamic_mlp(station_features)
        local_feature = attention_score * station_features_emb
        features = torch.cat((local_feature, station_features_emb), dim=-1)
        station_features_out, station_features_hidden = self.dynamic_encoder(features)
        pred = self.pred_mlp(station_features_hidden)
        return pred



