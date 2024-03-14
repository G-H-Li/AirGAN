import numpy as np
import torch
from torch import nn
from torch.nn import GRU, Sequential, Linear, Dropout, ReLU, Tanh
from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits


class NBSTLoss(nn.Module):
    def __init__(self, pm25_std, pm25_mean, alpha):
        super(NBSTLoss, self).__init__()
        self.pm25_std = pm25_std
        self.pm25_mean = pm25_mean
        self.alpha = alpha

    def forward(self, output, target):
        loss1 = l1_loss(output, target)
        output_val = output * self.pm25_std + self.pm25_mean
        target_val = target * self.pm25_std + self.pm25_mean
        output_val = (output_val <= 75).float()
        target_val = (target_val <= 75).float()
        loss2 = binary_cross_entropy_with_logits(output_val, target_val)
        return loss1 + self.alpha * loss2


class StaticAttention(nn.Module):
    def __init__(self, head_num, input_size):
        super(StaticAttention, self).__init__()
        self.input_size = input_size
        self.head_num = head_num

        self.multi_head_q = nn.Sequential(Linear(input_size, head_num * input_size),
                                          nn.LayerNorm(head_num * input_size))  #
        self.multi_head_k = nn.Sequential(Linear(input_size, head_num * input_size),
                                          nn.LayerNorm(head_num * input_size))  #
        self.multi_head_v = nn.Sequential(Linear(input_size, head_num * input_size),
                                          nn.LayerNorm(head_num * input_size))  #

        self.softmax = nn.Softmax(dim=-1)
        self.multi_head_out = nn.Sequential(nn.LayerNorm(head_num * input_size),
                                            Linear(head_num * input_size, input_size),
                                            nn.LayerNorm(input_size))  #
        self.multi_head_att_out = nn.Sequential(Linear(head_num, 1),
                                                nn.LayerNorm(1))  #

    def forward(self, local, dist, sites, mask=None):
        site_num = sites.size(-2)
        local = self.multi_head_q(local)
        sites = self.multi_head_v(sites)
        dist = self.multi_head_k(dist)

        local = (local.view(-1, 1, self.head_num, self.input_size).permute(2, 0, 1, 3)
                 .contiguous().view(-1, 1, self.input_size))
        sites = (sites.view(-1, site_num, self.head_num, self.input_size).permute(2, 0, 1, 3)
                 .contiguous().view(-1, site_num, self.input_size))
        dist = (dist.view(-1, site_num, self.head_num, self.input_size).permute(2, 0, 1, 3)
                .contiguous().view(-1, site_num, self.input_size))

        u = local @ dist.transpose(-1, -2)
        u = u / np.power(site_num, 0.5)

        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = attn @ sites

        output = output + local

        output = (output.view(self.head_num, -1, 1, self.input_size).permute(1, 2, 0, 3)
                  .contiguous().view(-1, 1, self.head_num * self.input_size))
        attn = (attn.view(self.head_num, -1, 1, site_num).permute(1, 2, 0, 3)
                .contiguous().view(-1, site_num, self.head_num))

        output = self.multi_head_out(output)
        attn = self.multi_head_att_out(attn).transpose(-1, -2)
        return attn, output


class DynamicSelfAttention(nn.Module):
    def __init__(self, seq_len, emb_dim, input_size):
        super(DynamicSelfAttention, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.input_size = input_size
        self.attn_layers = 4

        self.q_layer_norm = nn.Sequential(Linear(self.emb_dim, self.input_size), nn.LayerNorm(self.input_size))  #
        self.k_layer_norm = nn.Sequential(Linear(self.input_size, self.input_size), nn.LayerNorm(self.input_size))  #
        self.v_layer_norm = nn.Sequential(Linear(self.input_size, self.input_size), nn.LayerNorm(self.input_size))  #
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(self.input_size)

    def forward(self, q, k, v, mask=None):
        site_num = k.size(2)
        q = self.q_layer_norm(q.reshape(-1, site_num, self.emb_dim))
        k = self.k_layer_norm(k.reshape(-1, site_num, self.input_size))
        v = self.v_layer_norm(v.reshape(-1, site_num, self.input_size))

        u = q @ k.transpose(-2, -1)
        u = u / np.power(site_num, 0.5)

        if mask is not None:
            mask = mask.repeat(self.seq_len, 1, 1)
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        attn_output = attn @ v

        attn_output = attn_output + v

        # station_in = (attn_output.view(-1, self.seq_len, site_num, self.input_size).permute(0, 2, 1, 3)
        #               .contiguous().view(-1, site_num, self.input_size * self.seq_len))
        # station_output = self.station_out(station_in)
        attn_output = self.layer_norm(attn_output)
        station_output = attn_output.view(-1, self.seq_len, site_num, self.input_size)
        return station_output


class NBST(nn.Module):
    def __init__(self, seq_len, device, batch_size, in_dim, node_in_dim, hidden_dim, dropout, head_num, attn_layer):
        super(NBST, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.emb_dim = 10
        self.head_num = head_num
        self.attn_layer = attn_layer

        self.pm25_emb = nn.Embedding(6, self.emb_dim)
        self.weather_emb = nn.Embedding(17, self.emb_dim)
        self.wind_direc_emb = nn.Embedding(25, self.emb_dim)
        self.month_emb = nn.Embedding(12, self.emb_dim)
        self.day_emb = nn.Embedding(7, self.emb_dim)
        self.hour_emb = nn.Embedding(24, self.emb_dim)

        self.station_dist_mlp = nn.Sequential(Linear(self.node_in_dim + 2, self.hidden_dim * 2),
                                              Tanh(),
                                              Linear(self.hidden_dim * 2, self.hidden_dim),
                                              Tanh(),
                                              nn.LayerNorm(self.hidden_dim))  #

        self.static_mlp = nn.Sequential(Linear(self.node_in_dim, self.hidden_dim * 2),
                                        Tanh(),
                                        Linear(self.hidden_dim * 2, self.hidden_dim),
                                        Tanh(),
                                        nn.LayerNorm(self.hidden_dim))  #

        self.static_attn_layers = nn.ModuleList([StaticAttention(self.head_num, self.hidden_dim)])
        for _ in range(self.attn_layer - 1):
            self.static_attn_layers.append(StaticAttention(self.head_num, self.hidden_dim))
        self.static_attn_out_fc = nn.Sequential(Linear(self.attn_layer * self.hidden_dim, 2 * self.hidden_dim),
                                                Tanh(),
                                                Dropout(self.dropout),
                                                Linear(2 * self.hidden_dim, self.hidden_dim),
                                                Tanh(),
                                                Dropout(self.dropout),
                                                nn.LayerNorm(self.hidden_dim))
        # self.static_attention = StaticAttention(self.head_num, self.hidden_dim)

        self.station_dynamic_mlp = Sequential(Linear(self.in_dim + self.emb_dim * 6, self.hidden_dim * 2),
                                              Tanh(),
                                              Linear(self.hidden_dim * 2, self.hidden_dim),
                                              Tanh(),
                                              nn.LayerNorm(self.hidden_dim))  #

        self.local_dynamic_mlp = Sequential(Linear(self.in_dim + self.emb_dim * 5, self.hidden_dim * 2),
                                            Tanh(),
                                            Linear(self.hidden_dim * 2, self.hidden_dim),
                                            Tanh(),
                                            nn.LayerNorm(self.hidden_dim))  #

        self.station_dynamic_attn_layers = nn.ModuleList(
            [DynamicSelfAttention(self.seq_len, self.emb_dim, self.hidden_dim)])
        self.local_dynamic_attn_layers = nn.ModuleList(
            [DynamicSelfAttention(self.seq_len, self.hidden_dim, self.hidden_dim)])
        for _ in range(self.attn_layer - 1):
            self.station_dynamic_attn_layers.append(
                DynamicSelfAttention(self.seq_len, self.emb_dim, self.hidden_dim))
            self.local_dynamic_attn_layers.append(
                DynamicSelfAttention(self.seq_len, self.hidden_dim, self.hidden_dim))
        # self.station_dynamic_attn = DynamicSelfAttention(self.seq_len, self.emb_dim, self.hidden_dim, self.gru_layer,
        #                                                  self.dropout)
        # self.local_dynamic_attn = DynamicSelfAttention(self.seq_len, self.hidden_dim, self.hidden_dim, self.gru_layer,
        #                                                self.dropout)
        self.station_attn_out = nn.Sequential(nn.LayerNorm(self.seq_len * self.hidden_dim),
                                              Linear(self.hidden_dim * self.seq_len, self.hidden_dim),
                                              Dropout(dropout),
                                              nn.LayerNorm(self.hidden_dim))  #
        self.local_attn_out = nn.Sequential(nn.LayerNorm(self.seq_len * self.hidden_dim),
                                            Linear(self.hidden_dim * self.seq_len, self.hidden_dim),
                                            Dropout(dropout),
                                            nn.LayerNorm(self.hidden_dim))  #

        self.dynamic_attn_layers = nn.ModuleList([StaticAttention(self.head_num, self.hidden_dim)])
        for _ in range(self.attn_layer - 1):
            self.dynamic_attn_layers.append(StaticAttention(self.head_num, self.hidden_dim))
        # self.dynamic_attention = StaticAttention(self.head_num, self.hidden_dim)
        self.dynamic_attn_out_fc = nn.Sequential(Linear(self.attn_layer * self.hidden_dim, 2 * self.hidden_dim),
                                                 Tanh(),
                                                 Dropout(self.dropout),
                                                 Linear(2 * self.hidden_dim, self.hidden_dim),
                                                 Tanh(),
                                                 Dropout(self.dropout),
                                                 nn.LayerNorm(self.hidden_dim))

        self.stations_mlp = nn.Sequential(Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                                          Tanh(),
                                          Dropout(self.dropout),
                                          Linear(self.hidden_dim * 2, self.hidden_dim),
                                          Tanh(),
                                          Dropout(self.dropout),
                                          nn.LayerNorm(self.hidden_dim))

        self.ds_attention_mlp = nn.Sequential(Linear(self.hidden_dim * 2, self.hidden_dim),
                                              Tanh(),
                                              nn.LayerNorm(self.hidden_dim))

        self.pred_mlp = Sequential(Linear(3 * self.hidden_dim, self.hidden_dim * 2),
                                   Tanh(),
                                   Linear(2 * self.hidden_dim, self.seq_len),
                                   Tanh())

    def forward(self, station_dist, local_node, local_features, local_emb, station_nodes,
                station_features,
                station_emb):
        # local_node: batch_size, 1(node_num), 16(node_features_num)
        # local_features: batch_size, seq_len, 1(node_num), 4(features_num)
        # local_emb: batch_size, seq_len,1(node_num), 5(features_num)  station_emb: 6
        batch_size = local_node.size(0)
        seq_len = station_features.size(1)
        station_num = station_dist.size(1)

        month_emb = self.month_emb(station_emb[:, :, :, -3] - 1)
        day_emb = self.day_emb(station_emb[:, :, :, -2])
        hour_emb = self.hour_emb(station_emb[:, :, :, -1])
        station_pm25_emb = self.pm25_emb(station_emb[:, :, :, 0])
        station_weather_emb = self.weather_emb(station_emb[:, :, :, 1])
        station_wind_direc_emb = self.wind_direc_emb(station_emb[:, :, :, 2])
        local_weather_emb = self.weather_emb(local_emb[:, :, :, 0])
        local_wind_direc_emb = self.wind_direc_emb(local_emb[:, :, :, 1])
        station_dist_emb = self.station_dist_mlp(torch.cat((station_dist, station_nodes), dim=-1))

        # 站点静态信息提取
        station_nodes_emb = self.static_mlp(station_nodes)
        local_node_emb = self.static_mlp(local_node)

        # 站点与预测点静态注意力提取
        static_attention = None
        local_attn_outs = []
        for layer in self.static_attn_layers:
            static_attention, local_node_emb = layer(local_node_emb, station_dist_emb,
                                                     station_nodes_emb)
            local_attn_outs.append(local_node_emb)
        local_attn_outs = (torch.stack(local_attn_outs, dim=0).view(self.attn_layer, batch_size, -1)
                           .permute(1, 0, 2).contiguous().view(batch_size, -1))
        static_output = self.static_attn_out_fc(local_attn_outs)

        station_features = torch.cat((station_features, station_weather_emb,
                                      station_wind_direc_emb, station_pm25_emb,
                                      hour_emb, day_emb, month_emb), dim=-1)
        local_features = torch.cat((local_features, local_weather_emb, local_wind_direc_emb,
                                    hour_emb[:, :, [0]], day_emb[:, :, [0]], month_emb[:, :, [0]]), dim=-1)
        # 站点动态信息提取
        local_features_emb = self.local_dynamic_mlp(local_features)
        station_features_emb = self.station_dynamic_mlp(station_features)
        # VERSION1: 经过测试具有稳定的训练效果
        # station_mul = []
        # features_emb = []
        # for i in range(station_num):
        #     node_emb = station_nodes_emb[:, i]
        #     feature_emb = station_features_emb[:, :, i]
        #
        #     _, feature_emb = self.station_dynamic_encoder(feature_emb)
        #     feature_emb = feature_emb[-1]
        #
        #     station_feature = torch.cat((node_emb, feature_emb), dim=-1)
        #     station_feature = self.stations_mlp(station_feature)
        #
        #     features_emb.append(feature_emb)
        #     station_mul.append(station_feature)
        #
        # station_features_emb = torch.stack(features_emb, dim=1)
        # station_mul = torch.stack(station_mul, dim=1)

        # VERSION2: 比version1具有更好的训练表现
        # _, features_emb = self.station_dynamic_encoder(station_features_emb.transpose(1, 2)
        #                                                .contiguous().view(-1, seq_len, self.hidden_dim))
        # station_features_emb = features_emb[-1].view(batch_size, station_num, -1)
        # stations_features = torch.cat((station_nodes_emb, station_features_emb), dim=-1)
        # station_mul = self.stations_mlp(stations_features.view(batch_size * station_num, -1))
        # station_mul = station_mul.view(batch_size, station_num, -1)
        # _, local_features_emb = self.local_dynamic_encoder(local_features_emb)
        # local_features_emb = local_features_emb[-1].unsqueeze(1)

        # VERSION 3: 注意力叠加
        for local_layer, station_layer in zip(self.local_dynamic_attn_layers, self.station_dynamic_attn_layers):
            local_features_emb = local_layer(local_features_emb, local_features_emb, local_features_emb)
            station_features_emb = station_layer(station_pm25_emb, station_features_emb,
                                                 station_features_emb)
        station_features_emb = self.station_attn_out(
            station_features_emb.view(batch_size, -1, self.seq_len * self.hidden_dim))
        local_features_emb = self.local_attn_out(
            local_features_emb.view(batch_size, -1, self.seq_len * self.hidden_dim))

        # 站点与预测点动态注意力提取
        dynamic_attention = None
        local_feature_outs = []
        for layer in self.dynamic_attn_layers:
            dynamic_attention, local_features_emb = layer(local_features_emb,
                                                          station_features_emb + station_dist_emb,
                                                          station_features_emb)
            local_feature_outs.append(local_features_emb)
        local_feature_outs = (torch.stack(local_feature_outs, dim=0).view(self.attn_layer, batch_size, -1)
                              .permute(1, 0, 2).contiguous().view(batch_size, -1))
        dynamic_output = self.dynamic_attn_out_fc(local_feature_outs)

        # 计算注意力与站点特征关系
        stations_ds_features = self.stations_mlp(torch.cat((station_features_emb, station_nodes_emb), dim=-1))
        attn_features = self.ds_attention_mlp(torch.cat(
            (static_attention @ stations_ds_features, dynamic_attention @ stations_ds_features), dim=-1)
                                              .view(batch_size, -1))
        # attn_features = torch.sum(attn_features, dim=1).unsqueeze(1)  # -1, 1, hidden_dim
        pred_in = torch.cat((static_output, dynamic_output, attn_features), dim=-1)
        pred_out = self.pred_mlp(pred_in)
        pred = pred_out.view(batch_size, seq_len, -1)

        return pred
