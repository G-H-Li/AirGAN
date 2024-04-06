import numpy as np
import torch
from torch import nn
from torch.nn import GRU, Sequential, Linear, Dropout, Tanh, SiLU, ReLU, Sigmoid
from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits, cross_entropy


class NBSTLoss(nn.Module):
    def __init__(self, pm25_std, pm25_mean, alpha, device):
        super(NBSTLoss, self).__init__()
        self.pm25_std = pm25_std
        self.pm25_mean = pm25_mean
        self.alpha = alpha
        self.device = device

    def forward(self, output, target):
        loss1 = l1_loss(output, target)
        loss2 = cross_entropy(torch.cosine_similarity(output, target, dim=1),
                              torch.cosine_similarity(target, target, dim=1))
        return loss1 + self.alpha * loss2


class PreNorm(nn.Module):
    # Pre Normalization in Transformer
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # FFN in Transformer
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class StaticAttention(nn.Module):
    def __init__(self, head_num, input_size, dropout):
        super(StaticAttention, self).__init__()
        if input_size % head_num != 0:
            raise ValueError("input_size must be divisible by head_num")
        self.input_size = input_size
        self.head_num = head_num
        self.dropout = dropout
        head_dim = input_size // head_num
        self.scale = head_dim ** -0.5

        self.multi_head_q = Linear(input_size, input_size)  #
        self.multi_head_k = Linear(input_size, input_size)
        self.multi_head_v = Linear(input_size, input_size)

        self.relative_alpha = nn.Parameter(torch.randn(head_num, 1, 1))

        self.multi_head_out = nn.Sequential(nn.Linear(input_size, input_size),
                                            nn.Dropout(dropout),
                                            PreNorm(input_size, FeedForward(input_size, 2 * input_size, dropout)))

    def forward(self, q, k, v, mask=None):
        # q: batch_size, q_dim, input_size
        # kv: batch_size, site_num, input_size
        # mask: batch_size, q_dim, site_num
        B, S, C = k.shape
        # q: batch_size, head_num, q_dim, input_size//head_num
        q = self.multi_head_q(q).reshape(B, -1, self.head_num, C // self.head_num).permute(0, 2, 1, 3)
        k = self.multi_head_k(k).reshape(B, -1, self.head_num, C // self.head_num).permute(0, 2, 1, 3)
        v = self.multi_head_v(v).reshape(B, -1, self.head_num, C // self.head_num).permute(0, 2, 1, 3)
        # k, v: batch_size, head_num, site_num, input_size//head_num

        attn = (q @ k.transpose(-1, -2)) * self.scale
        if mask is not None:
            mask = mask.reshape(B, 1, -1, S).repeat(1, self.head_num, 1, 1)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)  # batch_size, head_num, q_dim, site_num
        output = attn @ v
        output = output + q * self.relative_alpha  # batch_size, head_num, q_dim, input_size//head_num

        output = output.transpose(1, 2).reshape(B, -1, C)

        output = self.multi_head_out(output)
        return output


class DynamicSelfAttention(nn.Module):
    def __init__(self, seq_len, q_dim, k_dim, v_dim, hidden_dim, dropout):
        super(DynamicSelfAttention, self).__init__()
        self.seq_len = seq_len
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        head_dim = hidden_dim // seq_len
        self.scale = head_dim ** -0.5

        self.q_linear = Linear(self.q_dim, self.hidden_dim)
        self.k_linear = Linear(self.k_dim, self.hidden_dim)
        self.v_linear = Linear(self.v_dim, self.hidden_dim)

        self.relative_alpha = nn.Parameter(torch.randn(seq_len, 1, 1))

        self.multi_head_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.Dropout(dropout),
                                            PreNorm(hidden_dim, FeedForward(hidden_dim, 2 * hidden_dim, dropout)))

        self.seq_out = nn.Sequential(nn.Linear(seq_len * hidden_dim, hidden_dim),
                                     SiLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, hidden_dim))

    def forward(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out + q * self.relative_alpha

        out = self.multi_head_out(out)
        out = self.seq_out(out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), -1))
        return out


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
        self.gru_layer = 2

        self.sigma_d = nn.Parameter(2 * torch.rand(1, device=self.device, requires_grad=True) - 1)
        self.sigma_r = nn.Parameter(2 * torch.rand(1, device=self.device, requires_grad=True) - 1)

        # self.pm25_emb = nn.Embedding(6, self.emb_dim)
        self.weather_emb = nn.Embedding(17, self.emb_dim)
        self.wind_direc_emb = nn.Embedding(25, self.emb_dim)
        self.month_emb = nn.Embedding(12, self.emb_dim)
        self.day_emb = nn.Embedding(7, self.emb_dim)
        self.hour_emb = nn.Embedding(24, self.emb_dim)

        self.static_emb = nn.Sequential(Linear(self.node_in_dim, self.hidden_dim),
                                        SiLU(),
                                        Linear(self.hidden_dim, self.hidden_dim))
        self.static_attn_layers = nn.ModuleList([StaticAttention(self.head_num, self.hidden_dim, self.dropout)])
        for _ in range(self.attn_layer - 1):
            self.static_attn_layers.append(StaticAttention(self.head_num, self.hidden_dim, self.dropout))

        # 时序时间尺度特征编码
        self.dynamic_emb = nn.GRU(self.in_dim + self.emb_dim * 5, self.hidden_dim,
                                  dropout=self.dropout, batch_first=True, num_layers=self.gru_layer)

        # 提取目标点位与监测点位之间的注意力关系
        self.dynamic_attn_layers = nn.ModuleList([StaticAttention(self.head_num, self.hidden_dim, self.dropout)])
        for _ in range(self.attn_layer - 1):
            self.dynamic_attn_layers.append(StaticAttention(self.head_num, self.hidden_dim, self.dropout))

        # pm25相关关系提取
        self.aqi_dynamic_layers = DynamicSelfAttention(self.seq_len, 1, self.hidden_dim,
                                                       self.hidden_dim, self.hidden_dim, self.dropout)
        self.aqi_static_layers = DynamicSelfAttention(self.seq_len, 1, self.hidden_dim,
                                                      self.hidden_dim, self.hidden_dim, self.dropout)

        self.pred_mlp = Sequential(Linear(2 * self.hidden_dim, self.hidden_dim),
                                   SiLU(),
                                   Dropout(self.dropout),
                                   Linear(self.hidden_dim, self.seq_len))

    def cal_pearson_corr(self, x, y):
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        mean_y = torch.mean(y, dim=-1, keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1))
        r = r_num / r_den
        return r

    def forward(self, station_dist, pm25_hist,
                local_node, local_features, local_emb,
                station_nodes, station_features, station_emb):
        # local_node: batch_size, 1(node_num), 16(node_features_num)
        # local_features: batch_size, seq_len, 1(node_num), 4(features_num)
        # local_emb: batch_size, seq_len,1(node_num), 5(features_num)  station_emb: 6
        batch_size = local_node.size(0)
        seq_len = station_features.size(1)
        station_num = station_dist.size(1)

        month_emb = self.month_emb(station_emb[:, :, :, -3] - 1)
        day_emb = self.day_emb(station_emb[:, :, :, -2])
        hour_emb = self.hour_emb(station_emb[:, :, :, -1])
        # station_pm25_emb = self.pm25_emb(station_emb[:, :, :, 0])
        station_weather_emb = self.weather_emb(station_emb[:, :, :, -5])
        station_wind_direc_emb = self.wind_direc_emb(station_emb[:, :, :, -4])
        local_weather_emb = self.weather_emb(local_emb[:, :, :, 0])
        local_wind_direc_emb = self.wind_direc_emb(local_emb[:, :, :, 1])
        # 站点动态信息提取
        station_features = torch.cat((station_features, station_weather_emb, station_wind_direc_emb,
                                      hour_emb, day_emb, month_emb), dim=-1)
        local_features = torch.cat((local_features, local_weather_emb, local_wind_direc_emb,
                                    hour_emb[:, :, [0]], day_emb[:, :, [0]], month_emb[:, :, [0]]), dim=-1)

        # 静态站点mask提取
        station_dist = station_dist[:, :, 0]
        static_mask = []
        for i in range(station_num):
            dist = station_dist[:, i]
            dist = torch.square(dist / self.sigma_d)
            pearson = self.cal_pearson_corr(local_node.squeeze(1), station_nodes[:, i])
            pearson = torch.square(pearson / self.sigma_r)
            static_mask.append(torch.log(dist + pearson))

        static_mask = torch.stack(static_mask, dim=1)
        static_mask = (static_mask <= 0).view(batch_size, station_num, 1).transpose(1, 2)

        # 站点与预测点静态注意力提取
        # 站点静态信息提取
        station_nodes_emb = self.static_emb(station_nodes)
        local_node_emb = self.static_emb(local_node)

        # 站点与预测点动态注意力提取
        _, local_features_hn = self.dynamic_emb(local_features.squeeze(2))
        local_features_emb = local_features_hn[-1].unsqueeze(1)  # batch_size, 1, hidden_dim
        station_features_out = []
        station_features_emb = []
        for i in range(station_num):
            station_features_output, station_hn = self.dynamic_emb(station_features[:, :, i])
            station_features_emb.append(station_hn[-1])
            station_features_out.append(station_features_output)
        station_features_emb = torch.stack(station_features_emb, dim=1)  # batch_size, station_num, hidden_dim
        station_features_out = torch.stack(station_features_out, dim=2)  # batch_size, seq_len, station_num, hidden_dim

        # AQI相关关系提取
        aqi_dynamic_cors = self.aqi_dynamic_layers(pm25_hist, station_features_out, station_features_out)
        aqi_static_cors = self.aqi_static_layers(pm25_hist,
                                                 station_nodes_emb.unsqueeze(1).repeat(1, seq_len, 1, 1),
                                                 station_nodes_emb.unsqueeze(1).repeat(1, seq_len, 1, 1))

        static_out = local_node_emb
        dynamic_out = local_features_emb
        for d_layer, s_layer in zip(self.dynamic_attn_layers, self.static_attn_layers):
            dynamic_out = d_layer(dynamic_out, aqi_dynamic_cors, station_features_emb)
            static_out = s_layer(static_out, aqi_static_cors, station_nodes_emb, static_mask)
        dynamic_output = static_out.squeeze(1)
        static_output = static_out.squeeze(1)

        # 预测
        pred_in = torch.cat((static_output, dynamic_output), dim=-1)
        pred_out = self.pred_mlp(pred_in)
        pred = pred_out.view(batch_size, seq_len, -1)

        return pred
