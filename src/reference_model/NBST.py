import torch
from torch import nn
from torch.nn import Sequential, Linear, Dropout, Tanh, SiLU, ReLU
from torch.nn.functional import softmax


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
        self.scale = input_size ** -0.5

        self.multi_head_q = Linear(input_size, input_size)  #
        self.multi_head_k = Linear(input_size, input_size)
        self.multi_head_v = Linear(input_size, input_size)

        self.relative_alpha = nn.Parameter(torch.randn(head_num, 1, 1))

        self.multi_head_out = nn.Sequential(nn.Linear(input_size, input_size),
                                            nn.Dropout(dropout),
                                            PreNorm(input_size, FeedForward(input_size, input_size, dropout)))
        self.attn_out = nn.Sequential(nn.Linear(head_num, 1),
                                      ReLU())

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
        # attn = self.attn_out(attn.permute(0, 2, 3, 1)).reshape(B, -1, S)
        return output


# class DynamicSelfAttention(nn.Module):
#     def __init__(self, seq_len, q_dim, k_dim, v_dim, hidden_dim, dropout):
#         super(DynamicSelfAttention, self).__init__()
#         self.seq_len = seq_len
#         self.q_dim = q_dim
#         self.k_dim = k_dim
#         self.v_dim = v_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         head_dim = hidden_dim // seq_len
#         self.scale = head_dim ** -0.5
#
#         self.q_linear = Linear(self.q_dim, self.hidden_dim)
#         self.k_linear = Linear(self.k_dim, self.hidden_dim)
#         self.v_linear = Linear(self.v_dim, self.hidden_dim)
#
#         self.relative_alpha = nn.Parameter(torch.randn(seq_len, 1, 1))
#
#         self.multi_head_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
#                                             nn.Dropout(dropout),
#                                             PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout)))
#
#         self.seq_out = nn.Sequential(nn.Linear(seq_len * hidden_dim, hidden_dim),
#                                      SiLU(),
#                                      nn.Dropout(dropout),
#                                      nn.Linear(hidden_dim, hidden_dim))
#
#     def forward(self, q, k, v):
#         q = self.q_linear(q)
#         k = self.k_linear(k)
#         v = self.v_linear(v)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#
#         attn = attn.softmax(dim=-1)
#         out = attn @ v
#         out = out + q * self.relative_alpha
#
#         out = self.multi_head_out(out)
#         out = self.seq_out(out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), -1))
#         return out


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, lstm_layers, dropout):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout, num_layers=lstm_layers)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, (h, _) = self.lstm(x)
        return x, h[-1]


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

        self.sigma_h = nn.Parameter(torch.randn(1, device=self.device, requires_grad=True))

        # self.pm25_emb = nn.Embedding(6, self.emb_dim)
        self.weather_emb = nn.Embedding(17, self.emb_dim)
        self.wind_direc_emb = nn.Embedding(9, self.emb_dim)
        self.month_emb = nn.Embedding(12, self.emb_dim)
        self.day_emb = nn.Embedding(7, self.emb_dim)
        self.hour_emb = nn.Embedding(24, self.emb_dim)

        self.static_mlp = nn.Sequential(Linear(self.node_in_dim, self.hidden_dim * 2),
                                        SiLU(),
                                        Dropout(self.dropout),
                                        Linear(self.hidden_dim * 2, self.hidden_dim),
                                        SiLU())  #
        self.static_query_fc = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                             ReLU())
        self.static_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.Softmax(dim=-1))
        self.static_attn_layer = nn.ModuleList([StaticAttention(self.head_num, self.hidden_dim, self.dropout)
                                                for _ in range(self.attn_layer)])
        self.static_attn_fusion = nn.Sequential(Linear(self.hidden_dim * (1 + self.attn_layer), 2 * self.hidden_dim),
                                                ReLU(),
                                                Dropout(self.dropout),
                                                Linear(2 * self.hidden_dim, self.hidden_dim),
                                                ReLU())

        self.dynamic_station_lstm = ConvLSTM(self.in_dim + 5 * self.emb_dim, self.hidden_dim,
                                             3, self.gru_layer, self.dropout)
        self.dynamic_local_lstm = ConvLSTM(self.in_dim - 1 + 5 * self.emb_dim, self.hidden_dim,
                                           3, self.gru_layer, self.dropout)

        self.dynamic_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_dim, self.hidden_dim),
                                               nn.Softmax(dim=-1))
        self.dynamic_attn_layer = nn.ModuleList([StaticAttention(self.head_num, self.hidden_dim, self.dropout)
                                                 for _ in range(self.attn_layer)])
        self.dynamic_attn_fusion = nn.Sequential(Linear(self.hidden_dim * (1 + self.attn_layer), 2 * self.hidden_dim),
                                                 ReLU(),
                                                 Dropout(self.dropout),
                                                 Linear(2 * self.hidden_dim, self.hidden_dim),
                                                 ReLU())

        self.pred_infer_mlp = Sequential(Linear(2 * self.hidden_dim + 1, self.hidden_dim),
                                         ReLU(),
                                         Linear(self.hidden_dim, 1),
                                         Tanh())

    def forward(self, station_dist,
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
        station_weather_emb = self.weather_emb(station_emb[:, :, :, -5])
        station_wind_direc_emb = self.wind_direc_emb(station_emb[:, :, :, -4])
        local_weather_emb = self.weather_emb(local_emb[:, :, :, 0])
        local_wind_direc_emb = self.wind_direc_emb(local_emb[:, :, :, 1])
        # 站点动态信息提取
        station_features = torch.cat((station_features, station_weather_emb, station_wind_direc_emb,
                                      hour_emb, day_emb, month_emb), dim=-1)
        local_features = torch.cat((local_features, local_weather_emb, local_wind_direc_emb,
                                    hour_emb[:, :, [0]], day_emb[:, :, [0]], month_emb[:, :, [0]]), dim=-1)

        # static_query = self.cal_static_corr(station_dist[:, :, 0], local_node.repeat(1, station_num, 1), station_nodes)
        # # 站点静态信息提取
        station_nodes_emb = self.static_mlp(station_nodes)
        local_node_emb = self.static_mlp(local_node)
        # static_query = self.static_query_fc(torch.cat((local_node_emb, static_query @ station_nodes_emb), dim=-1))
        static_fusion = self.static_attention(torch.cat((local_node_emb, station_nodes_emb), dim=1))
        static_output = [local_node_emb.squeeze(1)]
        static_out = local_node_emb
        for layer in self.static_attn_layer:
            static_out = layer(static_out, static_fusion, static_fusion)
            static_output.append(static_out.squeeze(1))
        static_out = torch.cat(static_output, dim=-1)
        static_output = self.static_attn_fusion(static_out)

        _, local_dynamic_h = self.dynamic_local_lstm(local_features.squeeze(2))
        station_dynamic_emb = []
        for i in range(station_num):
            station_dynamic_out, station_dynamic_h = self.dynamic_station_lstm(station_features[:, :, i])
            station_dynamic_emb.append(station_dynamic_h)
        station_dynamic_emb = torch.stack(station_dynamic_emb, dim=1)
        nodes = torch.cat((local_dynamic_h.unsqueeze(1), station_dynamic_emb), dim=1)
        dynamic_weight = self.dynamic_attention(nodes).squeeze(-1)
        # dynamic_gcn_out = self.dynamic_gcn(nodes, edge_index, dynamic_weight)
        dynamic_output = [local_dynamic_h]
        dynamic_out = local_dynamic_h.unsqueeze(1)
        for layer in self.dynamic_attn_layer:
            dynamic_out = layer(dynamic_out, dynamic_weight, dynamic_weight)
            dynamic_output.append(dynamic_out.squeeze(1))
        dynamic_out = torch.cat(dynamic_output, dim=-1)
        dynamic_out = self.dynamic_attn_fusion(dynamic_out)

        preds = []
        h0 = self.sigma_h * torch.ones((batch_size, 1), device=self.device)
        for i in range(seq_len):
            pred_in = torch.cat((static_output, dynamic_out, h0), dim=-1)
            pred_out = self.pred_infer_mlp(pred_in)
            preds.append(pred_out)
            h0 = pred_out
        pred_out = torch.stack(preds, dim=1)
        pred = pred_out.view(batch_size, seq_len, -1)

        return pred
