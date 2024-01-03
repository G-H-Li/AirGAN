import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer
from torch.nn.parameter import Parameter
from torch_scatter import scatter_mean


def batch_input(x, edge_w, edge_index):
    sta_num = x.shape[1]  # group_num/ city_num
    x = x.reshape(-1, x.shape[-1])  # (group_num*batch_size, loc_em+x_em)
    edge_w = edge_w.reshape(-1, edge_w.shape[-1])  # (batch_size* 15^2-15, edge_h)
    for i in range(edge_index.size(0)):
        edge_index[i, :] = torch.add(edge_index[i, :], i * sta_num)

    edge_index = edge_index.transpose(0, 1)
    edge_index = edge_index.reshape(2, -1)  # (2, batch_size* 15^2-15)
    return x, edge_w, edge_index


class GAGNN(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr,
                 city_loc, group_num):
        super(GAGNN, self).__init__()

        self.device = device
        self.city_num = city_num
        self.group_num = group_num
        self.pred_len = pred_len
        self.city_loc = torch.Tensor(np.float32(city_loc)).to(self.device)
        self.hist_len = hist_len
        self.batch_size = batch_size
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_w = torch.Tensor(np.float32(edge_attr[:, 0])).to(self.device)

        self.in_dim = in_dim
        self.edge_h = 12
        self.gnn_layer = 2
        self.gnn_h = 32
        self.x_em = 32
        self.loc_em = 12
        self.date_em = 4

        self.w = Parameter(torch.randn(city_num, group_num).to(device, non_blocking=True), requires_grad=True)

        self.encoder = TransformerEncoderLayer(self.in_dim, nhead=4, dim_feedforward=256)
        self.x_embed = Lin(self.hist_len * self.in_dim, self.x_em)
        self.loc_embed = Lin(2, self.loc_em)
        self.u_embed1 = nn.Embedding(12, self.date_em)  # month
        self.u_embed2 = nn.Embedding(7, self.date_em)  # week
        self.u_embed3 = nn.Embedding(24, self.date_em)  # hour

        self.edge_inf = Seq(Lin(self.x_em * 2 + self.date_em * 3 + self.loc_em * 2, self.edge_h), ReLU(inplace=True))
        self.group_gnn = nn.ModuleList([NodeModel(self.x_em + self.loc_em, self.edge_h, self.gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(self.gnn_h, self.edge_h, self.gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(self.x_em + self.gnn_h, 1, self.gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(self.gnn_h, 1, self.gnn_h))

        self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, city_num, group_num, device)
        self.predMLP = Seq(Lin(self.gnn_h, 16),
                           ReLU(inplace=True),
                           Lin(16, self.pred_len),
                           ReLU(inplace=True))

    def forward(self, pm25_hist, feature, time_feature):
        hist_feature = feature[:, :self.hist_len].transpose(1, 2)
        x = hist_feature.reshape(-1, feature.shape[2], feature.shape[3])
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = x.reshape(-1, self.city_num, self.hist_len * x.shape[-1])
        x = self.x_embed(x)

        # graph pooling
        w = F.softmax(self.w)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)

        city_loc = self.city_loc.unsqueeze(dim=0)
        city_loc = city_loc.repeat_interleave(self.batch_size, dim=0)

        loc = self.loc_embed(city_loc)
        x_loc = torch.cat([x, loc], dim=-1)
        g_x = torch.bmm(w1, x_loc)

        # group gnn
        u_month = time_feature[:, -1].to(torch.int32).squeeze() - 1
        u_day = time_feature[:, -2].to(torch.int32).squeeze() - 1
        u_hour = time_feature[:, -3].to(torch.int32).squeeze()
        u_em1 = self.u_embed1(u_month)
        u_em2 = self.u_embed2(u_day)
        u_em3 = self.u_embed3(u_hour)
        u_em = torch.cat([u_em1, u_em2, u_em3], dim=-1).to(torch.float)
        for i in range(self.group_num):
            for j in range(self.group_num):
                if i == j:
                    continue
                g_edge_input = torch.cat([g_x[:, i], g_x[:, j], u_em], dim=-1)
                tmp_g_edge_w = self.edge_inf(g_edge_input)
                tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)
                tmp_g_edge_index = torch.tensor([i, j]).unsqueeze(dim=0).to(self.device, non_blocking=True)
                if i == 0 and j == 1:
                    g_edge_w = tmp_g_edge_w
                    g_edge_index = tmp_g_edge_index
                else:
                    g_edge_w = torch.cat([g_edge_w, tmp_g_edge_w], dim=0)
                    g_edge_index = torch.cat([g_edge_index, tmp_g_edge_index], dim=0)
        g_edge_w = g_edge_w.transpose(0, 1)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)
        g_edge_index = g_edge_index.transpose(1, 2)
        g_x, g_edge_w, g_edge_index = batch_input(g_x, g_edge_w, g_edge_index)
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)

        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2, g_x)
        new_x = torch.cat([x, new_x], dim=-1)

        edge_w = self.edge_w.unsqueeze(dim=0)
        edge_w = edge_w.repeat_interleave(self.batch_size, dim=0)
        edge_index = self.edge_index.unsqueeze(dim=0)
        edge_index = edge_index.repeat_interleave(self.batch_size, dim=0)

        edge_w = edge_w.unsqueeze(dim=-1)
        new_x, edge_w, edge_index = batch_input(new_x, edge_w, edge_index)

        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)
        new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
        res = self.predMLP(new_x)
        res = res.unsqueeze(dim=-1)
        res = res.reshape(-1, self.pred_len, self.city_num, res.shape[-1])

        return res


class DecoderModule(nn.Module):
    def __init__(self, x_em, edge_h, gnn_h, gnn_layer, city_num, group_num, device):
        super(DecoderModule, self).__init__()
        self.device = device
        self.city_num = city_num
        self.group_num = group_num
        self.gnn_layer = gnn_layer
        self.x_embed = Lin(gnn_h, x_em)
        self.group_gnn = nn.ModuleList([NodeModel(x_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, edge_h, gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em + gnn_h, 1, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(gnn_h, 1, gnn_h))

    def forward(self, x, trans_w, g_edge_index, g_edge_w, edge_index, edge_w):
        x = self.x_embed(x)
        x = x.reshape(-1, self.city_num, x.shape[-1])
        w = Parameter(trans_w, requires_grad=False).to(self.device, non_blocking=True)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        g_x = torch.bmm(w1, x)
        g_x = g_x.reshape(-1, g_x.shape[-1])
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)
        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2, g_x)
        new_x = torch.cat([x, new_x], dim=-1)
        new_x = new_x.reshape(-1, new_x.shape[-1])
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)

        return new_x


class NodeModel(torch.nn.Module):
    def __init__(self, node_h, edge_h, gnn_h):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h + edge_h, gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h, gnn_h), ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
