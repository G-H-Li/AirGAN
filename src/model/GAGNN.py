import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer
from torch.nn.parameter import Parameter
from torch_scatter import scatter_mean

TIME_WINDOW = 24 / 3


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
    def __init__(self, mode, encoder, w_init, w, x_em, date_em, loc_em, edge_h, gnn_h, gnn_layer, city_num, group_num,
                 pred_step, device):
        super(GAGNN, self).__init__()
        self.device = device
        self.mode = mode
        self.encoder = encoder
        self.w_init = w_init
        self.city_num = city_num
        self.group_num = group_num
        self.edge_h = edge_h
        self.gnn_layer = gnn_layer
        self.pred_step = pred_step
        if self.encoder == 'self':
            self.encoder_layer = TransformerEncoderLayer(8, nhead=4, dim_feedforward=256)
            # self.x_embed = Lin(8, x_em)
            self.x_embed = Lin(24 * 8, x_em)

        elif self.encoder == 'lstm':
            self.input_LSTM = nn.LSTM(8, x_em, num_layers=1, batch_first=True)
        if self.w_init == 'rand':
            self.w = Parameter(torch.randn(city_num, group_num).to(device, non_blocking=True), requires_grad=True)
        elif self.w_init == 'group':
            self.w = Parameter(w, requires_grad=True)
        self.loc_embed = Lin(2, loc_em)
        self.u_embed1 = nn.Embedding(12, date_em)  # month
        self.u_embed2 = nn.Embedding(7, date_em)  # week
        self.u_embed3 = nn.Embedding(24, date_em)  # hour
        self.edge_inf = Seq(Lin(x_em * 2 + date_em * 3 + loc_em * 2, edge_h), ReLU(inplace=True))
        self.group_gnn = nn.ModuleList([NodeModel(x_em + loc_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, edge_h, gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em + gnn_h, 1, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(gnn_h, 1, gnn_h))
        if self.mode == 'ag':
            self.decoder = DecoderModule(x_em, edge_h, gnn_h, gnn_layer, city_num, group_num, device)
            self.predMLP = Seq(Lin(gnn_h, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))
        if self.mode == 'full':
            self.decoder = DecoderModule(x_em, edge_h, gnn_h, gnn_layer, city_num, group_num, device)
            self.predMLP = Seq(Lin(gnn_h, 16), ReLU(inplace=True), Lin(16, self.pred_step), ReLU(inplace=True))

    def forward(self, x, u, edge_index, edge_w, loc):
        x = x.reshape(-1, x.shape[2], x.shape[3])  # 保留小时和气象维度
        if self.encoder == 'self':
            # [S,B,E]
            # print(x.shape)
            x = x.transpose(0, 1)
            x = self.encoder_layer(x)
            x = x.transpose(0, 1)
            # print(x.shape)
            x = x.reshape(-1, self.city_num, 24 * x.shape[-1])
            x = self.x_embed(x)  # （batch_size, city_num, x_em）

        elif self.encoder == 'lstm':
            _, (x, _) = self.input_LSTM(x)
            x = x.reshape(-1, self.city_num, x.shape[-1])

        # graph pooling
        # print(self.w[10])
        w = F.softmax(self.w)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)  # (batch_size, group_num, city_num)
        # print(w.shape,x.shape)
        # print(loc.shape)
        loc = self.loc_embed(loc)  # (batch_size, city_num, loc_em)
        x_loc = torch.cat([x, loc], dim=-1)  # (batch_size, city_num, loc_em+x_em)
        g_x = torch.bmm(w1, x_loc)  # (batch_size, group_num, loc_em+x_em))
        # print(g_x.shape)

        # group gnn
        u_em1 = self.u_embed1(u[:, 0])
        u_em2 = self.u_embed2(u[:, 1])
        u_em3 = self.u_embed3(u[:, 2])
        u_em = torch.cat([u_em1, u_em2, u_em3], dim=-1)  # (batch_size, date_em*3)
        # print(u_em.shape)
        for i in range(self.group_num):
            for j in range(self.group_num):
                if i == j: continue
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
        g_edge_w = g_edge_w.transpose(0, 1)  # (batch_size, group_num^2-group_num, edge_h)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)  # (batch_size, group_num^2-group_num, 2)
        g_edge_index = g_edge_index.transpose(1, 2)  # (batch_size, 2, group_num^2-group_num)
        g_x, g_edge_w, g_edge_index = batch_input(g_x, g_edge_w, g_edge_index)
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)

        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])  # (batch_size, group_num, )
        # print(g_x.shape,self.w.shape)
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)  # (batch_size, city_num, group_num)
        new_x = torch.bmm(w2, g_x)
        # print(new_x.shape,x.shape)
        new_x = torch.cat([x, new_x], dim=-1)
        edge_w = edge_w.unsqueeze(dim=-1)
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        new_x, edge_w, edge_index = batch_input(new_x, edge_w, edge_index)
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)
        # print(new_x.shape)
        if self.mode == 'ag':
            for i in range(self.pred_step):
                new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
                tmp_res = self.predMLP(new_x)
                tmp_res = tmp_res.reshape(-1, self.city_num)
                tmp_res = tmp_res.unsqueeze(dim=-1)
                if i == 0:
                    res = tmp_res
                else:
                    res = torch.cat([res, tmp_res], dim=-1)
        if self.mode == 'full':
            new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
            res = self.predMLP(new_x)
            res = res.reshape(-1, self.city_num, self.pred_step)

        return res


class GAGNNUpdate(nn.Module):
    def __init__(self, pred_len, in_dim, city_num, device, group_num, city_loc):
        super(GAGNNUpdate, self).__init__()

        self.device = device
        self.city_num = city_num
        self.group_num = group_num
        self.pred_len = pred_len
        self.city_loc = city_loc

        self.in_dim = in_dim
        self.edge_h = 12
        self.gnn_layer = 2
        self.gnn_h = 32
        self.x_em = 32
        self.loc_em = 12
        self.date_em = 4

        self.w = Parameter(torch.randn(city_num, group_num).to(device, non_blocking=True), requires_grad=True)

        self.encoder = TransformerEncoderLayer(self.in_dim, nhead=4, dim_feedforward=256)
        self.x_embed = Lin(TIME_WINDOW * self.in_dim, self.x_em)
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

    def forward(self, x, edge_index, edge_w):
        x = x.reshape(-1, x.shape[2], x.shape[3])
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = x.reshape(-1, self.city_num, TIME_WINDOW * x.shape[-1])
        x = self.x_embed(x)

        # graph pooling
        w = F.softmax(self.w)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        loc = self.loc_embed(self.city_loc)
        x_loc = torch.cat([x, loc], dim=-1)
        g_x = torch.bmm(w1, x_loc)

        # group gnn
        u_em1 = self.u_embed1(x[:, -3])
        u_em2 = self.u_embed2(x[:, -4])
        u_em3 = self.u_embed3(x[:, -5])
        u_em = torch.cat([u_em1, u_em2, u_em3], dim=-1)
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
        edge_w = edge_w.unsqueeze(dim=-1)
        new_x, edge_w, edge_index = batch_input(new_x, edge_w, edge_index)

        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x, edge_index, edge_w)
        new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
        res = self.predMLP(new_x)
        res = res.reshape(-1, self.city_num, self.pred_len)

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
