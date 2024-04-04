import torch
from torch import nn
from torch_geometric.nn import GCNConv


class MCAM(nn.Module):
    def __init__(self, seq_len, in_dim, node_in_dim, device, hidden_dim):
        super(MCAM, self).__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.sigma_d = nn.Parameter(2 * torch.rand(1, device=self.device, requires_grad=True) - 1)
        self.sigma_r = nn.Parameter(2 * torch.rand(1, device=self.device, requires_grad=True) - 1)

        # static Channel
        self.static_fc = nn.Sequential(nn.Linear(self.node_in_dim, self.hidden_dim),
                                       nn.ReLU())
        self.static_attention = nn.Sequential(nn.Linear(1, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, 1),
                                              nn.Softmax())
        self.static_gcn = GCNConv(self.hidden_dim, self.hidden_dim, cached=True, node_dim=1, normalize=False)

        # dynamic Channel
        self.dynamic_node_lstm = nn.LSTM(self.in_dim - 1, self.hidden_dim, num_layers=2, batch_first=True)
        self.dynamic_station_lstm = nn.LSTM(self.in_dim, self.hidden_dim, num_layers=2, batch_first=True)
        self.dynamic_fc = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                        nn.ReLU())
        self.dynamic_attention = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_dim, 1),
                                               nn.Softmax())
        self.dynamic_gcn = GCNConv(self.hidden_dim, self.hidden_dim, cached=True, node_dim=1, normalize=False)

        self.pred_fc = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.seq_len))

    def cal_pearson_corr(self, x, y):
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        mean_y = torch.mean(y, dim=-1, keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1))
        r = r_num / r_den
        return r

    def forward(self, local_node, local_features, station_nodes, station_features, station_dist):
        batch_size = local_node.size(0)
        station_num = station_dist.size(1)
        station_dist = station_dist[:, :, 0]

        # generate edge_index
        seq = torch.arange(station_num+1, device=self.device).view(-1, 1)
        base = torch.zeros(station_num+1, 1).to(self.device)
        edge_index = torch.concatenate((base, seq), dim=-1).transpose(0, 1).to(torch.int64)

        # cal graph weight
        edge_weight = []
        station_dy_outs = []
        for i in range(station_num):
            dist = station_dist[:, i]
            dist = torch.exp(torch.square(dist / self.sigma_d) / -2)
            pearson = self.cal_pearson_corr(local_node.squeeze(1), station_nodes[:, i])
            pearson = torch.exp(torch.square(pearson / self.sigma_r) / -2)
            edge_weight.append(dist * pearson)

            station_o, (station_dy_out, station_c) = self.dynamic_station_lstm(station_features[:, :, i])
            station_dy_outs.append(station_dy_out.transpose(0, 1))

        edge_weight = torch.stack(edge_weight, dim=1)
        edge_weight = edge_weight / torch.sum(edge_weight, dim=1, keepdim=True)
        edge_weight = edge_weight.unsqueeze(-1)  # [batch, station_num, 1]
        edge_weight = self.static_attention(edge_weight)
        static_edge_weight = edge_weight.squeeze(-1)
        static_edge_index = edge_index[..., 1:]

        # static channel
        nodes = torch.cat((local_node, station_nodes), dim=1)
        static_outs = self.static_fc(nodes)

        # dynamic channel
        node_o, (node_dy_out, node_c) = self.dynamic_node_lstm(local_features.squeeze(2))
        node_dy_out = node_dy_out.permute(1, 0, 2).reshape(batch_size, -1, 2 * self.hidden_dim)
        station_dy_outs = torch.stack(station_dy_outs, dim=1).reshape(batch_size, -1, 2 * self.hidden_dim)
        dynamic_ins = torch.cat((node_dy_out, station_dy_outs), dim=1)
        dynamic_nodes = self.dynamic_fc(dynamic_ins)
        dynamic_edge_weight = self.dynamic_attention(dynamic_ins)
        dynamic_edge_weight = dynamic_edge_weight.squeeze(-1)

        static_outs = self.static_gcn(static_outs, static_edge_index, static_edge_weight)
        dynamic_outs = self.dynamic_gcn(dynamic_nodes, edge_index, dynamic_edge_weight)
        # pred
        outs = torch.concatenate((static_outs[:, 0], dynamic_outs[:, 0]), dim=-1)
        outs = self.pred_fc(outs)
        outs = outs.view(-1, self.seq_len, 1)
        return outs

