import torch
from pytorch_tcn import TCN
from torch import nn
from torch_geometric.nn import GCNConv


class ASTGC(nn.Module):
    def __init__(self, seq_len, in_dim, node_in_dim, hidden_dim, device):
        super(ASTGC, self).__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.target_fc = nn.Sequential(nn.Linear(self.node_in_dim, self.hidden_dim),
                                       nn.ReLU())
        self.target_tcn = nn.Sequential(TCN(self.in_dim - 1, [self.hidden_dim]),
                                        nn.ReLU())

        self.target_fusion_fc = nn.Sequential(nn.Linear(2 * self.seq_len * self.hidden_dim, self.hidden_dim),
                                              nn.ReLU())

        self.station_gcn = GCNConv(self.node_in_dim, self.hidden_dim, cached=True, node_dim=1, normalize=False)
        self.station_tcn = nn.Sequential(TCN(self.in_dim, [self.hidden_dim]),
                                        nn.ReLU())
        self.station_fusion_fc = nn.Sequential(nn.Linear(2 * self.seq_len * self.hidden_dim, self.hidden_dim),
                                               nn.ReLU())

        self.fusion_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, 1),
                                              nn.Softmax())

        self.fusion_gcn = GCNConv(self.hidden_dim, self.seq_len, cached=True, node_dim=1, normalize=False)

    def _get_edge_index(self, num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                edges.append([i, j])
        return torch.tensor(edges).t().contiguous()

    def forward(self, local_node, local_features, station_nodes, station_features, station_dist):
        batch_size = local_node.size(0)
        station_num = station_dist.size(1)
        station_dist = torch.triu(station_dist[:, :, :, 0])
        station_dist = station_dist[station_dist != 0].reshape(batch_size, -1)

        edge_index = self._get_edge_index(station_num).to(self.device)

        target_fc_out = self.target_fc(local_node)
        target_fc_out = target_fc_out.repeat(1, self.seq_len, 1)
        target_tcn_out = self.target_tcn(local_features.squeeze(2).transpose(1, 2))
        target_tcn_out = target_tcn_out.permute(0, 2, 1)

        target_out = torch.cat((target_fc_out, target_tcn_out), dim=-1).reshape(batch_size, 1, -1)
        target_out = self.target_fusion_fc(target_out)

        station_time_out = []
        for i in range(station_num):
            station_tcn_out = self.station_tcn(station_features[:, :, i].transpose(1, 2))
            station_time_out.append(station_tcn_out.transpose(1, 2))
        station_tcn_out = torch.stack(station_time_out, dim=1)
        station_out = self.station_gcn(station_nodes, edge_index, station_dist)
        station_out = station_out.unsqueeze(2).repeat(1, 1, self.seq_len, 1)
        station_out = torch.cat((station_out, station_tcn_out), dim=-1).reshape(batch_size, station_num, -1)
        station_out = self.station_fusion_fc(station_out)

        seq = torch.arange(station_num + 1, device=self.device).view(-1, 1)
        base = torch.zeros(station_num + 1, 1).to(self.device)
        edge_index = torch.concatenate((base, seq), dim=-1).transpose(0, 1).to(torch.int64)

        fusion_out = torch.cat((target_out, station_out), dim=1)
        fusion_attention = self.fusion_attention(station_out).reshape(batch_size, -1)
        fusion_out = self.fusion_gcn(fusion_out, edge_index[:, 1:], fusion_attention)

        pred_out = fusion_out[:, 0].view(-1, self.seq_len, 1)
        return pred_out
