from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.model.Base_GAN import Base_GAN
from src.utils.config import Config
from src.utils.utils import toggle_grad

"""
N: city num
F: feature num
H: history sequence num
P: predict sequence num
B: batch size
"""


def cal_loc_correlation(city_loc, edge_index, edge_attr) -> Tuple[list, list]:
    """
    calculate correlation between city based dfs algorithm
    :param edge_index: EDGE_NUM x 20
    :param city_loc: CITY_NUM x 2
    :param edge_attr: EDGE_NUM x 2 (distance, direction)
    :return: index correlation
    """
    city_cor_idx_dict: Dict[int, any] = dict()
    for i in range(city_loc.shape[0]):
        idx = np.where(edge_index == i)
        edge_attr_cor_idx = idx[1][:idx[1].size//2]
        edge_index_cor = edge_index[1, edge_attr_cor_idx]
        edge_attr_cor = edge_attr[edge_attr_cor_idx, 0]
        min_cor_idx = np.argsort(edge_attr_cor)
        loc_cor = edge_index_cor[min_cor_idx]
        city_cor_idx_dict[i] = loc_cor
    city2st_cor_list = list()
    dfs(city_cor_idx_dict, 0, city2st_cor_list)
    st2city_cor_list = [index for index, value in sorted(enumerate(city2st_cor_list), key=lambda x: x[1])]
    return city2st_cor_list, st2city_cor_list


def dfs(graph, node, visited: list):
    if node not in visited:
        visited.append(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)


def converse_st_graph(index_cor, data):
    """
    create spatial-temporal graph
    :param index_cor: loc_correlation index: list, length:city_num
    :param data: pm25_hist data: batch_size, len, city_num, pm25
    :return: pm25_hist(batch_size, city_num, len, pm25),
    """
    data = data[:, :, index_cor, :]
    data = data.transpose(2, 1)
    return data


class Generator(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, hidden_dim, conditional_dim,
                 feature_channels, batch_size):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.hist_len = hist_len
        self.conditional_dim = conditional_dim
        self.feature_channels = feature_channels
        self.batch_size = batch_size

        preprocess1 = nn.Sequential(
            nn.Conv2d(self.feature_channels + 2, 4 * 4 * 4 * self.hidden_dim, (33, 3)),
            nn.ReLU(True),
            nn.Conv2d(4 * 4 * 4 * self.hidden_dim, 4 * 4 * 4 * self.hidden_dim, (33, 3)),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * 4 * 4 * self.hidden_dim, 2 * self.hidden_dim, (16, 3)),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, (16, 3)),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.hidden_dim, 1, (35, 3), stride=(1, 3))

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess1 = preprocess1
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, condition):
        # noise: batch_size, city_num, hist_len, 1
        # condition: batch_size, city_num, hist_len, feature_num+pm25
        in_data = torch.cat((noise, condition), dim=-1)
        in_data = in_data.reshape(self.batch_size, -1, in_data.shape[1], in_data.shape[2])
        output = self.preprocess(in_data)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, feature_channels, hidden_dim, batch_size, conditional_dim):
        super(Discriminator, self).__init__()
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.conditional_dim = conditional_dim

        self.model = nn.Sequential(
            nn.Conv2d(2*(feature_channels+1), self.hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * self.hidden_dim, 4 * self.hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.output = nn.Linear(4 * 4 * 4 * self.hidden_dim, 1)

    def forward(self, st_graph, condition):
        # st_graph: batch_size, city_num, pred_len, pm25
        # condition: batch_size, city_num, hist_len, feature_num+pm25
        st_graph = st_graph.repeat((-1, condition.shape[-1]))
        in_data = torch.cat((st_graph, condition), dim=-1)
        in_data = in_data.reshape(self.batch_size, -1, in_data.shape[1], in_data.shape[2])
        output = self.model(in_data)
        output = output.view(-1, 4 * 4 * 4 * self.hidden_dim)
        out = self.output(output)
        return out


class CW_GAN(Base_GAN):
    def __init__(self, config: Config, device: torch.device, edge_index, edge_attr, city_loc, in_dim: int,
                 conditional_dim: int, feature_channels: int):
        super(CW_GAN, self).__init__(config, device)

        self.city2st, self.st2city = cal_loc_correlation(city_loc, edge_index, edge_attr)

        self.generator = Generator(self.config.hist_len, self.config.pred_len, in_dim,
                                   self.config.hidden_dim, conditional_dim,
                                   feature_channels, self.config.batch_size).to(device)
        self.discriminator = Discriminator(feature_channels, self.config.hidden_dim,
                                           self.config.batch_size, conditional_dim).to(device)

        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.criterion = nn.MSELoss()

    def batch_train(self, pm25_hist, feature_hist, pm25_labels):
        """
        trains the discriminator and generator per batch
        :param pm25_hist: batch_size, city_num, history_len, pm25
        :param pm25_labels: batch_size, city_num, pred_len, pm25
        :param feature_hist: batch_size, city_num, history_len, feature_num
        :return: loss of discriminator and generator
        """
        # converse data
        feature_hist = converse_st_graph(self.city2st, feature_hist)
        pm25_hist = converse_st_graph(self.city2st, pm25_hist)
        pm25_labels = converse_st_graph(self.city2st, pm25_labels)
        # train
        loss_d = 0
        real_labels = torch.ones(self.config.batch_size).to(self.device)
        fake_labels = torch.zeros(self.config.batch_size).to(self.device)
        conditionals = torch.cat((pm25_hist, feature_hist), dim=-1).to(self.device)
        # discriminator step
        for iter_d in range(self.config.critic_iters):
            toggle_grad(self.discriminator, True)
            self.discriminator.train()
            self.optimizerD.zero_grad()

            noise = torch.randn(self.config.batch_size, self.config.city_num, self.config.hist_len, 1).to(self.device)
            fake_data = self.generator(noise, conditionals)

            res_real = self.discriminator(pm25_labels, conditionals)
            loss_real = self.criterion(res_real, real_labels)

            res_fake = self.discriminator(fake_data, conditionals)
            loss_fake = self.criterion(res_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            self.optimizerD.step()

        # generator step
        toggle_grad(self.generator, True)
        self.generator.train()
        self.optimizerG.zero_grad()

        noise = torch.randn(self.config.batch_size, self.config.city_num, self.config.hist_len, 1).to(self.device)
        fake_data = self.generator(noise, conditionals)
        output_fake = self.discriminator(fake_data, conditionals)
        loss_g = self.criterion(output_fake, real_labels)

        loss_g.backward()
        self.optimizerG.step()
        return loss_d, loss_g

    def batch_valid(self, pm25_hist, feature_hist, pm25_labels):
        self.generator.eval()
        # converse data
        feature_hist = converse_st_graph(self.city2st, feature_hist)
        pm25_hist = converse_st_graph(self.city2st, pm25_hist)
        pm25_labels = converse_st_graph(self.city2st, pm25_labels)
        # valid
        conditionals = torch.cat((pm25_hist, feature_hist), dim=-1).to(self.device)
        noise = torch.randn(self.config.batch_size, self.config.city_num, self.config.hist_len, 1).to(self.device)
        fake_data = self.generator(noise, conditionals)
        valid_loss = self.criterion(fake_data, pm25_labels)
        return valid_loss

    def batch_test(self, pm25_hist, feature_hist, pm25_labels):
        self.generator.eval()
        # converse data
        feature_hist = converse_st_graph(self.city2st, feature_hist)
        pm25_hist = converse_st_graph(self.city2st, pm25_hist)
        pm25_labels = converse_st_graph(self.city2st, pm25_labels)
        # test
        conditionals = torch.cat((pm25_hist, feature_hist), dim=-1).to(self.device)
        noise = torch.randn(self.config.batch_size, self.config.city_num, self.config.hist_len, 1).to(self.device)
        fake_data = self.generator(noise, conditionals)
        test_loss = self.criterion(fake_data, pm25_labels)

        fake_data = converse_st_graph(self.st2city, fake_data)
        pm25_labels = converse_st_graph(self.st2city, pm25_labels)

        return test_loss, fake_data, pm25_labels
