import numpy as np
import os

import torch
import torch.utils.data as Data

# 获取当前脚本所在的目录
current_directory = os.path.dirname(__file__)
# 构造相对路径
relative_path = '..\\..\\data'  # 以当前脚本所在目录为基准的相对路径


class TrainDataset(Data.Dataset):
    def __init__(self):
        self.x = np.load(os.path.join(current_directory, relative_path, 'train_x.npy'), allow_pickle=True)
        self.u = np.load(os.path.join(current_directory, relative_path, 'train_u.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(current_directory, relative_path, 'train_y.npy'), allow_pickle=True)
        self.edge_w = np.load(os.path.join(current_directory, relative_path, 'edge_w.npy'), allow_pickle=True)
        self.edge_index = np.load(os.path.join(current_directory, relative_path, 'edge_index.npy'), allow_pickle=True)
        self.loc = np.load(os.path.join(current_directory, relative_path, 'loc_filled.npy'), allow_pickle=True)
        self.loc = self.loc.astype(np.float64)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        x = x.transpose(0, 1)
        y = torch.FloatTensor(self.y[index])
        y = y.transpose(0, 1)
        u = torch.tensor(self.u[index])
        edge_index = torch.tensor(self.edge_index)
        # edge_index = edge_index.expand((x.size[0],edge_index.size[0],edge_index.size[1]))
        edge_w = torch.FloatTensor(self.edge_w)
        # edge_w = edge_w.expand((x.size[0],edge_w.size[0]))
        loc = torch.FloatTensor(self.loc)

        return [x, u, y, edge_index, edge_w, loc]

    def __len__(self):
        return self.x.shape[0]


class ValDataset(Data.Dataset):
    def __init__(self):
        self.x = np.load(os.path.join(current_directory, relative_path, 'val_x.npy'), allow_pickle=True)
        self.u = np.load(os.path.join(current_directory, relative_path, 'val_u.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(current_directory, relative_path, 'val_y.npy'), allow_pickle=True)
        self.edge_w = np.load(os.path.join(current_directory, relative_path, 'edge_w.npy'), allow_pickle=True)
        self.edge_index = np.load(os.path.join(current_directory, relative_path, 'edge_index.npy'), allow_pickle=True)
        self.loc = np.load(os.path.join(current_directory, relative_path, 'loc_filled.npy'), allow_pickle=True)
        self.loc = self.loc.astype(np.float64)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        x = x.transpose(0, 1)
        y = torch.FloatTensor(self.y[index])
        y = y.transpose(0, 1)
        u = torch.tensor(self.u[index])
        edge_index = torch.tensor(self.edge_index)
        # edge_index = edge_index.expand((x.size[0],edge_index.size[0],edge_index.size[1]))
        edge_w = torch.FloatTensor(self.edge_w)
        # edge_w = edge_w.expand((x.size[0],edge_w.size[0]))
        loc = torch.FloatTensor(self.loc)

        return [x, u, y, edge_index, edge_w, loc]

    def __len__(self):
        return self.x.shape[0]


class TestDataset(Data.Dataset):
    def __init__(self):
        self.x = np.load(os.path.join(current_directory, relative_path, 'test_x.npy'), allow_pickle=True)
        self.u = np.load(os.path.join(current_directory, relative_path, 'test_u.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(current_directory, relative_path, 'test_y.npy'), allow_pickle=True)
        self.edge_w = np.load(os.path.join(current_directory, relative_path, 'edge_w.npy'), allow_pickle=True)
        self.edge_index = np.load(os.path.join(current_directory, relative_path, 'edge_index.npy'), allow_pickle=True)
        self.loc = np.load(os.path.join(current_directory, relative_path, 'loc_filled.npy'), allow_pickle=True)
        self.loc = self.loc.astype(np.float64)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        x = x.transpose(0, 1)
        y = torch.FloatTensor(self.y[index])
        y = y.transpose(0, 1)
        u = torch.tensor(self.u[index])
        edge_index = torch.tensor(self.edge_index)
        # edge_index = edge_index.expand((x.size[0],edge_index.size[0],edge_index.size[1]))
        edge_w = torch.FloatTensor(self.edge_w)
        # edge_w = edge_w.expand((x.size[0],edge_w.size[0]))
        loc = torch.FloatTensor(self.loc)

        return [x, u, y, edge_index, edge_w, loc]

    def __len__(self):
        return self.x.shape[0]
