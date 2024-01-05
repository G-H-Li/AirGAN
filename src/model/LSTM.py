import torch
from torch import nn

from src.model.Cells import LSTMCell


class LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.lstm_cell = LSTMCell(self.hid_dim, self.hid_dim)

    def forward(self, pm25_hist, feature, time_feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0
        xn = pm25_hist.reshape(pm25_hist.shape[0], pm25_hist.shape[2], -1)
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            pred = hn.view(self.batch_size, self.city_num, self.hid_dim)
            pred = self.fc_out(pred)
            pm25_pred.append(pred)
            xn = torch.cat((xn[:, :, 1:], pred), dim=-1)
        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred
