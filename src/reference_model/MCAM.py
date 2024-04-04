from torch import nn

class MCAM(nn.Module):
    def __init__(self):
        super(MCAM, self).__init__()
        self.x_embed = nn.Linear(64, 64)