import torch
from torch import nn


class PointNet(nn.Module):
    def __init__(self, feature_dim, dim=2, factor=1, act='leaky', mini=False):
        super().__init__()
        self.factor = factor
        self.dim = dim
        self.feature_dim = feature_dim
        c64 = 64 // self.factor
        self.mini = mini

        print(c64, self.factor)
        self.conv1 = torch.nn.Conv1d(self.dim, c64, 1)
        self.conv2 = torch.nn.Conv1d(c64, c64 * 2, 1)
        self.conv3 = torch.nn.Conv1d(c64 * 2, self.feature_dim, 1)
        self.lin = nn.Linear(self.feature_dim, self.feature_dim)

        self.bn1 = torch.nn.BatchNorm1d(c64)
        self.bn2 = torch.nn.BatchNorm1d(c64 * 2)
        self.bn3 = torch.nn.BatchNorm1d(self.feature_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.feature_dim)

        if act is 'leaky':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        if self.mini:
            x = x.view(-1, self.feature_dim, 1)
            return self.act(x)
        else:
            x = x.view(-1, self.feature_dim)
            x = self.act(self.bn4(self.lin(x).unsqueeze(-1)))
            return x
