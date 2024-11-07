import torch

from torch import nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool


class RegDGCNN(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k

        self.conv1 = DynamicEdgeConv(self.mlp([12, 64, 64]), self.k, aggr="max")
        self.conv2 = DynamicEdgeConv(self.mlp([64 * 2, 128, 128]), self.k, aggr="max")
        self.conv3 = DynamicEdgeConv(self.mlp([128 * 2, 256, 256]), self.k, aggr="max")
        self.conv4 = DynamicEdgeConv(self.mlp([256 * 2, 512, 512]), self.k, aggr="max")

        self.pool = global_mean_pool

        self.lin1 = nn.Linear(64 + 128 + 256 + 512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.lin3 = nn.Linear(256, 1)

    def mlp(self, channels: list[int]) -> nn.Sequential:
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pos, batch, normals = data.pos, data.batch, data.normals

        x = torch.cat([pos, normals], dim=1)

        x1 = self.conv1(x, batch)
        x1_pool = self.pool(x1, batch)

        x2 = self.conv2(x1, batch)
        x2_pool = self.pool(x2, batch)

        x3 = self.conv3(x2, batch)
        x3_pool = self.pool(x3, batch)

        x4 = self.conv4(x3, batch)
        x4_pool = self.pool(x4, batch)

        x = torch.cat([x1_pool, x2_pool, x3_pool, x4_pool], dim=1)

        x = nn.functional.relu(self.bn1(self.lin1(x)))
        x = self.dropout1(x)
        x = nn.functional.relu(self.bn2(self.lin2(x)))
        x = self.dropout2(x)
        return self.lin3(x)
