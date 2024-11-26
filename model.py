import torch

from torch import nn
from torch_geometric.nn import (
    DynamicEdgeConv,
    GATConv,
    global_max_pool,
    global_mean_pool,
    knn_graph,
)


class FiLM(nn.Module):
    def __init__(self, in_features: int, conditioning_features: int):
        super().__init__()
        self.gamma = nn.Linear(conditioning_features, in_features)
        self.beta = nn.Linear(conditioning_features, in_features)

    def forward(self, x: torch.Tensor, conditioning_input: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(conditioning_input)
        beta = self.beta(conditioning_input)
        return gamma * x + beta


class RND(nn.Module):
    def __init__(self, in_features: int, conditioning_features: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, in_features)
        )
        self.prior = nn.Sequential(
            nn.Linear(conditioning_features, 128), nn.ReLU(), nn.Linear(128, in_features)
        )

    def forward(self, x: torch.Tensor, conditioning_input: torch.Tensor) -> torch.Tensor:
        prior_output = self.prior(conditioning_input)
        predictor_output = self.predictor(x)
        bonus = torch.norm(predictor_output - prior_output, p=2, dim=-1)
        return bonus, prior_output, predictor_output


class RegDGCNN(nn.Module):
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

        self.conv1 = DynamicEdgeConv(self.mlp([2 * 3, 128, 128, 128]), self.k, aggr="max")
        self.film1 = FiLM(128, 128)
        self.attn1 = GATConv(128, 32, heads=4, concat=True)

        self.conv2 = DynamicEdgeConv(self.mlp([2 * 128, 256, 256, 256]), self.k, aggr="max")
        self.film2 = FiLM(256, 256)
        self.attn2 = GATConv(256, 64, heads=4, concat=True)

        self.conv3 = DynamicEdgeConv(self.mlp([2 * 256, 512, 512, 512]), self.k, aggr="max")
        self.film3 = FiLM(512, 512)
        self.attn3 = GATConv(512, 128, heads=4, concat=True)

        self.conv4 = DynamicEdgeConv(self.mlp([2 * 512, 1024, 1024, 1024]), self.k, aggr="max")
        self.film4 = FiLM(1024, 1024)
        self.attn4 = GATConv(1024, 256, heads=4, concat=True)

        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool

        self.embedding_size = (128 + 256 + 512 + 1024) * 2
        self.lin1 = nn.Linear(self.embedding_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(p=0.3)
        self.lin3 = nn.Linear(1024, 1)

        self.rnd = RND(in_features=1024, conditioning_features=1024)

    def mlp(self, channels: list[int]) -> nn.Sequential:
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def compute_edge_index(self, x: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return knn_graph(x, k=k, batch=batch)

    def forward(self, data: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x, batch = data.pos, data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Layer 1
        x1 = self.conv1(x, batch)
        x1 = self.film1(x1, x1.mean(dim=0, keepdim=True).repeat(x1.size(0), 1))
        edge_index1 = self.compute_edge_index(x1, batch, self.k)
        x1 = self.attn1(x1, edge_index1)
        x1_pool_mean = self.pool_mean(x1, batch)
        x1_pool_max = self.pool_max(x1, batch)

        # Layer 2
        x2 = self.conv2(x1, batch)
        x2 = self.film2(x2, x2.mean(dim=0, keepdim=True).repeat(x2.size(0), 1))
        edge_index2 = self.compute_edge_index(x2, batch, self.k)
        x2 = self.attn2(x2, edge_index2)
        x2_pool_mean = self.pool_mean(x2, batch)
        x2_pool_max = self.pool_max(x2, batch)

        # Layer 3
        x3 = self.conv3(x2, batch)
        x3 = self.film3(x3, x3.mean(dim=0, keepdim=True).repeat(x3.size(0), 1))
        edge_index3 = self.compute_edge_index(x3, batch, self.k)
        x3 = self.attn3(x3, edge_index3)
        x3_pool_mean = self.pool_mean(x3, batch)
        x3_pool_max = self.pool_max(x3, batch)

        # Layer 4
        x4 = self.conv4(x3, batch)
        x4 = self.film4(x4, x4.mean(dim=0, keepdim=True).repeat(x4.size(0), 1))
        edge_index4 = self.compute_edge_index(x4, batch, self.k)
        x4 = self.attn4(x4, edge_index4)
        x4_pool_mean = self.pool_mean(x4, batch)
        x4_pool_max = self.pool_max(x4, batch)

        rnd_bonus, prior_output, predictor_output = self.rnd(x4, x4.view(x4.size(0), -1))

        embedding = torch.cat(
            [
                x1_pool_mean,
                x1_pool_max,
                x2_pool_mean,
                x2_pool_max,
                x3_pool_mean,
                x3_pool_max,
                x4_pool_mean,
                x4_pool_max,
            ],
            dim=1,
        )

        x = torch.relu(self.bn1(self.lin1(embedding)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.lin2(x)))
        x = self.dropout2(x)
        output = self.lin3(x)

        output = torch.sigmoid(output) * 1.5

        if return_embedding:
            return output, embedding, rnd_bonus

        return output
