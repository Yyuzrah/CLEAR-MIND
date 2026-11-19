"""V.alpha EdgeConv point-cloud GNN with raw 64D spectrum late fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(features: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(features.transpose(2, 1), features)
    squared = torch.sum(features**2, dim=1, keepdim=True)
    pairwise_distance = -squared - inner - squared.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1).indices


def graph_feature(
    features: torch.Tensor,
    k: int = 16,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, channels, point_count = features.shape
    indices = knn(features, k=k) if indices is None else indices
    actual_k = indices.size(-1)
    base = torch.arange(batch_size, device=features.device).view(-1, 1, 1) * point_count
    flat_indices = (indices + base).reshape(-1)
    transposed = features.transpose(2, 1).contiguous()
    neighbors = transposed.reshape(batch_size * point_count, channels)[flat_indices]
    neighbors = neighbors.view(batch_size, point_count, actual_k, channels)
    centers = transposed.view(batch_size, point_count, 1, channels).expand(-1, -1, actual_k, -1)
    return torch.cat((neighbors - centers, centers), dim=-1).permute(0, 3, 1, 2).contiguous()


class GNNAlpha(nn.Module):
    def __init__(self, num_classes: int, k: int = 16, spectral_dim: int = 64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(2048 + 128, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(
        self,
        point_cloud: torch.Tensor,
        spectrum: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = point_cloud.size(0)
        # The formal baseline cache stores three auxiliary structural channels,
        # while V.alpha consumes only normalized xyz.
        xyz = point_cloud[:, :3]
        x = self.conv1(graph_feature(xyz, k=self.k)).max(dim=-1).values
        x = self.conv2(graph_feature(x, k=self.k)).max(dim=-1).values
        x = self.conv3(graph_feature(x, k=self.k)).max(dim=-1).values
        x = self.conv4(graph_feature(x, k=self.k)).max(dim=-1).values
        x = self.conv5(x)

        maximum = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        average = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        point_embedding = torch.cat((maximum, average), dim=1)
        spectral_embedding = self.spectral_proj(spectrum)
        fused = torch.cat((point_embedding, spectral_embedding), dim=-1)
        fused = F.leaky_relu(
            self.bn6(self.linear1(fused)),
            negative_slope=0.2,
        )
        fused = self.dp1(fused)
        fused = F.leaky_relu(
            self.bn7(self.linear2(fused)),
            negative_slope=0.2,
        )
        fused = self.dp2(fused)
        return self.linear3(fused)


TropicalMorphoGNN = GNNAlpha
