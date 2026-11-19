"""Native child-sum TreeLSTM architecture 3 from the result notebook."""

from __future__ import annotations

import torch
import torch.nn as nn

from em_connectome.adapters.treelstm import TreeBatch


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=True)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=True)
        self.U_f = nn.Linear(h_size, h_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        h_children_sum: torch.Tensor,
        c_children_weighted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        iou = self.W_iou(x) + self.U_iou(h_children_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + c_children_weighted
        h = o * torch.tanh(c)
        return h, c

    def forget_contribution(
        self,
        parent_x: torch.Tensor,
        child_h: torch.Tensor,
        child_c: torch.Tensor,
    ) -> torch.Tensor:
        forget_gate = torch.sigmoid(self.W_f(parent_x) + self.U_f(child_h))
        return forget_gate * child_c


class NativeTreeLSTM(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.h_size = h_size
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)

    def forward(
        self,
        x: torch.Tensor,
        parent_indices: torch.Tensor,
        levels: list[torch.Tensor],
        total_nodes: int,
    ) -> torch.Tensor:
        h_sum = torch.zeros(total_nodes + 1, self.h_size, device=x.device)
        c_weighted = torch.zeros(total_nodes + 1, self.h_size, device=x.device)
        h_out = torch.zeros(total_nodes, self.h_size, device=x.device)

        for level_nodes in levels:
            x_level = x[level_nodes]
            h_sum_level = h_sum[level_nodes]
            c_weighted_level = c_weighted[level_nodes]
            h_level, c_level = self.cell(
                x_level,
                h_sum_level,
                c_weighted_level,
            )
            h_out[level_nodes] = h_level
            parent_level = parent_indices[level_nodes]
            h_sum.index_add_(0, parent_level, h_level)

            valid_parent = parent_level < total_nodes
            valid_parent_indices = parent_level[valid_parent]
            forget_c = self.cell.forget_contribution(
                x[valid_parent_indices],
                h_level[valid_parent],
                c_level[valid_parent],
            )
            c_weighted.index_add_(0, valid_parent_indices, forget_c)
        return h_out


class TreeLSTMArch3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        node_feat_dim: int = 5,
        hidden_dim: int = 128,
        spectral_dim: int = 64,
    ):
        super().__init__()
        self.treelstm = NativeTreeLSTM(
            x_size=node_feat_dim,
            h_size=hidden_dim,
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: TreeBatch) -> torch.Tensor:
        hidden = self.treelstm(
            batch.x,
            batch.parent_indices,
            batch.levels,
            batch.total_nodes,
        )
        root_hidden = hidden[batch.root_indices]
        graph_count = batch.root_indices.numel()
        hidden_sum = torch.zeros(graph_count, hidden.shape[1], device=hidden.device)
        hidden_sum.index_add_(0, batch.batch_indices, hidden)
        node_counts = torch.bincount(
            batch.batch_indices,
            minlength=graph_count,
        ).unsqueeze(1)
        hidden_mean = hidden_sum / node_counts.clamp_min(1)

        hidden_max = torch.full_like(hidden_sum, -torch.inf)
        expanded_indices = batch.batch_indices.unsqueeze(1).expand_as(hidden)
        hidden_max.scatter_reduce_(
            0,
            expanded_indices,
            hidden,
            reduce="amax",
            include_self=True,
        )
        graph_embedding = self.graph_proj(torch.cat([root_hidden, hidden_mean, hidden_max], dim=-1))
        spectral_embedding = self.spectral_proj(batch.spectral_sig)
        return self.classifier(torch.cat([graph_embedding, spectral_embedding], dim=-1))


TopologicalTreeLSTM = TreeLSTMArch3
