"""Fixed 64→96→48 MLP from the formal reproduction notebook."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 64,
        hidden_dims: Sequence[int] = (96, 48),
        dropout: float = 0.05,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(width, hidden),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ]
            )
            width = hidden
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


ChampionMLP = MLP
