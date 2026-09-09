"""
mlp_model.py
=============
Stage: DAY 2, PART 2 — the point-based MLP itself.

Simple architecture as specified in the brief:
    input channels -> Linear -> activation -> Linear -> activation -> Linear -> 15 outputs

Deliberately small. Channel count and depth count are read from the data
at runtime (never hard-coded) by the caller in train_mlp.py.
"""

import torch
import torch.nn as nn


class PointMLP(nn.Module):
    def __init__(self, n_input_channels: int, n_output_depths: int = 15,
                 hidden_dim: int = 64):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_depths = n_output_depths
        self.net = nn.Sequential(
            nn.Linear(n_input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output_depths),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
