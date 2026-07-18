"""Feature extractor: stride-1 residual trunk with a small readout.

Keeps the one thing the legacy stack got right (same-padded, no pooling/stride
so single-cell geometry survives) and drops the giant flatten->512 linear that
held ~68% of the old model's parameters and memorized layouts. The trunk does
the work; the readout is a 1x1 channel reduction and a modest projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class GridTrunk(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: Box,
        features_dim: int = 256,
        channels: int = 64,
        n_blocks: int = 4,
        reduce_channels: int = 16,
    ) -> None:
        super().__init__(observation_space, features_dim)
        in_channels, height, width = observation_space.shape
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(channels) for _ in range(n_blocks)])
        # LayerNorm keeps the feature scale bounded: without it the residual
        # trunk's features grew to O(1000) at 16x16 and saturated downstream
        # Tanh heads into a constant value function (E3 v1 failure, 2026-07-18).
        self.readout = nn.Sequential(
            nn.Conv2d(channels, reduce_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(reduce_channels * height * width, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.readout(self.blocks(self.stem(observations)))
