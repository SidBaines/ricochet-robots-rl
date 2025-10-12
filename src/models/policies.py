from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class SmallCNN(BaseFeaturesExtractor):
    """A small CNN extractor that supports small inputs (>=4x4) safely.

    It uses 3x3 kernels with padding=1 to preserve spatial size, followed by global average pooling.
    """

    def __init__(self, observation_space: Box, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected image observation (C,H,W)"
        in_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, C, H, W)
        x = self.cnn(observations)
        # Global average pooling over H,W → (B, C)
        x = x.mean(dim=(-2, -1))
        x = self.proj(x)
        return x


