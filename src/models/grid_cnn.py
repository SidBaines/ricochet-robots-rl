"""Same-padded CNN feature extractor for small structured plane observations.

Designed for the ``image`` observation mode, where the board is encoded as
binary planes (walls, goal, target mask, robots) at native grid resolution,
e.g. ``(8, 8, 8)`` for an 8x8 board with two robots.

Unlike SB3's NatureCNN (which needs >=36x36 inputs and destroys single-cell
detail with an 8x8 stride-4 first convolution) or the repository's older
``SmallCNN`` (whose global average pooling discards exact positions), this
extractor keeps spatial resolution through every convolution and flattens at
the end, so precise robot/goal/wall geometry stays recoverable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridPlanesCnn(BaseFeaturesExtractor):
    """Stride-1, same-padded CNN followed by flatten + linear projection."""

    def __init__(
        self,
        observation_space: Box,
        features_dim: int = 256,
        conv_channels: int = 64,
        n_conv_layers: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 3:
            raise ValueError(f"Expected (C, H, W) observations, got {observation_space.shape}")
        if n_conv_layers < 1:
            raise ValueError("n_conv_layers must be at least 1")
        in_channels, height, width = observation_space.shape

        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(n_conv_layers):
            layers.append(nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = conv_channels
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * height * width, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(observations))
