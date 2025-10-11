from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers and optional normalization."""

    def __init__(self, channels: int, kernel_size: int = 3, use_norm: bool = True) -> None:
        super().__init__()
        p = kernel_size // 2
        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=p, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=p, bias=True)
        if use_norm:
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        if self.use_norm:
            out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_norm:
            out = self.norm2(out)
        out = out + residual
        return self.act(out)


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        p = kernel_size // 2

        # Channel configuration (hardcoded attributes for easy experimentation)
        # Notes/constraints:
        # - ResidualBlock(C) assumes in_channels == out_channels == C for identity skips.
        # - The 1×1 conv is the only width-changing op between stages; its out_channels
        #   must match the channel count expected by subsequent blocks.
        # - Global pooling in ResNetFeaturesExtractor removes the dependence on H×W for
        #   the projection layer, but activations earlier in the network still preserve
        #   spatial resolution.
        # - Keep kernel_size odd so padding=k//2 preserves spatial size.
        self.stem_channels: int = 32   # width after stem conv (was 32)
        self.stage1_channels: int = 32 # width inside stage 1 residual blocks (was 32)
        self.stage2_channels: int = 64 # width inside stage 2 residual blocks (was 64)
        self.num_blocks_stage1: int = 2 # number of BasicBlock2Branch in stage 1 (was 2)
        self.num_blocks_stage2: int = 7 # number of BasicBlock2Branch in stage 2 (was 7)

        # Stem
        self.use_norm = True

        stem_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, self.stem_channels, kernel_size, padding=p, bias=True)
        ]
        if self.use_norm:
            stem_layers.append(nn.GroupNorm(1, self.stem_channels))
        stem_layers.append(nn.ReLU(inplace=False))
        self.stem = nn.Sequential(*stem_layers)

        # Stages
        blocks: list[nn.Module] = []
        # Stage 1: residual blocks at stage1_channels
        # Constraint: stem_channels must equal stage1_channels for residual add without a projection.
        # If they differ, insert a 1×1 projection from stem_channels→stage1_channels before blocks.
        if self.stem_channels != self.stage1_channels:
            proj = [nn.Conv2d(self.stem_channels, self.stage1_channels, kernel_size=1, bias=True)]
            if self.use_norm:
                proj.append(nn.GroupNorm(1, self.stage1_channels))
            proj.append(nn.ReLU(inplace=False))
            blocks.append(nn.Sequential(*proj))
        blocks += [ResidualBlock(self.stage1_channels, kernel_size, use_norm=self.use_norm) for _ in range(self.num_blocks_stage1)]

        # Transition: channel jump via 1×1 conv (preserve H×W)
        if self.stage1_channels != self.stage2_channels:
            trans_layers: list[nn.Module] = [
                nn.Conv2d(self.stage1_channels, self.stage2_channels, kernel_size=1, bias=True)
            ]
            if self.use_norm:
                trans_layers.append(nn.GroupNorm(1, self.stage2_channels))
            trans_layers.append(nn.ReLU(inplace=False))
            blocks.append(nn.Sequential(*trans_layers))

        # Stage 2: residual blocks at stage2_channels
        blocks += [ResidualBlock(self.stage2_channels, kernel_size, use_norm=self.use_norm) for _ in range(self.num_blocks_stage2)]

        self.blocks = nn.Sequential(*blocks)

        # Expose backbone output channels for downstream projection sizing
        self.out_channels: int = self.stage2_channels

        # Init weights: Kaiming for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.blocks(x)


class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """ResNet-style extractor with residual blocks and global pooling.

    The projection head maps pooled backbone features (`out_channels`) to
    `features_dim` (default 256) for SB3 policy heads.
    """

    def __init__(self, observation_space: Box, features_dim: int = 256, kernel_size: int = 3) -> None:
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected image observation (C,H,W)"
        c_in, H, W = int(observation_space.shape[0]), int(observation_space.shape[1]), int(observation_space.shape[2])

        self.backbone = ResNetBackbone(c_in, kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(self.backbone.out_channels)
        self.proj = nn.Linear(self.backbone.out_channels, features_dim)
        # Avoid in-place here as well for safe autograd on MPS/CUDA/CPU
        self.act = nn.ReLU(inplace=False)
        self._features_dim = int(features_dim)

        # Init proj
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_in", nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.backbone(observations)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        return self.act(self.proj(x))
