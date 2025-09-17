from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BasicBlock2Branch(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        p = kernel_size // 2
        self.trunk = nn.Conv2d(channels, channels, kernel_size, padding=p, bias=True)
        self.branch_a = nn.Conv2d(channels, channels, kernel_size, padding=p, bias=True)
        self.branch_b = nn.Conv2d(channels, channels, kernel_size, padding=p, bias=True)
        # Avoid in-place to prevent versioning issues when reusing trunk output
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.trunk(x)
        a = self.branch_a(self.act(t))
        b = self.branch_b(self.act(t))
        return t + a + b


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        p = kernel_size // 2

        # Channel configuration (hardcoded attributes for easy experimentation)
        # Notes/constraints:
        # - BasicBlock2Branch(C) requires in_channels == out_channels == C (residual add).
        # - The 1x1 conv is the only width-changing op between stages; its out_channels
        #   must match the channel count expected by the subsequent blocks.
        # - The final backbone out_channels determines the input size of the projection
        #   head in ResNetFeaturesExtractor: proj_in = out_channels * H * W.
        # - To reduce memory on 128x128 RGB inputs, lower these widths (e.g., 16→32
        #   instead of 32→64), or introduce downsampling (not done here to preserve H×W).
        # - Keep kernel_size odd so padding=k//2 preserves spatial size.
        self.stem_channels: int = 8   # width after stem conv (was 32)
        self.stage1_channels: int = 8 # width inside stage 1 residual blocks (was 32)
        self.stage2_channels: int = 16 # width inside stage 2 residual blocks (was 64)
        self.num_blocks_stage1: int = 2 # number of BasicBlock2Branch in stage 1 (was 2)
        self.num_blocks_stage2: int = 7 # number of BasicBlock2Branch in stage 2 (was 7)

        # Stem
        self.stem = nn.Conv2d(in_channels, self.stem_channels, kernel_size, padding=p, bias=True)

        # Stages
        blocks: list[nn.Module] = []
        # Stage 1: residual blocks at stage1_channels
        # Constraint: stem_channels must equal stage1_channels for residual add without a projection.
        # If they differ, insert a 1×1 projection from stem_channels→stage1_channels before blocks.
        if self.stem_channels != self.stage1_channels:
            blocks.append(nn.Conv2d(self.stem_channels, self.stage1_channels, kernel_size=1, bias=True))
        blocks += [BasicBlock2Branch(self.stage1_channels, kernel_size) for _ in range(self.num_blocks_stage1)]

        # Transition: channel jump via 1×1 conv (preserve H×W)
        if self.stage1_channels != self.stage2_channels:
            blocks += [nn.Conv2d(self.stage1_channels, self.stage2_channels, kernel_size=1, bias=True)]

        # Stage 2: residual blocks at stage2_channels
        blocks += [BasicBlock2Branch(self.stage2_channels, kernel_size) for _ in range(self.num_blocks_stage2)]

        self.blocks = nn.Sequential(*blocks)

        # Expose backbone output channels for downstream projection sizing
        self.out_channels: int = self.stage2_channels

        # Init weights: Kaiming for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.blocks(x)


class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """ResNet baseline feature extractor: 9 blocks, no pooling; outputs projection dim.

    Features dim is the projection dim (default 256) for SB3 policy heads.
    """

    def __init__(self, observation_space: Box, features_dim: int = 256, kernel_size: int = 3) -> None:
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected image observation (C,H,W)"
        c_in, H, W = int(observation_space.shape[0]), int(observation_space.shape[1]), int(observation_space.shape[2])

        self.backbone = ResNetBackbone(c_in, kernel_size)
        # Projection sizing constraint: proj_in = backbone.out_channels * H * W
        # If you change backbone widths or add spatial downsampling, update this accordingly.
        self.proj = nn.Linear(self.backbone.out_channels * H * W, features_dim)
        # Avoid in-place here as well for safe autograd on MPS/CUDA/CPU
        self.act = nn.ReLU(inplace=False)
        self._features_dim = int(features_dim)

        # Init proj
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_in", nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.backbone(observations)
        flat = x.flatten(1)
        return self.act(self.proj(flat))


