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
from sb3_contrib.common.maskable.policies import (
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)
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


class SpatialExtractor(nn.Module):
    """Drop-in MlpExtractor computing spatially-grounded action scores.

    Receives SB3's flattened observation, restores (C, H, W), runs the shared
    residual trunk once, then:

    - actor latent = per-action scores read at each action's landing cell:
      a 1x1 conv produces one score map per action; the score for action a is
      its map contracted against the env's stop plane for a (one-hot of where
      that slide ends). "Evaluate the square you'd land on" — no flatten, no
      location-specific parameters, and the goal/stop conjunction that decides
      success is a local feature at the landing cell.
    - critic latent = global mean+max pooled trunk features through a linear.

    Motivated by the E3 v2 diagnosis: a flat readout failed to learn
    "this slide ends on the goal" at 16x16 (value flat in start distance,
    policy stuck near 3x uniform on the optimal move at d=1).
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        stop_offset: int,
        channels: int = 64,
        n_blocks: int = 6,
        vf_latent: int = 64,
    ) -> None:
        super().__init__()
        in_channels, height, width = obs_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.stop_offset = stop_offset
        self.latent_dim_pi = n_actions
        self.latent_dim_vf = vf_latent
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(channels) for _ in range(n_blocks)])
        self.score = nn.Conv2d(channels, n_actions, kernel_size=1)
        self.vf = nn.Sequential(
            nn.Linear(2 * channels, vf_latent),
            nn.LayerNorm(vf_latent),
            nn.ReLU(inplace=True),
        )

    def _trunk(self, features: torch.Tensor) -> torch.Tensor:
        x = features.reshape(features.shape[0], *self.obs_shape)
        return self.blocks(self.stem(x))

    def _actor(self, trunk: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        obs = features.reshape(features.shape[0], *self.obs_shape)
        stop = obs[:, self.stop_offset : self.stop_offset + self.n_actions]
        return (self.score(trunk) * stop).sum(dim=(2, 3))

    def _critic(self, trunk: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([trunk.mean(dim=(2, 3)), trunk.amax(dim=(2, 3))], dim=1)
        return self.vf(pooled)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self._trunk(features)
        return self._actor(trunk, features), self._critic(trunk)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self._actor(self._trunk(features), features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self._critic(self._trunk(features))


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


class SpatialPolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy with the SpatialExtractor as its torso.

    Uses SB3's default FlattenExtractor as the (identity) features extractor;
    all convolutional work happens inside the shared SpatialExtractor so actor
    and critic reuse one trunk pass. The stock action_net becomes a learned
    Linear(n_actions, n_actions) on top of the spatial scores (initialised
    near zero by SB3's small-gain orthogonal init, as usual for PPO).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        stop_offset: int,
        trunk_channels: int = 64,
        trunk_blocks: int = 6,
        vf_latent: int = 64,
        **kwargs,
    ) -> None:
        self._stop_offset = int(stop_offset)
        self._trunk_channels = int(trunk_channels)
        self._trunk_blocks = int(trunk_blocks)
        self._vf_latent = int(vf_latent)
        kwargs.pop("net_arch", None)  # torso structure is fixed by SpatialExtractor
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SpatialExtractor(
            obs_shape=self.observation_space.shape,
            n_actions=int(self.action_space.n),
            stop_offset=self._stop_offset,
            channels=self._trunk_channels,
            n_blocks=self._trunk_blocks,
            vf_latent=self._vf_latent,
        )


class SpatialRgbExtractor(nn.Module):
    """SpatialExtractor variant for Dict obs {image: pixels, planes: helpers}.

    The pixel stream is patch-embedded back to cell resolution with a
    cell-aligned strided conv (kernel = stride = cell_px), concatenated with
    the helper planes, and fed through the same residual trunk, pointer-style
    actor readout (contract score maps against the stop planes, which sit at
    offset 0 of the planes tensor) and pooled critic as SpatialExtractor.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        planes_shape: tuple[int, int, int],
        n_actions: int,
        cell_px: int,
        embed_channels: int = 32,
        channels: int = 64,
        n_blocks: int = 6,
        vf_latent: int = 64,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.planes_shape = planes_shape
        self.n_actions = n_actions
        self.image_numel = image_shape[0] * image_shape[1] * image_shape[2]
        self.latent_dim_pi = n_actions
        self.latent_dim_vf = vf_latent
        if image_shape[1] != planes_shape[1] * cell_px or image_shape[2] != planes_shape[2] * cell_px:
            raise ValueError("image dims must be cell_px x the planes grid")
        self.embed = nn.Sequential(
            nn.Conv2d(image_shape[0], embed_channels, kernel_size=cell_px, stride=cell_px),
            nn.ReLU(inplace=True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(embed_channels + planes_shape[0], channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(channels) for _ in range(n_blocks)])
        self.score = nn.Conv2d(channels, n_actions, kernel_size=1)
        self.vf = nn.Sequential(
            nn.Linear(2 * channels, vf_latent),
            nn.LayerNorm(vf_latent),
            nn.ReLU(inplace=True),
        )

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = features[:, : self.image_numel].reshape(features.shape[0], *self.image_shape)
        planes = features[:, self.image_numel :].reshape(features.shape[0], *self.planes_shape)
        return image, planes

    def _trunk(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, planes = self._split(features)
        cells = torch.cat([self.embed(image), planes], dim=1)
        return self.blocks(self.stem(cells)), planes

    def _actor(self, trunk: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
        stop = planes[:, : self.n_actions]
        return (self.score(trunk) * stop).sum(dim=(2, 3))

    def _critic(self, trunk: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([trunk.mean(dim=(2, 3)), trunk.amax(dim=(2, 3))], dim=1)
        return self.vf(pooled)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk, planes = self._trunk(features)
        return self._actor(trunk, planes), self._critic(trunk)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        trunk, planes = self._trunk(features)
        return self._actor(trunk, planes)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        trunk, _ = self._trunk(features)
        return self._critic(trunk)


class SpatialRgbPolicy(MaskableMultiInputActorCriticPolicy):
    """Masked actor-critic over Dict {image, planes} with SpatialRgbExtractor.

    The default CombinedExtractor flattens and concatenates the two streams
    (image first: gymnasium Dict spaces iterate alphabetically); the extractor
    splits them back by size. Requires stop planes at planes offset 0.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        cell_px: int,
        embed_channels: int = 32,
        trunk_channels: int = 64,
        trunk_blocks: int = 6,
        vf_latent: int = 64,
        **kwargs,
    ) -> None:
        self._cell_px = int(cell_px)
        self._embed_channels = int(embed_channels)
        self._trunk_channels = int(trunk_channels)
        self._trunk_blocks = int(trunk_blocks)
        self._vf_latent = int(vf_latent)
        kwargs.pop("net_arch", None)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SpatialRgbExtractor(
            image_shape=self.observation_space["image"].shape,
            planes_shape=self.observation_space["planes"].shape,
            n_actions=int(self.action_space.n),
            cell_px=self._cell_px,
            embed_channels=self._embed_channels,
            channels=self._trunk_channels,
            n_blocks=self._trunk_blocks,
            vf_latent=self._vf_latent,
        )
