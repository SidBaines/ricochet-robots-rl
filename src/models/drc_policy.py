from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import namedtuple

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

try:
    # sb3-contrib recurrent base policy
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy as _SB3RecurrentPolicy  # type: ignore
except Exception as _e:  # pragma: no cover - optional dependency
    _SB3RecurrentPolicy = object  # type: ignore

from stable_baselines3.common.distributions import Distribution, CategoricalDistribution
from stable_baselines3.common.utils import get_device

from .convlstm import ConvLSTMFeaturesExtractor  # type: ignore


class DRCRecurrentPolicy(_SB3RecurrentPolicy):
    """sb3-contrib-compatible recurrent policy that uses our DRC ConvLSTM features.

    Notes:
    - We piggyback on sb3-contrib's recurrent API (lstm_states, episode_starts) but
      store our ConvLSTM hidden state list directly inside lstm_states. RecurrentPPO
      treats lstm_states as an opaque object to pass between calls; only episode_starts
      is used for sequence padding/masking. We handle per-env resets using that mask.
    - Action space is assumed to be Discrete.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        lr_schedule,
        features_extractor_class: type[ConvLSTMFeaturesExtractor] = ConvLSTMFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        net_arch: Optional[Dict[str, Sequence[int]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        normalize_images: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            # The parent expects to create a features extractor via policy_kwargs,
            # but we will create and manage our own extractor explicitly below.
            **kwargs,
        )

        self._device = get_device("auto")
        self.normalize_images = bool(normalize_images)
        self.activation_fn = activation_fn
        self.ortho_init = bool(ortho_init)
        self.net_arch: Dict[str, Sequence[int]] = net_arch or {"pi": [128, 128], "vf": [128, 128]}

        # Build DRC features extractor (encoder→DRC→projection)
        self.features_extractor: ConvLSTMFeaturesExtractor = features_extractor_class(
            observation_space=self.observation_space,
            **(features_extractor_kwargs or {}),
        )
        self.features_dim: int = int(self.features_extractor._features_dim)  # type: ignore[attr-defined]

        # Determine encoded spatial size (after encoder pooling) to define hidden_size
        obs_shape = tuple(self.observation_space.shape)
        assert len(obs_shape) == 3, "Expected image obs (C,H,W)"
        H, W = int(obs_shape[1]), int(obs_shape[2])
        # Encoder halves spatial dims via AvgPool2d(2)
        self._enc_H = max(1, H // 2)
        self._enc_W = max(1, W // 2)
        self._hidden_channels = int(self.features_extractor.hidden_channels)
        self._num_layers = int(self.features_extractor.num_layers)
        self._hidden_size = int(self._hidden_channels * self._enc_H * self._enc_W)

        # Expose dummy LSTM modules (as nn.Module) to satisfy sb3-contrib expectations
        class _DummyLSTMMod(nn.Module):
            def __init__(self, hidden_size: int, num_layers: int) -> None:
                super().__init__()
                self.hidden_size = int(hidden_size)
                self.num_layers = int(num_layers)
        self.lstm_actor = _DummyLSTMMod(self._hidden_size, self._num_layers)
        self.lstm_critic = _DummyLSTMMod(self._hidden_size, self._num_layers)

        # Build heads
        self.mlp_extractor = None  # not used; keep attr for compatibility
        self.action_net = nn.Sequential(
            nn.Linear(self.features_dim, self.net_arch["pi"][0]),
            self.activation_fn(),
            nn.Linear(self.net_arch["pi"][0], self.net_arch["pi"][1]),
            self.activation_fn(),
            nn.Linear(self.net_arch["pi"][1], self.action_space.n),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, self.net_arch["vf"][0]),
            self.activation_fn(),
            nn.Linear(self.net_arch["vf"][0], self.net_arch["vf"][1]),
            self.activation_fn(),
            nn.Linear(self.net_arch["vf"][1], 1),
        )

        if self.ortho_init:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    # --- Helpers ---
    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalize_images and obs.dtype == torch.uint8:
            return obs.float().div_(255.0)
        return obs.float()

    def _apply_episode_starts(self, hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]], episode_starts: torch.Tensor) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if hidden_states is None:
            return None
        if not isinstance(episode_starts, torch.Tensor):
            return hidden_states
        mask = episode_starts.bool().to(hidden_states[0][0].device)
        if mask.ndim == 0:
            mask = mask.unsqueeze(0)
        # Zero out states where episode starts
        for i, (h, c) in enumerate(hidden_states):
            if mask.numel() != h.shape[0]:
                # batch mismatch; drop states to re-init in extractor
                return None
            if mask.any():
                h = h.clone()
                c = c.clone()
                h[mask] = 0.0
                c[mask] = 0.0
                hidden_states[i] = (h, c)
        return hidden_states

    # ---- Conversion between our list[(h,c)] L layers and sb3-contrib LSTMStates ----
    _LSTMState = namedtuple("_LSTMState", ["hidden", "cell"])  # (hidden, cell)
    _LSTMStates = namedtuple("_LSTMStates", ["pi", "vf"])     # (pi: _LSTMState, vf: _LSTMState)

    def _pack_states(self, states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]) -> Any:
        if states is None or len(states) == 0:
            return None
        # states: list over L layers, each (h,c) with shape [B,C,H',W']
        h_stack_spatial = torch.stack([hc[0] for hc in states], dim=0)  # [L,B,C,H',W']
        c_stack_spatial = torch.stack([hc[1] for hc in states], dim=0)
        L, B, C, H, W = h_stack_spatial.shape
        # Flatten spatial to hidden_size: [L,B,C*H*W]
        h_flat = h_stack_spatial.reshape(L, B, C * H * W)
        c_flat = c_stack_spatial.reshape(L, B, C * H * W)
        state = self._LSTMState(hidden=h_flat, cell=c_flat)
        return self._LSTMStates(pi=state, vf=state)

    def _unpack_states(self, lstm_states: Any) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if lstm_states is None:
            return None
        try:
            h_flat = lstm_states.pi.hidden  # [L,B,hidden]
            c_flat = lstm_states.pi.cell
        except Exception:
            return None
        if not isinstance(h_flat, torch.Tensor) or not isinstance(c_flat, torch.Tensor):
            return None
        L, B, hidden = h_flat.shape
        C = self._hidden_channels
        H = self._enc_H
        W = self._enc_W
        if hidden != C * H * W:
            return None
        h_stack = h_flat.reshape(L, B, C, H, W)
        c_stack = c_flat.reshape(L, B, C, H, W)
        h_list = list(torch.unbind(h_stack, dim=0))
        c_list = list(torch.unbind(c_stack, dim=0))
        return list(zip(h_list, c_list))

    def _dist_from_features(self, features: torch.Tensor) -> CategoricalDistribution:
        dist = CategoricalDistribution(self.action_space.n)
        dist.proba_distribution(action_logits=self.action_net(features))
        return dist

    # --- sb3-contrib required API ---
    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[Any],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        # observation: [B, ...]
        obs = self._preprocess_obs(obs)

        # Unpack our DRC states from sb3-contrib LSTMStates
        drc_states = self._unpack_states(lstm_states)

        # Apply per-env reset mask
        drc_states = self._apply_episode_starts(drc_states, episode_starts)

        # Extract features via DRC core
        features, new_states = self.features_extractor(obs, drc_states)

        # Heads
        logits = self.action_net(features)
        value = self.value_net(features).squeeze(-1)

        dist = CategoricalDistribution(self.action_space.n)
        dist.proba_distribution(action_logits=logits)

        actions = dist.get_actions()
        log_prob = dist.log_prob(actions)

        # Re-pack into sb3-contrib style states
        packed = self._pack_states(new_states)
        return actions, value, log_prob, packed

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[Any],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Any]:
        obs = self._preprocess_obs(obs)
        drc_states = self._unpack_states(lstm_states)
        drc_states = self._apply_episode_starts(drc_states, episode_starts)
        features, new_states = self.features_extractor(obs, drc_states)
        return self._dist_from_features(features), self._pack_states(new_states)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: Optional[Any],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._preprocess_obs(obs)
        drc_states = self._unpack_states(lstm_states)
        drc_states = self._apply_episode_starts(drc_states, episode_starts)
        features, new_states = self.features_extractor(obs, drc_states)

        logits = self.action_net(features)
        value = self.value_net(features).squeeze(-1)

        dist = CategoricalDistribution(self.action_space.n)
        dist.proba_distribution(action_logits=logits)

        if actions.dtype != torch.long:
            actions = actions.long()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_prob, entropy

    def _predict(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[Any],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        # sb3-contrib calls this from predict()
        dist, packed_states = self.get_distribution(obs, lstm_states, episode_starts)
        actions = dist.mode() if deterministic else dist.get_actions()
        # For predict(), return (hidden, cell) tensors tuple for actor branch
        if packed_states is None:
            ret_states: Any = None
        else:
            try:
                ret_states = (packed_states.pi.hidden, packed_states.pi.cell)
            except Exception:
                ret_states = None
        return actions, ret_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[Any],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        # Return only values tensor to satisfy SB3 buffer expectations
        obs = self._preprocess_obs(obs)
        drc_states = self._unpack_states(lstm_states)
        drc_states = self._apply_episode_starts(drc_states, episode_starts)
        features, _ = self.features_extractor(obs, drc_states)
        value = self.value_net(features).squeeze(-1)
        return value


