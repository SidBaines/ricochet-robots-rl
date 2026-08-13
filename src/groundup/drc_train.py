"""Recurrent PPO for the faithful DRC network over the ground-up RicochetEnv.

Self-contained plain-PyTorch PPO-clip specialized for
:class:`~src.groundup.drc.DRCNet` + discrete actions + action masks;
deliberately NOT built on stable-baselines3 (the network's explicit
``DRCState`` threading does not fit SB3's recurrent abstractions).
Hyperparameter semantics mirror ``src/groundup/train.py``'s ``TrainConfig``
where they map cleanly: GAE(lambda), per-minibatch advantage normalization,
entropy bonus, value MSE, grad clipping, linear lr anneal
(``learning_rate`` -> ``learning_rate_final``), optional ``target_kl`` early
stop with SB3's 1.5x rule (checked before the gradient step, like SB3).

Recurrence (the part the legacy stack botched):

- Collection carries one ``DRCState`` across ``num_envs`` parallel envs.  A
  per-env state row is zeroed on the step FOLLOWING an episode end via the
  stored ``episode_starts`` flags -- the state fed to the network at step t is
  ``reset_state(state, episode_starts[t])``.
- The buffer stores the recurrent state at the START of each rollout segment
  (pre-reset, detached) plus the per-step ``episode_starts``.  A PPO
  minibatch is a subset of env-sequences (``batch_size // n_steps`` of them);
  each is replayed IN FULL from the stored initial state with the current
  weights (truncated BPTT, truncation at segment boundaries), re-applying the
  identical per-step resets.  At unchanged weights the replayed
  logprobs/values equal the collected ones exactly (tested).
- Action masks flow into ``DRCNet.forward`` during collection, evaluation,
  AND the replayed policy/value computation (masks are stored in the buffer).
- Truncation handling depends on whether the horizon is OBSERVABLE
  (``bootstrap_on_truncation``, auto-detected from ``EnvConfig.time_channel``):

  * ``time_channel=False`` -- the step limit is invisible to the agent, so it
    is a training artifact and the episode "would have continued". The value of
    the final observation, evaluated under the post-step recurrent state, is
    discounted into the step reward; this is what SB3 does when an env reports
    ``TimeLimit.truncated``.
  * ``time_channel=True`` -- the remaining-step fraction is IN the observation,
    so the horizon is part of the MDP: the episode genuinely ends and
    ``V(s, remaining=0) == 0`` by definition. Truncation is treated as a real
    terminal and NO bootstrap is applied.

  Applying the bootstrap while the horizon is observed is not a tuning choice,
  it is a bug, and combining the two raises. ``env.step`` increments ``_steps``
  before building the observation, so the expiry observation has time channel
  exactly 0 -- and it is never stored in the buffer, hence never a regression
  target. Bootstrapping there injects an unpinned extrapolation straight into
  the returns, with no contraction guarantee. That is what destroyed the
  ``drc33_p1_12x12_planefree`` run: ``V`` at expiry reached -12.3 against a
  ``goal_reward`` of 1.0 by 87M steps (timing out "paid" twelve times better
  than solving the puzzle), the return scale ran away, and because
  ``max_grad_norm`` renormalizes the COMBINED gradient the value term crowded
  the policy out of the update (share 98.8% -> 3.7% -> 0.14%) until the policy
  collapsed. Full forensics: docs/Drc87MRestartCollapseInvestigation.md 6.1.
  Note ``src/groundup/env.py`` emits no ``TimeLimit.truncated``, so the
  feed-forward ``train.py``/SB3 path never had this defect.
- Shaping consistency: every env's ``EnvConfig.gamma`` must equal the PPO
  discount (checked at construction), otherwise the potential-based shaping
  is not policy-invariant.

Post-convergence hardening (added after the 8x8 gate run collapsed from
val_det 1.00 at t=600k to 0.37 at t=650k with no mean-KL signature; see
artifacts/groundup/drc33_8x8_gate/curve.jsonl):

- ``adv_std_floor``: per-minibatch advantage normalization divides by
  ``max(std, adv_std_floor)`` instead of ``std``.  At ~100% success the
  advantage std collapses toward zero and plain normalization amplifies
  residual noise to unit scale -- the identified collapse amplifier.
- ``adv_clip``: normalized advantages are clamped to +/- adv_clip.
- ``clip_range_vf``: PPO2-style value clipping (element-wise max of clipped
  and unclipped losses; default off), bounding the
  shared-trunk gradients from value-loss spikes.
- Keep-best checkpointing: ``model_best.pt`` is saved whenever the periodic
  val_det improves. Closeout always reports ``model_final.pt`` as the headline
  result and evaluates the periodic best separately; ``approx_kl_max`` (max
  per-minibatch KL per update) is logged for telemetry.

Setting ``adv_std_floor: 0``, ``adv_clip: null``, ``clip_range_vf: null``
recovers the pre-hardening update semantics exactly.

Usage:
    .venv-phase2/bin/python -m src.groundup.drc_train --config configs/<name>.yaml

Every run writes to <out_dir>/<run_name>/: config.json, curve.jsonl (periodic
masked validation evals + curriculum state + losses), report.json (final
evals and scripted baselines under the identical masked full-start protocol),
model_latest.pt / model_final.pt.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

from .curriculum import ReverseFrontierSampler, UniformSampler
from .drc import DRCNet, DRCState
from .env import EnvConfig, RicochetEnv
from .evaluate import _aggregate, scripted_baseline


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DRCTrainConfig:
    bank_path: str = ""
    out_dir: str = "artifacts/groundup"
    run_name: str = "drc_run"
    # splits (must match legacy discipline)
    eval_fraction: float = 0.25
    split_seed: int = 20_260_710
    val_test_seed: int = 20_260_711
    # env
    horizon: int = 32
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    shaping_coef: float = 0.1
    time_channel: bool = True
    canonical_identity: bool = True
    stop_planes: bool = False
    distance_plane: bool = False
    # curriculum: "reverse" | "uniform"
    curriculum: str = "reverse"
    tau: float = 0.75
    ema_beta: float = 0.02
    min_episodes: int = 200
    frontier_p: float = 0.5
    retention_p: float = 0.25
    uniform_p: float = 0.125
    # optional overfit mode: train on a tiny fixed subset, eval on the same
    overfit_n: Optional[int] = None
    # PPO. batch_size is in TIMESTEPS but must be a whole number of
    # n_steps-long env-sequences (a recurrent minibatch replays whole
    # sequences); defaults are shorter than train.py's because BPTT unrolls
    # the full n_steps.
    seed: int = 11
    num_envs: int = 16
    n_steps: int = 64
    batch_size: int = 256
    n_epochs: int = 2
    total_steps: int = 1_000_000
    # Learning-rate progress is measured against this horizon while
    # ``total_steps`` is the current stop/review target. This permits the
    # pre-registered 60M -> 120M -> 200M extensions without restarting or
    # prematurely exhausting the schedule.
    schedule_total_steps: Optional[int] = None
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-4
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
    # Post-convergence hardening (0 / null / null disables all three and
    # recovers the pre-hardening semantics exactly).
    # adv_std_floor: with goal_reward 1.0 (advantage scale O(0.1-1) while
    # learning), a per-minibatch advantage std below 0.05 practically only
    # occurs once the batch is (near-)solved and advantages are residual
    # noise; flooring the denominator caps the noise amplification at 20x
    # instead of inflating ~1e-3-scale residuals to unit scale.
    adv_std_floor: float = 0.05
    # adv_clip: normalized advantages are ~N(0,1), so +/-10 only removes
    # pathological outliers (e.g. produced by tiny-std minibatches).
    adv_clip: Optional[float] = 10.0
    # clip_range_vf: SB3-style value clipping, set to match clip_range (the
    # SB3 convention when enabled); bounds the per-update value change per
    # state so value-loss spikes (0.45-0.50 observed pre-collapse) cannot
    # drive huge gradients through the shared ConvLSTM trunk.
    clip_range_vf: Optional[float] = None
    # Whether a horizon truncation bootstraps V(final_obs) into the reward.
    # ``None`` auto-detects from ``time_channel`` (bootstrap only when the
    # horizon is hidden); an explicit value is validated against the envs and
    # raises on the unsound combination. This is a CONTRACT field: it is part of
    # the config fingerprint, so the fingerprint pins the reward semantics -- the
    # earlier derived-only version left runs before and after the
    # truncation-bootstrap fix sharing a fingerprint. Registered configs should
    # set it explicitly. See the module docstring and
    # docs/Drc87MRestartCollapseInvestigation.md 6.1.
    bootstrap_on_truncation: Optional[bool] = None
    # Auxiliary landing-cell loss (docs/DrcAuxLandingLossPlan.md). 0.0 = off, and
    # off is bit-identical to a build without the feature (regression-tested).
    # Supervises where each robot would stop per direction, from labels the env
    # computes for free; the observation stays plane-free.
    aux_landing_coef: float = 0.0
    # Contract fields this run may deliberately change relative to the checkpoint
    # it resumes, e.g. ["aux_landing_coef"] to branch an experiment arm off a
    # control checkpoint with a changed objective. Anything NOT listed here still
    # refuses to load, so semantics can never drift silently -- but a declared,
    # audited change is permitted. The resulting run is a BRANCH, not an exact
    # continuation, and the trainer says so loudly at startup.
    resume_allow_contract_change: Optional[Sequence[str]] = None
    # DRC network (paper defaults: DRC(3,3), 32 channels)
    drc_depth: int = 3
    drc_ticks: int = 3
    drc_hidden_channels: int = 32
    drc_cell_kernel: int = 3
    drc_encoder_channels: Optional[Sequence[int]] = None  # default: (hidden, hidden)
    drc_encoder_kernels: Optional[Sequence[int]] = None  # default: 4 each
    drc_encoder_strides: Optional[Sequence[int]] = None  # default: 1 each
    drc_encoder_relu: bool = False
    drc_mlp_hidden: int = 256
    # evaluation
    eval_every: int = 100_000
    eval_per_length: int = 25
    final_eval_per_length: int = 50
    final_eval_episodes: int = 3
    # Evaluation-only joint (optimal_length, robots_moved) diagnostic. Every
    # observed validation stratum contributes up to this many puzzles.
    hard_eval_per_stratum: int = 25
    eval_batch_size: int = 32
    # Immutable lightweight model and full training-state snapshots.
    checkpoint_every: Optional[int] = None
    resume_from: Optional[str] = None
    device: str = "auto"
    # W&B (opt-in, mirrors the train.py convention)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DRCTrainConfig":
        values = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)


def build_net(
    config: DRCTrainConfig, observation_shape: Tuple[int, int, int], n_actions: int
) -> DRCNet:
    """Instantiate a DRCNet from the config's drc_* fields and the env spaces."""
    if config.drc_encoder_channels is not None:
        channels = tuple(int(c) for c in config.drc_encoder_channels)
    else:
        channels = (config.drc_hidden_channels, config.drc_hidden_channels)
    if config.drc_encoder_kernels is not None:
        kernels = tuple(int(k) for k in config.drc_encoder_kernels)
    else:
        kernels = (4,) * len(channels)
    if config.drc_encoder_strides is not None:
        strides = tuple(int(s) for s in config.drc_encoder_strides)
    else:
        strides = (1,) * len(channels)
    return DRCNet(
        observation_shape=tuple(observation_shape),
        n_actions=int(n_actions),
        depth=config.drc_depth,
        ticks=config.drc_ticks,
        hidden_channels=config.drc_hidden_channels,
        cell_kernel_size=config.drc_cell_kernel,
        encoder_channels=channels,
        encoder_kernels=kernels,
        encoder_strides=strides,
        encoder_relu=config.drc_encoder_relu,
        mlp_hidden=config.drc_mlp_hidden,
    )


# --------------------------------------------------------------- core algorithm bits
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    episode_starts: torch.Tensor,
    last_values: torch.Tensor,
    last_episode_starts: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda) with SB3 semantics.

    Args:
        rewards / values: (T, B) float tensors.
        episode_starts: (T, B) bool; True where step t is the FIRST step of an
            episode (i.e. the previous step ended one).
        last_values: (B,) value of the observation after the last stored step.
        last_episode_starts: (B,) bool; True where the last stored step ended
            its episode (so ``last_values`` must not be bootstrapped).

    Returns:
        (advantages, returns), both (T, B). ``returns = advantages + values``.
    """
    n_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_values)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - last_episode_starts.float()
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[t + 1].float()
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def normalize_advantages(
    advantages: torch.Tensor,
    std_floor: float = 0.0,
    clip: Optional[float] = None,
) -> torch.Tensor:
    """Per-minibatch advantage normalization with hardening guards.

    ``std_floor`` floors the normalization denominator (``std_floor=0``
    reproduces the classic ``(a - mean) / (std + 1e-8)`` exactly); ``clip``
    clamps the result to +/- clip (``None`` disables). See the module
    docstring for why the guards exist.
    """
    if advantages.numel() > 1:
        denom = torch.clamp(advantages.std(), min=std_floor) + 1e-8
        advantages = (advantages - advantages.mean()) / denom
    if clip is not None:
        advantages = advantages.clamp(-clip, clip)
    return advantages


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range_vf: Optional[float] = None,
) -> torch.Tensor:
    """Value MSE with PPO2-style clipping (element-wise max of losses).

    The clip-only variant (SB3's ``clip_range_vf``: loss on the clamped
    prediction alone) under-corrects large errors — gradients vanish once the
    prediction has moved past the clip, so the value net can only drift
    clip_range_vf per rollout while the truncation bootstrap keeps feeding the
    biased V back into rewards. That positive feedback produced exponential
    value divergence in the drc33_8x8_gate_v2 run (value_loss 43 -> 9.4e6,
    policy entropy -> 0). The max form keeps a full correction gradient
    whenever the unclipped error dominates; ``None`` is the plain MSE.
    """
    unclipped = (returns - values) ** 2
    if clip_range_vf is None:
        return unclipped.mean()
    clipped_pred = old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
    clipped = (returns - clipped_pred) ** 2
    return torch.maximum(unclipped, clipped).mean()


def replay_sequences(
    net: DRCNet,
    obs: torch.Tensor,
    masks: torch.Tensor,
    episode_starts: torch.Tensor,
    initial_state: DRCState,
    *,
    return_maps: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replay stored sequences through the net (BPTT over the whole segment).

    Applies exactly the collection-time state semantics: before step t the
    state rows flagged by ``episode_starts[t]`` are zeroed. Gradients flow
    through time within the segment; ``initial_state`` is treated as constant
    (truncated BPTT at segment boundaries).

    Args:
        obs: (T, B, C, H, W) float32.
        masks: (T, B, A) bool.
        episode_starts: (T, B) bool.
        initial_state: DRCState with batch dim B (pre-reset segment state).

    Returns:
        (logits (T, B, A), values (T, B)), plus the spatial readout maps
        (T, B, C, H, W) when ``return_maps`` (for the auxiliary head).
    """
    state = initial_state
    logits_seq: List[torch.Tensor] = []
    values_seq: List[torch.Tensor] = []
    maps_seq: List[torch.Tensor] = []
    for t in range(obs.shape[0]):
        state = DRCNet.reset_state(state, episode_starts[t])
        if return_maps:
            logits, values, state, readout_map = net(
                obs[t], state, action_mask=masks[t], return_map=True
            )
            maps_seq.append(readout_map)
        else:
            logits, values, state = net(obs[t], state, action_mask=masks[t])
        logits_seq.append(logits)
        values_seq.append(values)
    if return_maps:
        return torch.stack(logits_seq), torch.stack(values_seq), torch.stack(maps_seq)
    return torch.stack(logits_seq), torch.stack(values_seq)


class RolloutBuffer:
    """Fixed-size (n_steps, num_envs) on-policy storage for one segment."""

    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        device: torch.device,
    ) -> None:
        self.obs = torch.zeros((n_steps, num_envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)
        self.masks = torch.zeros((n_steps, num_envs, n_actions), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
        self.episode_starts = torch.zeros((n_steps, num_envs), dtype=torch.bool, device=device)
        # Auxiliary landing-cell targets: flat cell index per (robot, direction).
        # Only filled when the aux loss is enabled.
        self.landing: Optional[torch.Tensor] = None
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        # DRCState at the START of the segment, before any episode_starts[0]
        # reset is applied (the replay applies it, exactly like collection).
        self.initial_state: Optional[DRCState] = None


class RecurrentPPO:
    """Single-purpose recurrent PPO for DRCNet on masked discrete envs.

    ``envs`` is a plain list of gymnasium-style envs exposing ``reset()``,
    ``step()`` (5-tuple), ``action_masks()``, ``observation_space`` and
    ``action_space``. Envs are stepped synchronously with a single batched
    network forward per vec step.
    """

    def __init__(
        self,
        net: DRCNet,
        envs: Sequence,
        *,
        n_steps: int = 64,
        batch_size: int = 256,
        n_epochs: int = 2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        learning_rate_final: Optional[float] = None,
        target_kl: Optional[float] = 0.05,
        adv_std_floor: float = 0.05,
        adv_clip: Optional[float] = 10.0,
        clip_range_vf: Optional[float] = None,
        bootstrap_on_truncation: Optional[bool] = None,
        aux_landing_coef: float = 0.0,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if not envs:
            raise ValueError("Need at least one env")
        if batch_size % n_steps != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a whole number of n_steps-long "
                f"sequences (n_steps={n_steps}): recurrent minibatches replay whole sequences"
            )
        seqs = batch_size // n_steps
        if not 1 <= seqs <= len(envs):
            raise ValueError(
                f"batch_size/n_steps = {seqs} sequences per minibatch must be in [1, num_envs={len(envs)}]"
            )
        for env in envs:
            env_gamma = getattr(getattr(env, "config", None), "gamma", None)
            if env_gamma is not None and abs(float(env_gamma) - float(gamma)) > 1e-12:
                raise ValueError(
                    f"env shaping gamma ({env_gamma}) != PPO gamma ({gamma}); "
                    "potential-based shaping is only policy-invariant when they match"
                )
        # Truncation bootstrapping is sound ONLY when the step limit is invisible
        # to the agent. See the module docstring: with time_channel=True the
        # horizon is part of the observation, V(s, remaining=0) == 0 by
        # definition, and the expiry observation is never a regression target --
        # so bootstrapping injects an unpinned extrapolation into the reward.
        # Envs that expose no ``config.time_channel`` (duck-typed/foreign envs)
        # are assumed to hide their limit, which is the SB3-compatible default.
        observes_horizon = {
            getattr(getattr(env, "config", None), "time_channel", None) for env in envs
        }
        if len(observes_horizon) > 1:
            raise ValueError(
                f"Envs disagree on time_channel ({sorted(map(str, observes_horizon))}); "
                "truncation semantics must be identical across envs"
            )
        horizon_is_observed = observes_horizon.pop() is True
        if bootstrap_on_truncation is None:
            bootstrap_on_truncation = not horizon_is_observed
        elif bootstrap_on_truncation and horizon_is_observed:
            raise ValueError(
                "bootstrap_on_truncation=True is invalid when the env exposes the "
                "remaining horizon (time_channel=True): the MDP is then "
                "finite-horizon and fully observed, so V(s, remaining=0) == 0 by "
                "definition, and the expiry observation is never stored in the "
                "buffer -- bootstrapping there injects an untrained extrapolation "
                "into the reward (see docs/Drc87MRestartCollapseInvestigation.md)"
            )
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self.envs = list(envs)
        self.num_envs = len(self.envs)
        self.obs_shape = tuple(self.envs[0].observation_space.shape)
        self.n_actions = int(self.envs[0].action_space.n)
        if tuple(net.observation_shape) != self.obs_shape or net.n_actions != self.n_actions:
            raise ValueError(
                f"net expects obs {net.observation_shape} / {net.n_actions} actions, "
                f"env provides {self.obs_shape} / {self.n_actions}"
            )
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr0 = learning_rate
        self.lr_final = learning_rate if learning_rate_final is None else learning_rate_final
        self.target_kl = target_kl
        self.adv_std_floor = adv_std_floor
        self.adv_clip = adv_clip
        self.clip_range_vf = clip_range_vf

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self._gen = torch.Generator().manual_seed(seed)  # minibatch shuffling
        self.num_timesteps = 0

        # --- auxiliary landing-cell head (docs/DrcAuxLandingLossPlan.md) ------
        # Supervises "where would each robot stop if it slid this way", the
        # knowledge the 120M probes showed is NOT encoded (mid/long decodability
        # sits BELOW a random-init control and declines through training).
        # Labels come free from the env; the observation is untouched.
        #
        # The head lives OUTSIDE self.net with its OWN optimizer, deliberately:
        # a mid-training arm must resume a checkpoint written without it, and
        # this keeps net.state_dict() and the main Adam state byte-identical to
        # the control arm's. Aux gradients still reach the shared trunk.
        self.aux_landing_coef = float(aux_landing_coef)
        self.aux_head: Optional[torch.nn.Module] = None
        self.aux_optimizer: Optional[torch.optim.Optimizer] = None
        if self.aux_landing_coef > 0.0:
            self.aux_head = torch.nn.Conv2d(
                net.hidden_channels, self.n_actions, kernel_size=1
            ).to(self.device)
            self.aux_optimizer = torch.optim.Adam(
                self.aux_head.parameters(), lr=learning_rate
            )

        obs0 = []
        for env in self.envs:
            obs, _info = env.reset()
            obs0.append(obs)
        self._obs = np.stack(obs0).astype(np.float32)
        self._episode_start = np.ones(self.num_envs, dtype=bool)
        self._state = self.net.initial_state(self.num_envs, device=self.device)

    # --------------------------------------------------------------- resume
    def state_dict(self) -> Dict[str, object]:
        """Complete iteration-boundary state for exact crash recovery."""
        env_states = []
        for env in self.envs:
            if not hasattr(env, "state_dict"):
                raise TypeError(f"Environment {type(env).__name__} does not support exact resume")
            env_states.append(env.state_dict())
        state: Dict[str, object] = {
            # v2 (2026-07-25): ``device`` is required. Schema 1 checkpoints are
            # rejected outright rather than silently resumed without the
            # cross-device check -- see load_state_dict.
            "schema_version": 2,
            "contract": {
                "num_envs": self.num_envs,
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
            },
            # The RESOLVED device, not the config's ``device`` string. ``auto``
            # resolves to mps or cpu per machine while the config records only
            # "auto", so the config-contract fingerprint cannot distinguish them
            # and a cross-device resume would otherwise pass every check while
            # being measurably non-exact. Checked in ``load_state_dict``.
            "device": str(self.device),
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "minibatch_generator_state": self._gen.get_state(),
            "num_timesteps": self.num_timesteps,
            "obs": self._obs.copy(),
            "episode_start": self._episode_start.copy(),
            "recurrent_state": {
                "h": self._state.h.detach().cpu(),
                "c": self._state.c.detach().cpu(),
            },
            "env_states": env_states,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["device_rng_state"] = ("cuda", torch.cuda.get_rng_state_all())
        elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
            state["device_rng_state"] = ("mps", torch.mps.get_rng_state())
        return state

    def load_state_dict(self, state: Dict[str, object], *, allow_device_change: bool = False) -> None:
        if int(state.get("schema_version", -1)) != 2:
            raise ValueError(
                f"Unsupported recurrent PPO checkpoint schema: "
                f"{state.get('schema_version')} (this build writes and requires 2). "
                "Schema 1 predates the truncation-bootstrap fix and the recorded "
                "resolved device, so its training state cannot be resumed exactly; "
                "load its weights from the matching model_*.pt instead."
            )
        # This format's whole point is EXACT resume, and exactness holds only
        # within a device: replaying the same 87M checkpoint gave iteration-1
        # value_loss 18.1032 on cpu vs 19.8086 on mps.
        stored_device = state["device"]
        if not allow_device_change and str(stored_device) != str(self.device):
            raise ValueError(
                f"Checkpoint was written on device '{stored_device}' but this run "
                f"resolved to '{self.device}'; resume would not be exact. Note "
                "device='auto' resolves per machine, so the config contract cannot "
                "catch this. Set device explicitly, or pass "
                "allow_device_change=True to accept an inexact resume (prefer "
                "loading model_*.pt weights instead for a genuine migration)."
            )
        if allow_device_change and str(stored_device) != str(self.device):
            print(
                f"[drc] WARNING: resuming a '{stored_device}' checkpoint on "
                f"'{self.device}'; this resume is NOT exact.",
                flush=True,
            )
        expected = {
            "num_envs": self.num_envs,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
        }
        actual = state["contract"]
        if actual != expected:
            raise ValueError(f"Recurrent PPO contract changed: checkpoint={actual} current={expected}")

        self.net.load_state_dict(state["net_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._gen.set_state(state["minibatch_generator_state"].cpu())
        self.num_timesteps = int(state["num_timesteps"])

        obs = np.asarray(state["obs"], dtype=np.float32)
        episode_start = np.asarray(state["episode_start"], dtype=bool)
        if obs.shape != self._obs.shape or episode_start.shape != self._episode_start.shape:
            raise ValueError(
                f"Rollout state shape changed: checkpoint={obs.shape}/{episode_start.shape} "
                f"current={self._obs.shape}/{self._episode_start.shape}"
            )
        self._obs[...] = obs
        self._episode_start[...] = episode_start
        recurrent = state["recurrent_state"]
        self._state = DRCState(
            h=recurrent["h"].to(self.device),
            c=recurrent["c"].to(self.device),
        )
        if len(state["env_states"]) != self.num_envs:
            raise ValueError(
                f"Environment count changed: checkpoint={len(state['env_states'])} "
                f"current={self.num_envs}"
            )
        for env, env_state in zip(self.envs, state["env_states"]):
            if not hasattr(env, "load_state_dict"):
                raise TypeError(f"Environment {type(env).__name__} does not support exact resume")
            env.load_state_dict(env_state)

        torch.set_rng_state(state["torch_rng_state"].cpu())
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        device_rng = state.get("device_rng_state")
        if device_rng is not None:
            kind, rng_state = device_rng
            if kind == "cuda" and self.device.type == "cuda":
                torch.cuda.set_rng_state_all([value.cpu() for value in rng_state])
            elif kind == "mps" and self.device.type == "mps":
                torch.mps.set_rng_state(rng_state.cpu())

    # -------------------------------------------------------------- collection
    def collect_rollout(self) -> RolloutBuffer:
        """Collect n_steps transitions from all envs (one forward per vec step)."""
        net = self.net
        buf = RolloutBuffer(self.n_steps, self.num_envs, self.obs_shape, self.n_actions, self.device)
        buf.initial_state = DRCState(
            h=self._state.h.detach().clone(), c=self._state.c.detach().clone()
        )
        if self.aux_landing_coef > 0.0:
            buf.landing = torch.zeros(
                (self.n_steps, self.num_envs, self.n_actions), dtype=torch.long, device=self.device
            )
        with torch.no_grad():
            for t in range(self.n_steps):
                masks_np = np.stack([env.action_masks() for env in self.envs])
                obs_t = torch.as_tensor(self._obs, device=self.device)
                mask_t = torch.as_tensor(masks_np, device=self.device)
                ep_start_t = torch.as_tensor(self._episode_start, device=self.device)
                buf.obs[t] = obs_t
                buf.masks[t] = mask_t
                buf.episode_starts[t] = ep_start_t
                if buf.landing is not None:
                    buf.landing[t] = torch.as_tensor(
                        np.stack([env.landing_targets() for env in self.envs]),
                        device=self.device,
                    )

                state_in = DRCNet.reset_state(self._state, ep_start_t)
                logits, values, new_state = net(obs_t, state_in, action_mask=mask_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                buf.actions[t] = actions
                buf.log_probs[t] = dist.log_prob(actions)
                buf.values[t] = values

                actions_np = actions.cpu().numpy()
                rewards = np.zeros(self.num_envs, dtype=np.float32)
                new_starts = np.zeros(self.num_envs, dtype=bool)
                trunc_slots: List[int] = []
                trunc_obs: List[np.ndarray] = []
                for b, env in enumerate(self.envs):
                    obs_b, reward, terminated, truncated, _info = env.step(int(actions_np[b]))
                    rewards[b] = reward
                    if terminated or truncated:
                        if truncated and not terminated and self.bootstrap_on_truncation:
                            # Hidden step limit: the episode "would have
                            # continued", so bootstrap the value of the final
                            # observation (SB3's TimeLimit.truncated semantics).
                            # Skipped when the horizon is observable -- expiry is
                            # then a real terminal with value 0 by definition.
                            trunc_slots.append(b)
                            trunc_obs.append(obs_b)
                        obs_b, _info = env.reset()
                        new_starts[b] = True
                    self._obs[b] = obs_b
                if trunc_slots:
                    final_obs = torch.as_tensor(np.stack(trunc_obs), device=self.device)
                    # State rows AFTER the step that produced the final obs,
                    # before any episode reset (the value head ignores masks).
                    boot_state = DRCState(
                        h=new_state.h[:, trunc_slots], c=new_state.c[:, trunc_slots]
                    )
                    _, boot_values, _ = net(final_obs, boot_state)
                    rewards[trunc_slots] += self.gamma * boot_values.cpu().numpy()
                buf.rewards[t] = torch.as_tensor(rewards, device=self.device)

                self._state = new_state
                self._episode_start = new_starts
                self.num_timesteps += self.num_envs

            # Value of the observation after the last stored step (GAE bootstrap).
            ep_start_t = torch.as_tensor(self._episode_start, device=self.device)
            state_in = DRCNet.reset_state(self._state, ep_start_t)
            _, last_values, _ = net(torch.as_tensor(self._obs, device=self.device), state_in)

        buf.advantages, buf.returns = compute_gae(
            buf.rewards,
            buf.values,
            buf.episode_starts,
            last_values,
            ep_start_t,
            self.gamma,
            self.gae_lambda,
        )
        return buf

    # ------------------------------------------------------------------ update
    def update(self, buf: RolloutBuffer, progress_remaining: float = 1.0) -> Dict[str, object]:
        """One PPO update over the buffer: n_epochs of whole-sequence minibatches."""
        lr = self.lr_final + (self.lr0 - self.lr_final) * progress_remaining
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        seqs_per_minibatch = self.batch_size // self.n_steps
        pg_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fracs: List[float] = []
        aux_losses: List[float] = []
        n_grad_updates = 0
        continue_training = True

        for _epoch in range(self.n_epochs):
            perm = torch.randperm(self.num_envs, generator=self._gen).tolist()
            for start in range(0, self.num_envs, seqs_per_minibatch):
                idx = perm[start : start + seqs_per_minibatch]
                init = DRCState(
                    h=buf.initial_state.h[:, idx], c=buf.initial_state.c[:, idx]
                )
                want_maps = self.aux_head is not None
                if want_maps:
                    logits, values, maps = replay_sequences(
                        self.net, buf.obs[:, idx], buf.masks[:, idx],
                        buf.episode_starts[:, idx], init, return_maps=True,
                    )
                else:
                    logits, values = replay_sequences(
                        self.net, buf.obs[:, idx], buf.masks[:, idx],
                        buf.episode_starts[:, idx], init,
                    )
                dist = Categorical(logits=logits.reshape(-1, self.n_actions))
                actions = buf.actions[:, idx].reshape(-1)
                new_log_prob = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                values = values.reshape(-1)

                advantages = normalize_advantages(
                    buf.advantages[:, idx].reshape(-1),
                    std_floor=self.adv_std_floor,
                    clip=self.adv_clip,
                )
                log_ratio = new_log_prob - buf.log_probs[:, idx].reshape(-1)
                ratio = torch.exp(log_ratio)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range),
                ).mean()
                value_loss = clipped_value_loss(
                    values,
                    buf.values[:, idx].reshape(-1),
                    buf.returns[:, idx].reshape(-1),
                    clip_range_vf=self.clip_range_vf,
                )
                loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

                aux_loss_value = 0.0
                if self.aux_head is not None:
                    # Per (robot, direction) pair, classify the landing cell over
                    # the H*W grid -- the same formulation the landing probes
                    # score, so aux loss and probe accuracy are comparable.
                    t_len, n_seq = maps.shape[0], maps.shape[1]
                    flat_maps = maps.reshape(t_len * n_seq, *maps.shape[2:])
                    cell_logits = self.aux_head(flat_maps).flatten(2)  # (N, pairs, H*W)
                    targets = buf.landing[:, idx].reshape(t_len * n_seq, self.n_actions)
                    aux_loss = torch.nn.functional.cross_entropy(
                        cell_logits.reshape(-1, cell_logits.shape[-1]), targets.reshape(-1)
                    )
                    loss = loss + self.aux_landing_coef * aux_loss
                    aux_loss_value = float(aux_loss.detach())
                aux_losses.append(aux_loss_value)

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                    clip_frac = float(((ratio - 1.0).abs() > self.clip_range).float().mean())
                pg_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach()))
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

                # SB3 rule: check before applying this minibatch's gradient.
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.optimizer.zero_grad()
                if self.aux_optimizer is not None:
                    self.aux_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.aux_optimizer is not None:
                    # Clipped separately: the head is not in net.parameters(), and
                    # its own gradient must not eat the trunk's budget.
                    torch.nn.utils.clip_grad_norm_(self.aux_head.parameters(), self.max_grad_norm)
                    self.aux_optimizer.step()
                n_grad_updates += 1
            if not continue_training:
                break

        returns_flat = buf.returns.flatten()
        var_returns = float(torch.var(returns_flat))
        explained_variance = (
            None
            if var_returns == 0.0
            else 1.0 - float(torch.var(returns_flat - buf.values.flatten())) / var_returns
        )
        return {
            "lr": lr,
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            # Max per-minibatch KL: collapse telemetry the mean hides.
            "approx_kl_max": float(np.max(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": explained_variance,
            "n_grad_updates": n_grad_updates,
            "early_stop": not continue_training,
            "aux_landing_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
        }

    # ------------------------------------------------------------------- learn
    def learn(
        self,
        total_steps: int,
        on_iteration: Optional[Callable[["RecurrentPPO", Dict[str, object]], None]] = None,
        *,
        schedule_total_steps: Optional[int] = None,
    ) -> None:
        schedule_steps = total_steps if schedule_total_steps is None else schedule_total_steps
        if schedule_steps < total_steps:
            raise ValueError(
                f"schedule_total_steps ({schedule_steps}) must be >= total_steps ({total_steps})"
            )
        while self.num_timesteps < total_steps:
            buf = self.collect_rollout()
            progress_remaining = max(0.0, 1.0 - self.num_timesteps / schedule_steps)
            metrics = self.update(buf, progress_remaining)
            if on_iteration is not None:
                on_iteration(self, metrics)


# ------------------------------------------------------------------- evaluation
def evaluate_drc(
    net: DRCNet,
    puzzles: Sequence,
    config: EnvConfig,
    *,
    episodes_per_puzzle: int = 1,
    deterministic: bool = True,
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    """Masked, full-start, recurrent-aware evaluation over a puzzle list.

    Mirrors ``evaluate.evaluate_policy`` (same protocol, same aggregate shape
    via ``_aggregate``) but threads a per-slot DRCState, zeroing a slot's
    state whenever a new puzzle is loaded into it.

    RNG discipline: evaluation is a pure function of ``seed`` and must not
    perturb the caller's generators. It therefore samples from a LOCAL
    ``torch.Generator`` rather than reseeding the global one. The previous
    ``torch.manual_seed(seed)`` reseeded the CPU *and* device generators, and
    because a periodic eval runs immediately before every checkpoint it made the
    stored RNG state the same constant in every checkpoint and re-anchored the
    training action-sampling stream to seed 0 every ``eval_every`` steps -- so
    exploration noise repeated rather than continuing. See
    docs/Drc87MRestartCollapseInvestigation.md 3.2. Sampling is done on CPU so
    the action stream is identical on every device.
    """
    device = next(net.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    work = [(i, rep) for i in range(len(puzzles)) for rep in range(episodes_per_puzzle)]
    work.reverse()  # pop() consumes in order
    n_slots = min(batch_size, len(work))
    envs = [RicochetEnv(puzzles, config, sampler=None, seed=seed + s) for s in range(n_slots)]
    init = net.initial_state(n_slots, device=device)
    state_h, state_c = init.h.clone(), init.c.clone()
    slot_puzzle: List[Optional[int]] = [None] * n_slots
    obs_list: List[Optional[np.ndarray]] = [None] * n_slots
    records: List[dict] = []

    def load_next(slot: int) -> None:
        state_h[:, slot].zero_()  # fresh episode -> fresh recurrent state
        state_c[:, slot].zero_()
        if not work:
            slot_puzzle[slot] = None
            obs_list[slot] = None
            return
        idx, _rep = work.pop()
        slot_puzzle[slot] = idx
        obs, _ = envs[slot].reset(options={"puzzle_index": idx})
        obs_list[slot] = obs

    for slot in range(n_slots):
        load_next(slot)

    was_training = net.training
    net.eval()
    with torch.no_grad():
        while any(p is not None for p in slot_puzzle):
            active = [s for s in range(n_slots) if slot_puzzle[s] is not None]
            obs_batch = torch.as_tensor(np.stack([obs_list[s] for s in active]), device=device)
            mask_batch = torch.as_tensor(
                np.stack([envs[s].action_masks() for s in active]), device=device
            )
            state_in = DRCState(h=state_h[:, active], c=state_c[:, active])
            logits, _values, new_state = net(obs_batch, state_in, action_mask=mask_batch)
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                # Categorical.sample() takes no generator, so draw explicitly
                # from the local one (on CPU: device-independent stream).
                probs = torch.softmax(logits.float(), dim=-1).cpu()
                actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            state_h[:, active] = new_state.h
            state_c[:, active] = new_state.c
            actions_np = actions.cpu().numpy()
            for k, s in enumerate(active):
                obs, _rew, term, trunc, info = envs[s].step(int(actions_np[k]))
                if term or trunc:
                    puzzle = puzzles[slot_puzzle[s]]
                    records.append(
                        {
                            "success": bool(info["is_success"]),
                            "optimal_length": puzzle.optimal_length,
                            "robots_moved": puzzle.robots_moved,
                            "steps": info["steps"],
                        }
                    )
                    load_next(s)
                else:
                    obs_list[s] = obs
    net.train(was_training)
    result = _aggregate(records)
    result["deterministic"] = deterministic
    return result


# ----------------------------------------------------------- logging helpers
# (kept identical in behaviour/metric names to src/groundup/train.py)
def _flatten_eval(prefix: str, summary: dict) -> dict:
    out = {f"{prefix}/success_rate": summary["success_rate"]}
    for length, v in summary.get("by_length", {}).items():
        out[f"{prefix}/len_{length}"] = v["success_rate"]
    for robots, v in summary.get("by_robots_moved", {}).items():
        out[f"{prefix}/robots_{robots}"] = v["success_rate"]
    return out


def _flatten_sampler(stats: dict) -> dict:
    out = {}
    if "frontier" in stats:
        out["curriculum/frontier"] = stats["frontier"]
        for d, value in stats.get("ema", {}).items():
            out[f"curriculum/ema_d{d}"] = value
    return out


def _maybe_init_wandb(config: "DRCTrainConfig", out_dir: Path):
    if config.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or config.run_name,
        config=dataclasses.asdict(config),
        dir=str(out_dir),
    )
    run.define_metric("train/timesteps")
    for pattern in (
        "train/*",
        "curriculum/*",
        "val_det/*",
        "final/*",
        "best_periodic/*",
    ):
        run.define_metric(pattern, step_metric="train/timesteps")
    return run


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


_RESUME_MUTABLE_CONFIG_FIELDS = {
    "out_dir",
    "run_name",
    "total_steps",
    "resume_from",
    "eval_every",
    "checkpoint_every",
    "eval_per_length",
    "final_eval_per_length",
    "final_eval_episodes",
    "hard_eval_per_stratum",
    "eval_batch_size",
    "wandb_project",
    "wandb_run_name",
    # An operational declaration about THIS launch, not a property of the
    # algorithm, so it must not move the contract fingerprint.
    "resume_allow_contract_change",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _config_contract(config: DRCTrainConfig) -> Dict[str, object]:
    values = dataclasses.asdict(config)
    return {key: value for key, value in values.items() if key not in _RESUME_MUTABLE_CONFIG_FIELDS}


def _contract_differences(stored: Dict[str, object], expected: Dict[str, object]) -> List[str]:
    """Fields in which a stored config contract is incompatible with the current one.

    Additive evolution is allowed: a field that did not exist when the checkpoint
    was written is compatible **iff** the current config leaves it at its
    default, i.e. the run behaves exactly as the checkpoint's build did. Set it
    to anything else and the resume is refused, so the contract still pins the
    semantics. Without this, adding any config field would make every existing
    checkpoint unresumable -- which would have blocked the aux-loss arms from
    continuing the 120M run at all.
    """
    defaults = {field.name: field.default for field in dataclasses.fields(DRCTrainConfig)}
    changed: List[str] = []
    for key in set(stored) | set(expected):
        if key in stored and key in expected:
            if stored[key] != expected[key]:
                changed.append(key)
        elif key not in stored:
            if expected[key] != defaults.get(key):
                changed.append(key)  # new field, actively set -> semantics differ
        else:
            changed.append(key)  # field removed from the config class
    return sorted(changed)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _puzzle_fingerprint(puzzles: Sequence) -> str:
    rows = [
        (
            puzzle.layout_hash,
            puzzle.seed,
            puzzle.optimal_length,
            puzzle.robots_moved,
            puzzle.actions,
        )
        for puzzle in puzzles
    ]
    return _sha256_json(rows)


def _suite_provenance(
    name: str,
    puzzles: Sequence,
    *,
    seed: Optional[int],
) -> Dict[str, object]:
    """Stable identity for one concrete evaluation suite."""
    return {
        "name": name,
        "seed": seed,
        "fingerprint": _puzzle_fingerprint(puzzles),
        "puzzles": len(puzzles),
    }


def _load_model_checkpoint(net: DRCNet, path: str | Path) -> Dict[str, object]:
    """Load the existing schema-2 model format into ``net`` for evaluation."""
    checkpoint_path = Path(path)
    payload = torch.load(
        checkpoint_path,
        map_location=next(net.parameters()).device,
        weights_only=False,
    )
    if payload.get("kind") != "drc_model" or int(payload.get("schema_version", -1)) != 2:
        raise ValueError(f"Not a supported schema-2 DRC model checkpoint: {checkpoint_path}")
    net.load_state_dict(payload["state_dict"])
    return payload


def _model_checkpoint_source(
    path: str | Path,
    payload: Dict[str, object],
    *,
    source_timesteps: Optional[int] = None,
    provenance_reconstructed: bool = False,
) -> Dict[str, object]:
    """Describe the exact model checkpoint used by an evaluation."""
    checkpoint_path = Path(path)
    metadata = payload.get("metadata", {})
    stored_config = payload.get("config", {})
    timestep = source_timesteps
    if timestep is None and isinstance(metadata, dict):
        timestep = metadata.get("timesteps")
    checkpoint_kind = metadata.get("kind") if isinstance(metadata, dict) else None
    originating_run = (
        stored_config.get("run_name") if isinstance(stored_config, dict) else None
    ) or checkpoint_path.parent.name
    source = {
        "source_type": "model_checkpoint",
        "originating_run_name": originating_run,
        "source_timesteps": None if timestep is None else int(timestep),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_kind": checkpoint_kind or checkpoint_path.stem,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
    }
    if provenance_reconstructed:
        source["provenance_reconstructed"] = True
    return source


def _state_dicts_equal(
    first: Dict[str, torch.Tensor], second: Dict[str, torch.Tensor]
) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[key].detach().cpu(), second[key].detach().cpu())
        for key in first
    )


def _reconstruct_legacy_best_source(
    resume_path: str | Path,
    resume_payload: Dict[str, object],
    best_state_dict: Dict[str, torch.Tensor],
    best_timesteps: Optional[int],
) -> Dict[str, object]:
    """Recover provenance omitted by older outer-schema-1 checkpoints.

    Prefer a sibling schema-2 ``model_best.pt`` only after proving its tensors
    equal the embedded best state. Otherwise the training checkpoint itself is
    the exact, hash-addressed container from which the evaluated state came.
    """
    resume_path = Path(resume_path)
    candidate = resume_path.parent / "model_best.pt"
    if candidate.is_file():
        try:
            candidate_payload = torch.load(candidate, map_location="cpu", weights_only=False)
            candidate_metadata = candidate_payload.get("metadata", {})
            candidate_timestep = (
                candidate_metadata.get("timesteps")
                if isinstance(candidate_metadata, dict)
                else None
            )
            timestep_matches = (
                best_timesteps is None
                or candidate_timestep is None
                or int(candidate_timestep) == int(best_timesteps)
            )
            if (
                candidate_payload.get("kind") == "drc_model"
                and int(candidate_payload.get("schema_version", -1)) == 2
                and timestep_matches
                and _state_dicts_equal(candidate_payload["state_dict"], best_state_dict)
            ):
                return _model_checkpoint_source(
                    candidate,
                    candidate_payload,
                    source_timesteps=best_timesteps,
                    provenance_reconstructed=True,
                )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            pass

    stored_config = resume_payload.get("config", {})
    originating_run = (
        stored_config.get("run_name") if isinstance(stored_config, dict) else None
    ) or resume_path.parent.name
    return {
        "source_type": "model_checkpoint",
        "originating_run_name": originating_run,
        "source_timesteps": None if best_timesteps is None else int(best_timesteps),
        "checkpoint_path": str(resume_path.resolve()),
        "checkpoint_kind": "training_state_embedded_best",
        "checkpoint_sha256": _file_sha256(resume_path),
        "state_location": "run_state.best_state_dict",
        "provenance_reconstructed": True,
    }


def _reconstruct_legacy_best_selection_provenance(
    resume_payload: Dict[str, object],
    best_source: Dict[str, object],
) -> Dict[str, object]:
    """Recover the periodic suite identity omitted by legacy run state."""
    metadata = resume_payload.get("metadata", {})
    stored_suites = metadata.get("evaluation_suites", {}) if isinstance(metadata, dict) else {}
    stored_suite = (
        stored_suites.get("validation_periodic")
        if isinstance(stored_suites, dict)
        else None
    )
    if isinstance(stored_suite, dict):
        suite = dict(stored_suite)
    else:
        split_fingerprints = (
            metadata.get("split_fingerprints", {}) if isinstance(metadata, dict) else {}
        )
        fingerprint = (
            split_fingerprints.get("validation_periodic_suite")
            if isinstance(split_fingerprints, dict)
            else None
        )
        stored_config = resume_payload.get("config", {})
        overfit_n = stored_config.get("overfit_n") if isinstance(stored_config, dict) else None
        bank_path = stored_config.get("bank_path") if isinstance(stored_config, dict) else None
        suite = {
            "name": "validation_periodic",
            "seed": 7 if overfit_n is None and bank_path else None,
            "fingerprint": fingerprint,
            "puzzles": None,
        }
    return {
        "source": dict(best_source),
        "suite": suite,
        "evaluation_seed": 0,
        "episodes_per_puzzle": 1,
        "provenance_reconstructed": True,
    }


def _with_evaluation_provenance(
    summary: Dict[str, object],
    *,
    source: Dict[str, object],
    suite: Dict[str, object],
    evaluation_seed: int,
    episodes_per_puzzle: int,
) -> Dict[str, object]:
    result = dict(summary)
    result["provenance"] = {
        "source": dict(source),
        "suite": dict(suite),
        "evaluation_seed": int(evaluation_seed),
        "episodes_per_puzzle": int(episodes_per_puzzle),
    }
    return result


def _evaluate_model_with_provenance(
    net: DRCNet,
    puzzles: Sequence,
    env_config: EnvConfig,
    *,
    source: Dict[str, object],
    suite: Dict[str, object],
    episodes_per_puzzle: int,
    deterministic: bool,
    batch_size: int,
    seed: int = 0,
) -> Dict[str, object]:
    summary = evaluate_drc(
        net,
        puzzles,
        env_config,
        episodes_per_puzzle=episodes_per_puzzle,
        deterministic=deterministic,
        batch_size=batch_size,
        seed=seed,
    )
    return _with_evaluation_provenance(
        summary,
        source=source,
        suite=suite,
        evaluation_seed=seed,
        episodes_per_puzzle=episodes_per_puzzle,
    )


def _evaluate_scripted_with_provenance(
    puzzles: Sequence,
    env_config: EnvConfig,
    *,
    policy: str,
    suite: Dict[str, object],
    episodes_per_puzzle: int,
    seed: int = 0,
) -> Dict[str, object]:
    summary = scripted_baseline(
        puzzles,
        env_config,
        policy=policy,
        episodes_per_puzzle=episodes_per_puzzle,
        seed=seed,
    )
    return _with_evaluation_provenance(
        summary,
        source={"source_type": "scripted_policy", "policy": policy, "seed": int(seed)},
        suite=suite,
        evaluation_seed=seed,
        episodes_per_puzzle=episodes_per_puzzle,
    )


def _git_provenance() -> Dict[str, object]:
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _atomic_torch_save(payload: Dict[str, object], path: Path, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_model(
    net: DRCNet,
    config: DRCTrainConfig,
    path: Path,
    *,
    metadata: Optional[Dict[str, object]] = None,
    immutable: bool = False,
) -> None:
    _atomic_torch_save(
        {
            "kind": "drc_model",
            "schema_version": 2,
            "state_dict": net.state_dict(),
            "config": dataclasses.asdict(config),
            "config_fingerprint": _sha256_json(dataclasses.asdict(config)),
            "observation_shape": tuple(net.observation_shape),
            "n_actions": int(net.n_actions),
            "metadata": metadata or {},
        },
        path,
        immutable=immutable,
    )


def _save_training_checkpoint(
    algo: RecurrentPPO,
    sampler: object,
    config: DRCTrainConfig,
    path: Path,
    *,
    run_state: Dict[str, object],
    metadata: Dict[str, object],
    immutable: bool = False,
) -> None:
    if not hasattr(sampler, "state_dict"):
        raise TypeError(f"Sampler {type(sampler).__name__} does not support exact resume")
    _atomic_torch_save(
        {
            "kind": "drc_training_state",
            "schema_version": 1,
            "config": dataclasses.asdict(config),
            "config_contract": _config_contract(config),
            "config_contract_fingerprint": _sha256_json(_config_contract(config)),
            "algo_state": algo.state_dict(),
            "sampler_state": sampler.state_dict(),
            "run_state": run_state,
            "metadata": metadata,
        },
        path,
        immutable=immutable,
    )


def _load_training_checkpoint(
    path: str | Path,
    *,
    algo: RecurrentPPO,
    sampler: object,
    config: DRCTrainConfig,
    allow_device_change: bool = False,
) -> Dict[str, object]:
    payload = torch.load(path, map_location=algo.device, weights_only=False)
    if payload.get("kind") != "drc_training_state" or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Not a supported DRC training-state checkpoint: {path}")
    expected_contract = _config_contract(config)
    changed = _contract_differences(payload.get("config_contract", {}), expected_contract)
    declared = set(config.resume_allow_contract_change or ())
    undeclared = [key for key in changed if key not in declared]
    if undeclared:
        raise ValueError(
            f"Resume config contract changed in fields: {undeclared}. If this is a "
            "deliberate experiment branch rather than a resume, declare them in "
            "resume_allow_contract_change."
        )
    if changed:
        print(
            "[drc] BRANCH, not an exact continuation: deliberately changed contract "
            f"fields {sorted(changed)} relative to {path}. Curves before and after "
            "this point are NOT the same experiment.",
            flush=True,
        )
    if not hasattr(sampler, "load_state_dict"):
        raise TypeError(f"Sampler {type(sampler).__name__} does not support exact resume")
    sampler.load_state_dict(payload["sampler_state"])
    algo.load_state_dict(payload["algo_state"], allow_device_change=allow_device_change)
    # Runtime-only information for the caller. Do not persist it into the
    # checkpoint schema: outer schema 1 remains readable by older tooling.
    loaded = dict(payload)
    loaded["_resume_contract_changes"] = tuple(sorted(changed))
    return loaded


# ------------------------------------------------------------------ entrypoint
def build_and_train(config: DRCTrainConfig, *, puzzles_override: Optional[Sequence] = None) -> dict:
    """Train a DRCNet with recurrent PPO; mirrors train.py's run layout.

    ``puzzles_override`` is a test hook: a puzzle list used directly for
    training AND evaluation (like overfit mode), bypassing the bank.
    """
    from .puzzles import joint_stratified_subset, load_puzzles, stratified_subset

    periodic_suite_seed = 7
    final_suite_seed = 8
    hard_suite_seed = 9

    schedule_total_steps = (
        config.total_steps if config.schedule_total_steps is None else config.schedule_total_steps
    )
    if schedule_total_steps < config.total_steps:
        raise ValueError(
            f"schedule_total_steps ({schedule_total_steps}) must be >= "
            f"total_steps ({config.total_steps})"
        )
    if config.hard_eval_per_stratum < 0:
        raise ValueError(
            "hard_eval_per_stratum must be non-negative, got "
            f"{config.hard_eval_per_stratum}"
        )
    out = Path(config.out_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "curve.jsonl"
    if config.resume_from is None and curve_path.exists():
        raise FileExistsError(
            f"Run already has a curve; use resume_from or a new run_name: {curve_path}"
        )
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))
    wandb_run = _maybe_init_wandb(config, out)

    env_config = EnvConfig(
        horizon=config.horizon,
        step_penalty=config.step_penalty,
        goal_reward=config.goal_reward,
        shaping_coef=config.shaping_coef,
        gamma=config.gamma,  # shaping gamma == PPO gamma (checked again below)
        time_channel=config.time_channel,
        canonical_identity=config.canonical_identity,
        stop_planes=config.stop_planes,
        distance_plane=config.distance_plane,
    )

    if puzzles_override is not None:
        train_puzzles = list(puzzles_override)
        val_puzzles = list(train_puzzles)
        val_suite = list(train_puzzles)
        final_suite = list(train_puzzles)
        periodic_selection_seed: Optional[int] = None
        final_selection_seed: Optional[int] = None
    else:
        split_kwargs = dict(
            eval_fraction=config.eval_fraction,
            split_seed=config.split_seed,
            val_test_seed=config.val_test_seed,
        )
        train_puzzles = load_puzzles(config.bank_path, "train", **split_kwargs)
        if config.overfit_n is not None:
            train_puzzles = stratified_subset(
                train_puzzles, per_length=max(1, config.overfit_n // 4), seed=config.seed
            )[: config.overfit_n]
            val_puzzles = list(train_puzzles)
            val_suite = list(train_puzzles)
            final_suite = list(train_puzzles)
            periodic_selection_seed = None
            final_selection_seed = None
        else:
            val_puzzles = load_puzzles(config.bank_path, "val", **split_kwargs)
            val_suite = stratified_subset(
                val_puzzles, config.eval_per_length, seed=periodic_suite_seed
            )
            final_suite = stratified_subset(
                val_puzzles, config.final_eval_per_length, seed=final_suite_seed
            )
            periodic_selection_seed = periodic_suite_seed
            final_selection_seed = final_suite_seed
    hard_suite = joint_stratified_subset(
        val_puzzles,
        config.hard_eval_per_stratum,
        seed=hard_suite_seed,
    )
    evaluation_suites = {
        "validation_periodic": _suite_provenance(
            "validation_periodic",
            val_suite,
            seed=periodic_selection_seed,
        ),
        "validation_final": _suite_provenance(
            "validation_final",
            final_suite,
            seed=final_selection_seed,
        ),
        "validation_hard": _suite_provenance(
            "validation_hard_joint_strata",
            hard_suite,
            seed=hard_suite_seed,
        ),
    }
    print(
        f"[drc] train={len(train_puzzles)} puzzles, val_suite={len(val_suite)}, "
        f"final_suite={len(final_suite)}, hard_suite={len(hard_suite)}",
        flush=True,
    )

    if config.curriculum == "reverse":
        sampler = ReverseFrontierSampler(
            train_puzzles,
            tau=config.tau,
            ema_beta=config.ema_beta,
            min_episodes=config.min_episodes,
            frontier_p=config.frontier_p,
            retention_p=config.retention_p,
            uniform_p=config.uniform_p,
        )
    elif config.curriculum == "uniform":
        sampler = UniformSampler(train_puzzles)
    else:
        raise ValueError(f"Unknown curriculum: {config.curriculum}")

    torch.manual_seed(config.seed)
    envs = [
        RicochetEnv(train_puzzles, env_config, sampler=sampler, seed=config.seed * 1000 + r)
        for r in range(config.num_envs)
    ]
    device = _resolve_device(config.device)
    net = build_net(config, envs[0].observation_space.shape, envs[0].action_space.n)
    algo = RecurrentPPO(
        net,
        envs,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        learning_rate_final=config.learning_rate_final,
        target_kl=config.target_kl,
        adv_std_floor=config.adv_std_floor,
        adv_clip=config.adv_clip,
        clip_range_vf=config.clip_range_vf,
        bootstrap_on_truncation=config.bootstrap_on_truncation,
        aux_landing_coef=config.aux_landing_coef,
        device=device,
        seed=config.seed,
    )
    n_params = sum(p.numel() for p in net.parameters())
    provenance = {
        "git": _git_provenance(),
        "config_fingerprint": _sha256_json(dataclasses.asdict(config)),
        "config_contract_fingerprint": _sha256_json(_config_contract(config)),
        "bank_sha256": (
            _file_sha256(config.bank_path)
            if puzzles_override is None and Path(config.bank_path).is_file()
            else None
        ),
        "split_fingerprints": {
            "train": _puzzle_fingerprint(train_puzzles),
            "validation_pool": _puzzle_fingerprint(val_puzzles),
            "validation_periodic_suite": _puzzle_fingerprint(val_suite),
            "validation_final_suite": _puzzle_fingerprint(final_suite),
            "validation_hard_suite": _puzzle_fingerprint(hard_suite),
        },
        "evaluation_suites": evaluation_suites,
        "observation_shape": tuple(net.observation_shape),
        "n_actions": int(net.n_actions),
        "policy_params": int(n_params),
        # Derived from time_channel rather than a config field of its own, so the
        # contract fingerprint is unchanged and existing checkpoints stay
        # resumable. That means the fingerprint alone does NOT distinguish runs
        # made before and after the truncation-bootstrap fix -- record it here so
        # run_history.jsonl and every snapshot carry the semantics explicitly.
        "bootstrap_on_truncation": bool(algo.bootstrap_on_truncation),
    }
    print(
        f"[drc] device={device} DRC({config.drc_depth},{config.drc_ticks}) "
        f"policy_params={n_params:,} "
        f"bootstrap_on_truncation={algo.bootstrap_on_truncation}",
        flush=True,
    )

    start_time = time.time()
    elapsed_before = 0.0
    next_eval = config.eval_every
    next_checkpoint = config.checkpoint_every
    best_val_det: Optional[float] = None
    best_timesteps: Optional[int] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_source: Optional[Dict[str, object]] = None
    best_selection_provenance: Optional[Dict[str, object]] = None
    resume_contract_changes: Tuple[str, ...] = ()

    if config.resume_from is not None:
        resume_payload = _load_training_checkpoint(
            config.resume_from,
            algo=algo,
            sampler=sampler,
            config=config,
        )
        run_state = resume_payload.get("run_state", {})
        resume_contract_changes = tuple(resume_payload.get("_resume_contract_changes", ()))
        elapsed_before = float(run_state.get("elapsed_s", 0.0))
        if not resume_contract_changes:
            best_val_det = run_state.get("best_val_det")
            best_timesteps = run_state.get("best_timesteps")
            best_state_dict = run_state.get("best_state_dict")
            best_source = run_state.get("best_source")
            best_selection_provenance = run_state.get("best_selection_provenance")
            if best_state_dict is not None and best_source is None:
                best_source = _reconstruct_legacy_best_source(
                    config.resume_from,
                    resume_payload,
                    best_state_dict,
                    best_timesteps,
                )
            if (
                best_state_dict is not None
                and best_source is not None
                and best_selection_provenance is None
            ):
                best_selection_provenance = (
                    _reconstruct_legacy_best_selection_provenance(
                        resume_payload,
                        best_source,
                    )
                )
        # A declared semantic branch starts a new experiment's selection
        # process. It resumes optimizer/environment state, but not its parent's
        # notion of which weights are best.
        else:
            best_val_det = None
            best_timesteps = None
            best_state_dict = None
            best_source = None
            best_selection_provenance = None
        next_eval = int(run_state.get("next_eval", 0))
        if next_eval <= algo.num_timesteps:
            next_eval = (algo.num_timesteps // config.eval_every + 1) * config.eval_every
        saved_next_checkpoint = run_state.get("next_checkpoint")
        if config.checkpoint_every is None:
            next_checkpoint = None
        elif saved_next_checkpoint is None or int(saved_next_checkpoint) <= algo.num_timesteps:
            next_checkpoint = (
                algo.num_timesteps // config.checkpoint_every + 1
            ) * config.checkpoint_every
        else:
            next_checkpoint = int(saved_next_checkpoint)
        continuation_kind = "branch training state" if resume_contract_changes else "exact training state"
        print(
            f"[drc] resumed {continuation_kind} from {config.resume_from} "
            f"at t={algo.num_timesteps:,}",
            flush=True,
        )
    if algo.num_timesteps >= config.total_steps:
        raise ValueError(
            f"Nothing to train: checkpoint t={algo.num_timesteps} >= "
            f"target total_steps={config.total_steps}"
        )

    history_entry = {
        "started_at_unix": time.time(),
        "resume_from": config.resume_from,
        "start_timesteps": int(algo.num_timesteps),
        "target_total_steps": int(config.total_steps),
        "schedule_total_steps": int(schedule_total_steps),
        "resume_contract_changes": list(resume_contract_changes),
        **provenance,
    }
    with (out / "run_history.jsonl").open("a") as handle:
        handle.write(json.dumps(history_entry) + "\n")

    def elapsed_total() -> float:
        return elapsed_before + (time.time() - start_time)

    def snapshot_metadata(kind: str) -> Dict[str, object]:
        elapsed = elapsed_total()
        return {
            "kind": kind,
            "timesteps": int(algo.num_timesteps),
            "elapsed_s": round(elapsed, 3),
            "steps_per_s": round(algo.num_timesteps / max(elapsed, 1e-9), 3),
            **provenance,
        }

    def current_run_state() -> Dict[str, object]:
        return {
            "elapsed_s": elapsed_total(),
            "best_val_det": best_val_det,
            "best_timesteps": best_timesteps,
            "best_state_dict": best_state_dict,
            "best_source": best_source,
            "best_selection_provenance": best_selection_provenance,
            "next_eval": next_eval,
            "next_checkpoint": next_checkpoint,
        }

    def save_training_latest() -> None:
        _save_training_checkpoint(
            algo,
            sampler,
            config,
            out / "training_latest.pt",
            run_state=current_run_state(),
            metadata=snapshot_metadata("training_latest"),
        )

    def on_iteration(algo: RecurrentPPO, metrics: Dict[str, object]) -> None:
        nonlocal next_eval, next_checkpoint, best_val_det, best_timesteps
        nonlocal best_state_dict, best_source, best_selection_provenance
        saved_something = False
        if algo.num_timesteps >= next_eval:
            while next_eval <= algo.num_timesteps:
                next_eval += config.eval_every
            latest_path = out / "model_latest.pt"
            _save_model(
                net,
                config,
                latest_path,
                metadata=snapshot_metadata("model_latest"),
            )
            latest_payload = _load_model_checkpoint(net, latest_path)
            latest_source = _model_checkpoint_source(latest_path, latest_payload)
            det = _evaluate_model_with_provenance(
                net,
                val_suite,
                env_config,
                source=latest_source,
                suite=evaluation_suites["validation_periodic"],
                episodes_per_puzzle=1,
                deterministic=True,
                batch_size=config.eval_batch_size,
            )
            if best_val_det is None or det["success_rate"] > best_val_det:
                best_val_det = float(det["success_rate"])
                best_timesteps = int(algo.num_timesteps)
                best_state_dict = {
                    key: value.detach().cpu().clone() for key, value in net.state_dict().items()
                }
                best_selection_provenance = dict(det["provenance"])
                _save_model(
                    net,
                    config,
                    out / "model_best.pt",
                    metadata=snapshot_metadata("model_best"),
                )
                best_payload = torch.load(
                    out / "model_best.pt", map_location="cpu", weights_only=False
                )
                best_source = _model_checkpoint_source(
                    out / "model_best.pt", best_payload
                )
                best_selection_provenance["source"] = dict(best_source)
            entry = {
                "timesteps": int(algo.num_timesteps),
                "elapsed_s": round(elapsed_total(), 1),
                "steps_per_s": round(
                    algo.num_timesteps / max(1e-9, elapsed_total()), 1
                ),
                "val_det": det,
                "best_val_det": best_val_det,
                "best_timesteps": best_timesteps,
                "best_source": best_source,
                "best_selection_provenance": best_selection_provenance,
                "sampler": sampler.stats(),
                "losses": metrics,
            }
            with curve_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            if wandb_run is not None:
                payload = {
                    "train/timesteps": int(algo.num_timesteps),
                    "train/steps_per_s": entry["steps_per_s"],
                    **{
                        f"train/{key}": value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                    **_flatten_eval("val_det", det),
                    **_flatten_sampler(sampler.stats()),
                }
                wandb_run.log(payload, step=int(algo.num_timesteps))
            print(
                f"[drc] t={entry['timesteps']:>9,} ({entry['steps_per_s']:.0f} st/s) "
                f"val_det={det['success_rate']:.3f} "
                f"by_len={ {k: v['success_rate'] for k, v in det['by_length'].items()} } "
                f"sampler={sampler.stats().get('frontier', '-')}",
                flush=True,
            )
            saved_something = True

        if next_checkpoint is not None and algo.num_timesteps >= next_checkpoint:
            while next_checkpoint <= algo.num_timesteps:
                next_checkpoint += config.checkpoint_every
            actual = int(algo.num_timesteps)
            metadata = snapshot_metadata("periodic")
            _save_model(
                net,
                config,
                out / f"model_{actual}.pt",
                metadata=metadata,
                immutable=True,
            )
            _save_training_checkpoint(
                algo,
                sampler,
                config,
                out / f"training_{actual}.pt",
                run_state=current_run_state(),
                metadata=metadata,
                immutable=True,
            )
            with (out / "snapshot_manifest.jsonl").open("a") as handle:
                handle.write(json.dumps(metadata) + "\n")
            saved_something = True

        if saved_something:
            save_training_latest()

    algo.learn(
        config.total_steps,
        on_iteration,
        schedule_total_steps=schedule_total_steps,
    )
    final_model_path = out / "model_final.pt"
    _save_model(
        net,
        config,
        final_model_path,
        metadata=snapshot_metadata("model_final"),
    )
    # Headline evaluation is checkpoint-backed by construction: reload the
    # just-written model instead of relying on whichever weights happen to be
    # resident in memory.
    final_model_payload = _load_model_checkpoint(net, final_model_path)
    final_source = _model_checkpoint_source(final_model_path, final_model_payload)
    _save_training_checkpoint(
        algo,
        sampler,
        config,
        out / "training_final.pt",
        run_state=current_run_state(),
        metadata=snapshot_metadata("training_final"),
    )
    save_training_latest()

    final_val_det = _evaluate_model_with_provenance(
        net,
        final_suite,
        env_config,
        source=final_source,
        suite=evaluation_suites["validation_final"],
        episodes_per_puzzle=1,
        deterministic=True,
        batch_size=config.eval_batch_size,
    )
    final_val_stoch = _evaluate_model_with_provenance(
        net,
        final_suite,
        env_config,
        source=final_source,
        suite=evaluation_suites["validation_final"],
        episodes_per_puzzle=config.final_eval_episodes,
        deterministic=False,
        batch_size=config.eval_batch_size,
    )
    final_hard_val_det = _evaluate_model_with_provenance(
        net,
        hard_suite,
        env_config,
        source=final_source,
        suite=evaluation_suites["validation_hard"],
        episodes_per_puzzle=1,
        deterministic=True,
        batch_size=config.eval_batch_size,
    )

    best_periodic_val_det: Optional[Dict[str, object]] = None
    best_periodic_val_stoch: Optional[Dict[str, object]] = None
    best_periodic_hard_val_det: Optional[Dict[str, object]] = None
    if best_state_dict is not None:
        if best_source is None:
            raise RuntimeError("Best weights exist without checkpoint provenance")
        net.load_state_dict(best_state_dict)
        best_periodic_val_det = _evaluate_model_with_provenance(
            net,
            final_suite,
            env_config,
            source=best_source,
            suite=evaluation_suites["validation_final"],
            episodes_per_puzzle=1,
            deterministic=True,
            batch_size=config.eval_batch_size,
        )
        best_periodic_val_stoch = _evaluate_model_with_provenance(
            net,
            final_suite,
            env_config,
            source=best_source,
            suite=evaluation_suites["validation_final"],
            episodes_per_puzzle=config.final_eval_episodes,
            deterministic=False,
            batch_size=config.eval_batch_size,
        )
        best_periodic_hard_val_det = _evaluate_model_with_provenance(
            net,
            hard_suite,
            env_config,
            source=best_source,
            suite=evaluation_suites["validation_hard"],
            episodes_per_puzzle=1,
            deterministic=True,
            batch_size=config.eval_batch_size,
        )
        # Do not leave a completed run's in-memory network pointing at a
        # secondary comparison checkpoint.
        _load_model_checkpoint(net, final_model_path)

    report = {
        "config": dataclasses.asdict(config),
        "policy_params": int(n_params),
        "elapsed_s": round(elapsed_total(), 1),
        "actual_timesteps": int(algo.num_timesteps),
        "schedule_total_steps": int(schedule_total_steps),
        "provenance": provenance,
        "best_val_det": best_val_det,
        "best_timesteps": best_timesteps,
        "best_source": best_source,
        "best_selection_provenance": best_selection_provenance,
        # Backwards-compatible explicit primary source. This is now invariant:
        # final_* always means model_final.pt.
        "report_eval_weights": "final",
        "final_val_det": final_val_det,
        "final_val_stoch": final_val_stoch,
        "final_hard_val_det": final_hard_val_det,
        "best_periodic_val_det": best_periodic_val_det,
        "best_periodic_val_stoch": best_periodic_val_stoch,
        "best_periodic_hard_val_det": best_periodic_hard_val_det,
        "baseline_random": _evaluate_scripted_with_provenance(
            final_suite,
            env_config,
            policy="random",
            suite=evaluation_suites["validation_final"],
            episodes_per_puzzle=5,
        ),
        "baseline_target_only": _evaluate_scripted_with_provenance(
            final_suite,
            env_config,
            policy="target_only",
            suite=evaluation_suites["validation_final"],
            episodes_per_puzzle=5,
        ),
        "sampler_final": sampler.stats(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    report_at_steps = out / f"report_{algo.num_timesteps}.json"
    if not report_at_steps.exists():
        report_at_steps.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "final_val_det",
                    "final_val_stoch",
                    "final_hard_val_det",
                    "best_periodic_val_det",
                    "best_periodic_val_stoch",
                    "best_periodic_hard_val_det",
                    "baseline_random",
                    "baseline_target_only",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    if wandb_run is not None:
        final_wandb = {
            "train/timesteps": int(algo.num_timesteps),
            **_flatten_eval("final/val_det", report["final_val_det"]),
            **_flatten_eval("final/val_stoch", report["final_val_stoch"]),
            **_flatten_eval("final/hard_val_det", report["final_hard_val_det"]),
            **_flatten_eval("final/baseline_random", report["baseline_random"]),
            **_flatten_eval("final/baseline_target_only", report["baseline_target_only"]),
        }
        if best_periodic_val_det is not None:
            final_wandb.update(
                {
                    **_flatten_eval("best_periodic/val_det", best_periodic_val_det),
                    **_flatten_eval("best_periodic/val_stoch", best_periodic_val_stoch),
                    **_flatten_eval(
                        "best_periodic/hard_val_det", best_periodic_hard_val_det
                    ),
                }
            )
        wandb_run.log(final_wandb, step=int(algo.num_timesteps))
        wandb_run.finish()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()
    config = DRCTrainConfig.from_yaml(args.config)
    if args.set:
        values = dataclasses.asdict(config)
        for item in args.set:
            key, _, raw = item.partition("=")
            if key not in values:
                raise ValueError(f"Unknown override key: {key}")
            values[key] = yaml.safe_load(raw)
        config = DRCTrainConfig(**values)
    build_and_train(config)


if __name__ == "__main__":
    main()
