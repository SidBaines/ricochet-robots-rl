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
- On TimeLimit truncation the value of the final observation -- evaluated
  under the post-step recurrent state -- is discounted into the step reward,
  the same semantics SB3 implements via ``TimeLimit.truncated`` infos.
- Shaping consistency: every env's ``EnvConfig.gamma`` must equal the PPO
  discount (checked at construction), otherwise the potential-based shaping
  is not policy-invariant.

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
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-4
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
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
    eval_batch_size: int = 32
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


def replay_sequences(
    net: DRCNet,
    obs: torch.Tensor,
    masks: torch.Tensor,
    episode_starts: torch.Tensor,
    initial_state: DRCState,
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
        (logits (T, B, A), values (T, B)).
    """
    state = initial_state
    logits_seq: List[torch.Tensor] = []
    values_seq: List[torch.Tensor] = []
    for t in range(obs.shape[0]):
        state = DRCNet.reset_state(state, episode_starts[t])
        logits, values, state = net(obs[t], state, action_mask=masks[t])
        logits_seq.append(logits)
        values_seq.append(values)
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

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self._gen = torch.Generator().manual_seed(seed)  # minibatch shuffling
        self.num_timesteps = 0

        obs0 = []
        for env in self.envs:
            obs, _info = env.reset()
            obs0.append(obs)
        self._obs = np.stack(obs0).astype(np.float32)
        self._episode_start = np.ones(self.num_envs, dtype=bool)
        self._state = self.net.initial_state(self.num_envs, device=self.device)

    # -------------------------------------------------------------- collection
    def collect_rollout(self) -> RolloutBuffer:
        """Collect n_steps transitions from all envs (one forward per vec step)."""
        net = self.net
        buf = RolloutBuffer(self.n_steps, self.num_envs, self.obs_shape, self.n_actions, self.device)
        buf.initial_state = DRCState(
            h=self._state.h.detach().clone(), c=self._state.c.detach().clone()
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
                        if truncated and not terminated:
                            # Bootstrap the value of the final observation
                            # (SB3's TimeLimit.truncated semantics).
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
        n_grad_updates = 0
        continue_training = True

        for _epoch in range(self.n_epochs):
            perm = torch.randperm(self.num_envs, generator=self._gen).tolist()
            for start in range(0, self.num_envs, seqs_per_minibatch):
                idx = perm[start : start + seqs_per_minibatch]
                init = DRCState(
                    h=buf.initial_state.h[:, idx], c=buf.initial_state.c[:, idx]
                )
                logits, values = replay_sequences(
                    self.net, buf.obs[:, idx], buf.masks[:, idx], buf.episode_starts[:, idx], init
                )
                dist = Categorical(logits=logits.reshape(-1, self.n_actions))
                actions = buf.actions[:, idx].reshape(-1)
                new_log_prob = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                values = values.reshape(-1)

                advantages = buf.advantages[:, idx].reshape(-1)
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                log_ratio = new_log_prob - buf.log_probs[:, idx].reshape(-1)
                ratio = torch.exp(log_ratio)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range),
                ).mean()
                value_loss = F.mse_loss(buf.returns[:, idx].reshape(-1), values)
                loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
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
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": explained_variance,
            "n_grad_updates": n_grad_updates,
            "early_stop": not continue_training,
        }

    # ------------------------------------------------------------------- learn
    def learn(
        self,
        total_steps: int,
        on_iteration: Optional[Callable[["RecurrentPPO", Dict[str, object]], None]] = None,
    ) -> None:
        while self.num_timesteps < total_steps:
            buf = self.collect_rollout()
            progress_remaining = max(0.0, 1.0 - self.num_timesteps / total_steps)
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
    """
    device = next(net.parameters()).device
    torch.manual_seed(seed)
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
                actions = Categorical(logits=logits).sample()
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
    for pattern in ("train/*", "curriculum/*", "val_det/*", "final/*"):
        run.define_metric(pattern, step_metric="train/timesteps")
    return run


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _save_model(net: DRCNet, config: DRCTrainConfig, path: Path) -> None:
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": dataclasses.asdict(config),
            "observation_shape": tuple(net.observation_shape),
            "n_actions": int(net.n_actions),
        },
        path,
    )


# ------------------------------------------------------------------ entrypoint
def build_and_train(config: DRCTrainConfig, *, puzzles_override: Optional[Sequence] = None) -> dict:
    """Train a DRCNet with recurrent PPO; mirrors train.py's run layout.

    ``puzzles_override`` is a test hook: a puzzle list used directly for
    training AND evaluation (like overfit mode), bypassing the bank.
    """
    from .puzzles import load_puzzles, stratified_subset

    out = Path(config.out_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
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
        val_suite = list(train_puzzles)
        final_suite = list(train_puzzles)
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
            val_suite = list(train_puzzles)
            final_suite = list(train_puzzles)
        else:
            val_puzzles = load_puzzles(config.bank_path, "val", **split_kwargs)
            val_suite = stratified_subset(val_puzzles, config.eval_per_length, seed=7)
            final_suite = stratified_subset(val_puzzles, config.final_eval_per_length, seed=8)
    print(
        f"[drc] train={len(train_puzzles)} puzzles, val_suite={len(val_suite)}, "
        f"final_suite={len(final_suite)}",
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
        device=device,
        seed=config.seed,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"[drc] device={device} DRC({config.drc_depth},{config.drc_ticks}) "
        f"policy_params={n_params:,}",
        flush=True,
    )

    curve_path = out / "curve.jsonl"
    start_time = time.time()
    next_eval = config.eval_every

    def on_iteration(algo: RecurrentPPO, metrics: Dict[str, object]) -> None:
        nonlocal next_eval
        if algo.num_timesteps < next_eval:
            return
        next_eval += config.eval_every
        det = evaluate_drc(
            net,
            val_suite,
            env_config,
            episodes_per_puzzle=1,
            deterministic=True,
            batch_size=config.eval_batch_size,
        )
        entry = {
            "timesteps": int(algo.num_timesteps),
            "elapsed_s": round(time.time() - start_time, 1),
            "steps_per_s": round(algo.num_timesteps / max(1e-9, time.time() - start_time), 1),
            "val_det": det,
            "sampler": sampler.stats(),
            "losses": metrics,
        }
        with curve_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        if wandb_run is not None:
            payload = {
                "train/timesteps": int(algo.num_timesteps),
                "train/steps_per_s": entry["steps_per_s"],
                **{f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
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
        _save_model(net, config, out / "model_latest.pt")

    algo.learn(config.total_steps, on_iteration)
    _save_model(net, config, out / "model_final.pt")

    report = {
        "config": dataclasses.asdict(config),
        "policy_params": int(n_params),
        "elapsed_s": round(time.time() - start_time, 1),
        "final_val_det": evaluate_drc(
            net,
            final_suite,
            env_config,
            episodes_per_puzzle=1,
            deterministic=True,
            batch_size=config.eval_batch_size,
        ),
        "final_val_stoch": evaluate_drc(
            net,
            final_suite,
            env_config,
            episodes_per_puzzle=config.final_eval_episodes,
            deterministic=False,
            batch_size=config.eval_batch_size,
        ),
        "baseline_random": scripted_baseline(
            final_suite, env_config, policy="random", episodes_per_puzzle=5
        ),
        "baseline_target_only": scripted_baseline(
            final_suite, env_config, policy="target_only", episodes_per_puzzle=5
        ),
        "sampler_final": sampler.stats(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "final_val_det",
                    "final_val_stoch",
                    "baseline_random",
                    "baseline_target_only",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/timesteps": int(config.total_steps),
                **_flatten_eval("final/val_det", report["final_val_det"]),
                **_flatten_eval("final/val_stoch", report["final_val_stoch"]),
                **_flatten_eval("final/baseline_random", report["baseline_random"]),
                **_flatten_eval("final/baseline_target_only", report["baseline_target_only"]),
            },
            step=int(config.total_steps),
        )
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
