"""Evaluation protocol: batched policy evaluation plus scripted baselines.

All evaluation (policy and baselines) runs masked and on full starts, so every
number in a report is produced under one identical protocol.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .env import EnvConfig, RicochetEnv
from .puzzles import Puzzle


def _stack_obs(obs_list: List) -> object:
    """Stack per-env observations into a batch; supports Dict observations."""
    if isinstance(obs_list[0], dict):
        return {key: np.stack([obs[key] for obs in obs_list]) for key in obs_list[0]}
    return np.stack(obs_list)


def _aggregate(records: List[dict]) -> Dict[str, object]:
    overall = float(np.mean([r["success"] for r in records])) if records else 0.0
    by_length: Dict[int, List[int]] = defaultdict(list)
    by_robots: Dict[int, List[int]] = defaultdict(list)
    for r in records:
        by_length[r["optimal_length"]].append(r["success"])
        by_robots[r["robots_moved"]].append(r["success"])
    return {
        "success_rate": round(overall, 4),
        "episodes": len(records),
        "by_length": {
            length: {"success_rate": round(float(np.mean(v)), 4), "n": len(v)}
            for length, v in sorted(by_length.items())
        },
        "by_robots_moved": {
            robots: {"success_rate": round(float(np.mean(v)), 4), "n": len(v)}
            for robots, v in sorted(by_robots.items())
        },
    }


def evaluate_policy(
    model,
    puzzles: Sequence[Puzzle],
    config: EnvConfig,
    *,
    episodes_per_puzzle: int = 1,
    deterministic: bool = True,
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    """Masked, full-start evaluation of a MaskablePPO model over a puzzle list."""
    import torch

    torch.manual_seed(seed)
    work = [
        (i, rep) for i in range(len(puzzles)) for rep in range(episodes_per_puzzle)
    ]
    work.reverse()  # pop() consumes in order
    n_slots = min(batch_size, len(work))
    envs = [RicochetEnv(puzzles, config, sampler=None, seed=seed + s) for s in range(n_slots)]
    slot_puzzle: List[Optional[int]] = [None] * n_slots
    obs_list: List[Optional[np.ndarray]] = [None] * n_slots
    records: List[dict] = []

    def load_next(slot: int) -> None:
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

    while any(p is not None for p in slot_puzzle):
        active = [s for s in range(n_slots) if slot_puzzle[s] is not None]
        obs_batch = _stack_obs([obs_list[s] for s in active])
        mask_batch = np.stack([envs[s].action_masks() for s in active])
        actions, _ = model.predict(
            obs_batch, action_masks=mask_batch, deterministic=deterministic
        )
        for k, s in enumerate(active):
            obs, _rew, term, trunc, info = envs[s].step(int(actions[k]))
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
    result = _aggregate(records)
    result["deterministic"] = deterministic
    return result


def scripted_baseline(
    puzzles: Sequence[Puzzle],
    config: EnvConfig,
    *,
    policy: str = "random",
    episodes_per_puzzle: int = 5,
    seed: int = 0,
) -> Dict[str, object]:
    """Masked scripted baselines under the identical protocol.

    - 'random': uniform over legal (moving) actions.
    - 'target_only': uniform over the target robot's legal actions, falling
      back to any legal action when the target robot is boxed in.
    """
    if policy not in {"random", "target_only"}:
        raise ValueError(policy)
    rng = np.random.default_rng(seed)
    env = RicochetEnv(puzzles, config, sampler=None, seed=seed)
    records: List[dict] = []
    for idx, puzzle in enumerate(puzzles):
        for _ in range(episodes_per_puzzle):
            env.reset(options={"puzzle_index": idx})
            while True:
                mask = env.action_masks()
                if policy == "target_only":
                    target_mask = np.zeros_like(mask)
                    target_mask[0:4] = mask[0:4]  # canonical: target is robot 0
                    mask = target_mask if target_mask.any() else mask
                legal = np.flatnonzero(mask)
                action = int(legal[rng.integers(len(legal))])
                _obs, _rew, term, trunc, info = env.step(action)
                if term or trunc:
                    records.append(
                        {
                            "success": bool(info["is_success"]),
                            "optimal_length": puzzle.optimal_length,
                            "robots_moved": puzzle.robots_moved,
                            "steps": info["steps"],
                        }
                    )
                    break
    result = _aggregate(records)
    result["policy"] = policy
    return result
