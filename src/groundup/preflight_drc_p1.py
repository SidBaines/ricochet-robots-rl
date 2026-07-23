"""Preflight verifier for the 12x12 helper-free DRC research run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .drc_train import DRCTrainConfig, _file_sha256, _puzzle_fingerprint, build_net
from .env import EnvConfig, RicochetEnv
from .puzzles import load_puzzles


def run_preflight(config: DRCTrainConfig, *, sample_size: int = 100) -> dict:
    split_kwargs = {
        "eval_fraction": config.eval_fraction,
        "split_seed": config.split_seed,
        "val_test_seed": config.val_test_seed,
    }
    splits = {
        name: load_puzzles(config.bank_path, name, **split_kwargs)
        for name in ("train", "val", "test")
    }
    hashes = {name: {puzzle.layout_hash for puzzle in puzzles} for name, puzzles in splits.items()}
    overlap = {
        "train_val": len(hashes["train"] & hashes["val"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "val_test": len(hashes["val"] & hashes["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"Layout split overlap: {overlap}")

    rng = np.random.default_rng(config.seed)
    all_puzzles = splits["train"] + splits["val"] + splits["test"]
    selected = rng.choice(len(all_puzzles), size=min(sample_size, len(all_puzzles)), replace=False)

    replay_config = EnvConfig(
        horizon=max(config.horizon, 32),
        shaping_coef=0.0,
        time_channel=False,
        canonical_identity=False,
    )
    replay_env = RicochetEnv(all_puzzles, replay_config, seed=config.seed)
    replayed = 0
    for index in selected:
        puzzle = all_puzzles[int(index)]
        replay_env.reset(options={"puzzle_index": int(index)})
        terminated = False
        info = {}
        for robot, direction in puzzle.actions:
            _obs, _reward, terminated, truncated, info = replay_env.step(robot * 4 + direction)
            if truncated:
                raise AssertionError(f"Stored plan truncated for puzzle {puzzle.seed}")
        if not terminated or not info.get("is_success"):
            raise AssertionError(f"Stored plan failed for puzzle {puzzle.seed}")
        replayed += 1

    env_config = EnvConfig(
        horizon=config.horizon,
        step_penalty=config.step_penalty,
        goal_reward=config.goal_reward,
        shaping_coef=config.shaping_coef,
        gamma=config.gamma,
        time_channel=config.time_channel,
        canonical_identity=config.canonical_identity,
        stop_planes=config.stop_planes,
        distance_plane=config.distance_plane,
    )
    if config.stop_planes or config.distance_plane:
        raise AssertionError("Primary DRC config must exclude stop and distance planes")
    env = RicochetEnv(splits["train"], env_config, seed=config.seed)
    expected_channels = 5 + env.num_robots + int(config.time_channel)
    if env.observation_space.shape != (expected_channels, 12, 12):
        raise AssertionError(
            f"Unexpected helper-free observation shape: {env.observation_space.shape}"
        )

    mask_checks = 0
    for index in selected[: min(30, len(selected))]:
        train_index = int(index) % len(splits["train"])
        _obs, _info = env.reset(options={"puzzle_index": train_index})
        mask = env.action_masks()
        before = list(env._robots)
        for action in range(env.action_space.n):
            moved = (
                env._slide_from(before, action // 4, action % 4, env._puzzle)
                != before[action // 4]
            )
            if bool(mask[action]) != moved:
                raise AssertionError(
                    f"Action-mask mismatch puzzle={env._puzzle.seed} action={action}"
                )
            mask_checks += 1

    net = build_net(config, env.observation_space.shape, env.action_space.n)
    if net._state_hw != (12, 12):
        raise AssertionError(f"Structured DRC encoder changed grid geometry: {net._state_hw}")

    build_fingerprint = hashlib.sha256(
        json.dumps(
            [
                [puzzle.seed, puzzle.optimal_length, puzzle.layout_hash]
                for puzzle in sorted(all_puzzles, key=lambda item: item.seed)
            ]
        ).encode()
    ).hexdigest()[:16]
    return {
        "passed": True,
        "config": config.run_name,
        "bank_path": config.bank_path,
        "bank_file_sha256": _file_sha256(config.bank_path),
        "bank_build_fingerprint": build_fingerprint,
        "split_counts": {name: len(puzzles) for name, puzzles in splits.items()},
        "split_fingerprints": {
            name: _puzzle_fingerprint(puzzles) for name, puzzles in splits.items()
        },
        "split_overlap": overlap,
        "stored_plans_replayed": replayed,
        "mask_checks": mask_checks,
        "observation_shape": list(env.observation_space.shape),
        "observation_channels": {
            "wall_up": 0,
            "wall_down": 1,
            "wall_left": 2,
            "wall_right": 3,
            "goal": 4,
            "robots": list(range(5, 5 + env.num_robots)),
            "time": expected_channels - 1 if config.time_channel else None,
            "stop_planes": None,
            "distance_plane": None,
        },
        "action_count": int(env.action_space.n),
        "drc_state_hw": list(net._state_hw),
        "policy_params": sum(parameter.numel() for parameter in net.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = DRCTrainConfig.from_yaml(args.config)
    report = run_preflight(config, sample_size=args.sample_size)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
