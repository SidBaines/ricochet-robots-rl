"""Measure reverse-start baselines and derive frontier mastery thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase2 import Phase2Config, _lookup_puzzles, _puzzle_id, _sha256_json, _wilson_interval  # noqa: E402
from src.env.criteria_env import layout_at_start_distance  # noqa: E402
from src.env.puzzle_bank import PuzzleBank, create_fixed_layout_from_seed  # noqa: E402
from src.env.ricochet_env import RicochetRobotsEnv  # noqa: E402


def _rollout(layout, config, controller: str, rng: np.random.Generator) -> bool:
    env = RicochetRobotsEnv(**config.environment_spec(0).to_env_kwargs(fixed_layout=layout))
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        if controller == "random":
            action = int(rng.integers(0, env.action_space.n))
        else:
            action = int(layout.target_robot) * 4 + int(rng.integers(0, 4))
        _, _, terminated, truncated, _ = env.step(action)
    env.close()
    return bool(terminated and not truncated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", required=True)
    parser.add_argument("--max-puzzles", type=int, default=500)
    parser.add_argument("--episodes-per-puzzle", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=20_260_717)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = Phase2Config.from_mapping(raw)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    lookup = _lookup_puzzles(PuzzleBank(args.bank_dir), config)
    training = [lookup[puzzle_id] for puzzle_id in manifest["memberships"]["train"]]
    result = {
        "schema_version": 1,
        "parent_split_fingerprint": manifest["fingerprint"],
        "selection_seed": args.selection_seed,
        "episodes_per_puzzle": args.episodes_per_puzzle,
        "max_puzzles": args.max_puzzles,
        "distances": {},
    }
    for distance in args.distances:
        eligible = [puzzle for puzzle in training if puzzle.optimal_length >= distance]
        rng = np.random.default_rng(args.selection_seed + distance)
        order = rng.permutation(len(eligible))[: args.max_puzzles]
        selected = [eligible[int(index)] for index in order]
        layouts = [
            layout_at_start_distance(puzzle, create_fixed_layout_from_seed(puzzle), distance)
            for puzzle in selected
        ]
        controllers = {}
        for controller_index, controller in enumerate(("random", "target_robot_only")):
            rollout_rng = np.random.default_rng(args.selection_seed + distance * 10 + controller_index)
            successes = sum(
                _rollout(layout, config, controller, rollout_rng)
                for layout in layouts
                for _ in range(args.episodes_per_puzzle)
            )
            episodes = len(layouts) * args.episodes_per_puzzle
            controllers[controller] = {
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes,
                "wilson_95": _wilson_interval(successes, episodes),
            }
        target_rate = controllers["target_robot_only"]["success_rate"]
        result["distances"][str(distance)] = {
            "puzzle_ids": [_puzzle_id(puzzle) for puzzle in selected],
            "controllers": controllers,
            "frontier_threshold": min(0.9, max(0.7, target_rate + 0.1)),
        }
    result["fingerprint"] = _sha256_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != result:
            raise ValueError(f"Existing calibration differs: {args.output}")
    else:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reverse calibration fingerprint={result['fingerprint']}")
    for distance, values in result["distances"].items():
        random_rate = values["controllers"]["random"]["success_rate"]
        target_rate = values["controllers"]["target_robot_only"]["success_rate"]
        print(
            f"d={distance}: random={random_rate:.1%} target={target_rate:.1%} "
            f"threshold={values['frontier_threshold']:.1%}"
        )


if __name__ == "__main__":
    main()
