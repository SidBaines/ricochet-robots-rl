"""Evaluate a saved ground-up model on any bank split.

Usage:
    .venv-phase2/bin/python -m src.groundup.eval_cli \
        --model artifacts/groundup/e2_8x8_scratch/model_final.zip \
        --config artifacts/groundup/e2_8x8_scratch/config.json \
        --split test --per-length 60 --episodes 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import EnvConfig
from .evaluate import evaluate_policy, scripted_baseline
from .puzzles import load_puzzles, stratified_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True, help="config.json of the training run")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--per-length", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--suite-seed", type=int, default=17)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train_config = json.loads(Path(args.config).read_text())
    env_config = EnvConfig(
        horizon=train_config["horizon"],
        step_penalty=train_config["step_penalty"],
        goal_reward=train_config["goal_reward"],
        shaping_coef=train_config["shaping_coef"],
        gamma=train_config["gamma"],
        time_channel=train_config["time_channel"],
        canonical_identity=train_config["canonical_identity"],
        stop_planes=train_config.get("stop_planes", False),
        distance_plane=train_config.get("distance_plane", False),
        rgb_obs=train_config.get("rgb_obs", False),
        rgb_cell_px=train_config.get("rgb_cell_px", 5),
    )
    puzzles = load_puzzles(
        train_config["bank_path"],
        args.split,
        eval_fraction=train_config["eval_fraction"],
        split_seed=train_config["split_seed"],
        val_test_seed=train_config["val_test_seed"],
    )
    suite = stratified_subset(puzzles, args.per_length, seed=args.suite_seed)

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(args.model, device=train_config.get("device", "auto"))
    result = {
        "model": args.model,
        "split": args.split,
        "suite_size": len(suite),
        "det": evaluate_policy(model, suite, env_config, episodes_per_puzzle=1, deterministic=True),
        "stoch": evaluate_policy(
            model, suite, env_config, episodes_per_puzzle=args.episodes, deterministic=False
        ),
    }
    if args.baselines:
        result["baseline_random"] = scripted_baseline(
            suite, env_config, policy="random", episodes_per_puzzle=5
        )
        result["baseline_target_only"] = scripted_baseline(
            suite, env_config, policy="target_only", episodes_per_puzzle=5
        )
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
