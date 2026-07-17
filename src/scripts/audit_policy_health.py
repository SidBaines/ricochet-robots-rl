"""Audit a saved Phase 2 policy on a fixed bank suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase2 import (  # noqa: E402
    Phase2Config,
    _bc_dataset_impl,
    _layout_id,
    _lookup_puzzles,
    _observation_and_mask_probe,
    _policy_health_payload,
    _puzzle_id,
    evaluate_controller,
)
from src.env.puzzle_bank import PuzzleBank  # noqa: E402


def _safe(value):
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _expert_metrics(model, puzzles, config, *, seed: int) -> dict:
    observations = []
    actions = []
    masks = []
    strata = []
    for puzzle in puzzles:
        obs, act, mask = _bc_dataset_impl(
            [puzzle],
            config,
            seed=seed,
            include_masks=True,
        )
        observations.append(obs)
        actions.append(act)
        masks.append(mask)
        strata.extend([(int(puzzle.optimal_length), int(puzzle.robots_moved))] * len(act))
    obs_tensor = torch.as_tensor(np.concatenate(observations), dtype=torch.float32, device=model.device)
    action_tensor = torch.as_tensor(np.concatenate(actions), device=model.device)
    mask_tensor = torch.as_tensor(np.concatenate(masks), device=model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        if config.use_action_masking:
            distribution.apply_masking(mask_tensor)
        probabilities = distribution.distribution.probs
        predictions = probabilities.argmax(dim=-1)
        target_probability = probabilities.gather(1, action_tensor[:, None]).squeeze(1)
    correct = (predictions == action_tensor).cpu().numpy()
    target_probability_np = target_probability.cpu().numpy()
    grouped = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[stratum].append(index)
    return {
        "pairs": len(strata),
        "top1_agreement": float(correct.mean()),
        "mean_target_probability": float(target_probability_np.mean()),
        "by_stratum": {
            f"{length},{robots}": {
                "pairs": len(indices),
                "top1_agreement": float(correct[indices].mean()),
                "mean_target_probability": float(target_probability_np[indices].mean()),
            }
            for (length, robots), indices in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("training_suite", "validation_suite", "test_suite", "train", "validation", "test"),
        default="validation_suite",
    )
    parser.add_argument("--max-puzzles", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = Phase2Config.from_mapping(raw)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    lookup = _lookup_puzzles(PuzzleBank(args.bank_dir), config)
    ids = manifest[args.suite] if args.suite.endswith("_suite") else manifest["memberships"][args.suite]
    puzzles = [lookup[puzzle_id] for puzzle_id in ids[: args.max_puzzles]]

    if config.use_action_masking:
        from sb3_contrib import MaskablePPO as Algorithm
    else:
        from stable_baselines3 import PPO as Algorithm
    model = Algorithm.load(args.checkpoint, device=args.device)
    observations, movement_masks = _observation_and_mask_probe(puzzles, config, seed=20260717)
    parameter_counts = Counter()
    for name, parameter in model.policy.named_parameters():
        component = name.split(".", 1)[0]
        parameter_counts[component] += parameter.numel()
    deterministic = evaluate_controller(
        "model",
        puzzles,
        config.environment_spec(20260718),
        episodes_per_puzzle=1,
        seed=20260719,
        model=model,
        deterministic=True,
        action_masking=config.use_action_masking,
    )
    stochastic = evaluate_controller(
        "model",
        puzzles,
        config.environment_spec(20260720),
        episodes_per_puzzle=args.episodes,
        seed=20260721,
        model=model,
        deterministic=False,
        action_masking=config.use_action_masking,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "algorithm_class": type(model).__name__,
        "policy_class": type(model.policy).__name__,
        "policy_kwargs_from_checkpoint": _safe(model.policy_kwargs),
        "parameter_counts": dict(parameter_counts),
        "total_parameters": sum(parameter_counts.values()),
        "suite": args.suite,
        "puzzles": len(puzzles),
        "puzzle_ids": [_puzzle_id(puzzle) for puzzle in puzzles],
        "layout_ids": [_layout_id(puzzle) for puzzle in puzzles],
        "suite_histogram": dict(Counter(
            f"{puzzle.optimal_length},{puzzle.robots_moved}" for puzzle in puzzles
        )),
        "health": _policy_health_payload(model, observations, movement_masks),
        "expert_states": _expert_metrics(model, puzzles, config, seed=20260722),
        "deterministic_rollout": deterministic["summary"],
        "stochastic_rollout": stochastic["summary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        f"checkpoint: {args.checkpoint}",
        f"policy: {result['policy_class']} ({result['total_parameters']:,} parameters)",
        f"suite: {args.suite} ({len(puzzles)} puzzles)",
        f"expert top-1: {result['expert_states']['top1_agreement']:.1%}",
        f"expert target probability: {result['expert_states']['mean_target_probability']:.1%}",
        f"deterministic success: {deterministic['summary']['success_rate']:.1%}",
        f"stochastic success: {stochastic['summary']['success_rate']:.1%}",
        f"logit between-state variance: {result['health']['health/logit_between_state_variance']:.6g}",
        f"no-movement probability mass: {result['health']['health/no_movement_probability_mass']:.1%}",
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
