"""Prospective Phase 5 resource-allocation and promotion rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class SuccessiveHalvingRules:
    """Rules fixed before inspecting a successive-halving rung."""

    minimum_stochastic_margin: float = 0.01
    require_disjoint_confidence_intervals: bool = True
    confidence_interval_field: str = "success_rate_puzzle_bootstrap_95"
    consistent_lead_checkpoints: int = 2
    deterministic_non_regression_tolerance: float = 0.02
    multi_robot_non_regression_tolerance: float = 0.02
    required_robot_bands: tuple[int, ...] = (2, 3)
    require_zero_no_movement: bool = True
    require_zero_invalid_actions: bool = True
    retain_all_on_ambiguity: bool = True

    def __post_init__(self) -> None:
        if self.minimum_stochastic_margin < 0.0:
            raise ValueError("minimum_stochastic_margin must be non-negative")
        if self.consistent_lead_checkpoints <= 0:
            raise ValueError("consistent_lead_checkpoints must be positive")
        if self.deterministic_non_regression_tolerance < 0.0:
            raise ValueError("deterministic_non_regression_tolerance must be non-negative")
        if self.multi_robot_non_regression_tolerance < 0.0:
            raise ValueError("multi_robot_non_regression_tolerance must be non-negative")
        if not self.required_robot_bands:
            raise ValueError("required_robot_bands must not be empty")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SuccessiveHalvingRules":
        normalized = dict(values)
        if "required_robot_bands" in normalized:
            normalized["required_robot_bands"] = tuple(
                int(value) for value in normalized["required_robot_bands"]
            )
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown selection-rule keys: {', '.join(sorted(unknown))}")
        return cls(**normalized)


def _single_seed_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    seeds = result.get("seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or len(seeds) != 1:
        raise ValueError("Each candidate result must contain exactly one seed")
    return seeds[0]


def _point_at(curve: Sequence[Mapping[str, Any]], timesteps: int) -> Mapping[str, Any]:
    matches = [point for point in curve if int(point["actual_timesteps"]) == int(timesteps)]
    if len(matches) != 1:
        raise ValueError(f"Expected one evaluation at rung {timesteps}; found {len(matches)}")
    return matches[0]


def _robot_success(summary: Mapping[str, Any], robot_band: int) -> float:
    by_robot = summary.get("by_robots_moved", {})
    key = str(robot_band)
    if key not in by_robot:
        raise ValueError(f"Evaluation is missing robot-movement band {robot_band}")
    return float(by_robot[key]["success_rate"])


def _candidate_at_rung(
    name: str,
    result: Mapping[str, Any],
    *,
    rung_timesteps: int,
    rules: SuccessiveHalvingRules,
) -> Dict[str, Any]:
    seed_result = _single_seed_result(result)
    curve = list(seed_result["curve"])
    point = _point_at(curve, rung_timesteps)
    baseline = min(curve, key=lambda item: int(item["actual_timesteps"]))
    stochastic = point["stochastic"]
    deterministic = point["deterministic"]
    baseline_stochastic = baseline["stochastic"]
    baseline_deterministic = baseline["deterministic"]
    health_failures = []
    deterministic_regression = (
        float(baseline_deterministic["success_rate"])
        - float(deterministic["success_rate"])
    )
    if deterministic_regression > rules.deterministic_non_regression_tolerance:
        health_failures.append(
            f"deterministic regression {deterministic_regression:.4f} exceeds "
            f"{rules.deterministic_non_regression_tolerance:.4f}"
        )
    robot_regressions: Dict[str, float] = {}
    for robot_band in rules.required_robot_bands:
        regression = (
            _robot_success(baseline_stochastic, robot_band)
            - _robot_success(stochastic, robot_band)
        )
        robot_regressions[str(robot_band)] = regression
        if regression > rules.multi_robot_non_regression_tolerance:
            health_failures.append(
                f"robot band {robot_band} regression {regression:.4f} exceeds "
                f"{rules.multi_robot_non_regression_tolerance:.4f}"
            )
    if rules.require_zero_no_movement and float(stochastic["no_movement_rate"]) != 0.0:
        health_failures.append("masked stochastic evaluation contains no-movement actions")
    if rules.require_zero_invalid_actions and float(stochastic["invalid_action_rate"]) != 0.0:
        health_failures.append("masked stochastic evaluation contains invalid actions")
    interval = stochastic.get(rules.confidence_interval_field)
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise ValueError(
            f"Candidate {name} is missing {rules.confidence_interval_field}"
        )
    return {
        "name": name,
        "eligible": not health_failures,
        "health_failures": health_failures,
        "rung_timesteps": int(rung_timesteps),
        "stochastic_success_rate": float(stochastic["success_rate"]),
        "deterministic_success_rate": float(deterministic["success_rate"]),
        "stochastic_confidence_interval": [float(interval[0]), float(interval[1])],
        "deterministic_regression": deterministic_regression,
        "robot_band_regressions": robot_regressions,
        "curve": {
            int(curve_point["actual_timesteps"]): float(curve_point["stochastic"]["success_rate"])
            for curve_point in curve
            if int(curve_point["actual_timesteps"]) <= rung_timesteps
        },
    }


def adjudicate_successive_halving(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    rung_timesteps: int,
    target_survivors: int,
    rules: SuccessiveHalvingRules,
) -> Dict[str, Any]:
    """Choose a clear healthy leader, retaining candidates when evidence is ambiguous."""
    if len(candidates) < 2:
        raise ValueError("Successive halving requires at least two candidates")
    if target_survivors <= 0 or target_survivors >= len(candidates):
        raise ValueError("target_survivors must be between one and candidate_count - 1")
    summaries = {
        name: _candidate_at_rung(
            name,
            result,
            rung_timesteps=rung_timesteps,
            rules=rules,
        )
        for name, result in sorted(candidates.items())
    }
    healthy = [summary for summary in summaries.values() if summary["eligible"]]
    unhealthy = [summary["name"] for summary in summaries.values() if not summary["eligible"]]
    decision: Dict[str, Any] = {
        "schema_version": 1,
        "rung_timesteps": int(rung_timesteps),
        "target_survivors": int(target_survivors),
        "candidates": summaries,
        "selected": [],
        "eliminated": unhealthy,
        "ambiguous": False,
        "ambiguity_reasons": [],
        "outcome": "halt_no_healthy_candidate",
    }
    if not healthy:
        return decision
    ranked = sorted(
        healthy,
        key=lambda summary: (
            -summary["stochastic_success_rate"],
            -summary["deterministic_success_rate"],
            summary["name"],
        ),
    )
    if len(ranked) <= target_survivors:
        decision["selected"] = [summary["name"] for summary in ranked]
        decision["outcome"] = "promote_all_healthy"
        return decision

    provisional = ranked[:target_survivors]
    boundary_winner = provisional[-1]
    boundary_loser = ranked[target_survivors]
    margin = (
        boundary_winner["stochastic_success_rate"]
        - boundary_loser["stochastic_success_rate"]
    )
    if margin < rules.minimum_stochastic_margin:
        decision["ambiguity_reasons"].append(
            f"stochastic margin {margin:.4f} is below {rules.minimum_stochastic_margin:.4f}"
        )
    if rules.require_disjoint_confidence_intervals:
        winner_low = boundary_winner["stochastic_confidence_interval"][0]
        loser_high = boundary_loser["stochastic_confidence_interval"][1]
        if winner_low <= loser_high:
            decision["ambiguity_reasons"].append(
                "puzzle-clustered stochastic confidence intervals overlap"
            )

    common_timesteps = sorted(
        set(boundary_winner["curve"]) & set(boundary_loser["curve"])
    )
    comparison_timesteps = common_timesteps[-rules.consistent_lead_checkpoints:]
    if len(comparison_timesteps) < rules.consistent_lead_checkpoints:
        decision["ambiguity_reasons"].append(
            f"only {len(comparison_timesteps)} common checkpoints are available for trend confirmation"
        )
    elif any(
        boundary_winner["curve"][timesteps] < boundary_loser["curve"][timesteps]
        for timesteps in comparison_timesteps
    ):
        decision["ambiguity_reasons"].append(
            f"leader did not lead at all last {rules.consistent_lead_checkpoints} checkpoints"
        )

    if decision["ambiguity_reasons"] and rules.retain_all_on_ambiguity:
        decision["ambiguous"] = True
        decision["selected"] = [summary["name"] for summary in ranked]
        decision["outcome"] = "retain_all_healthy_on_ambiguity"
        return decision
    decision["selected"] = [summary["name"] for summary in provisional]
    decision["eliminated"].extend(summary["name"] for summary in ranked[target_survivors:])
    decision["outcome"] = "promote_clear_leader"
    return decision


def promotion_config_mapping(
    result: Mapping[str, Any],
    *,
    candidate_name: str,
    rung_timesteps: int,
    target_checkpoints: Sequence[int],
    plateau: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Turn a rung result into an exact-resume Phase2Config mapping."""
    config = dict(result["config"])
    seed_result = _single_seed_result(result)
    records = [
        record for record in seed_result["checkpoints"]
        if int(record["actual_timesteps"]) == int(rung_timesteps)
    ]
    if len(records) != 1:
        raise ValueError(f"Candidate {candidate_name} has no unique rung checkpoint")
    record = records[0]
    if not record.get("model") or not record.get("training_state"):
        raise ValueError(
            f"Candidate {candidate_name} rung checkpoint is not exactly resumable"
        )
    checkpoints = tuple(int(value) for value in target_checkpoints)
    if not checkpoints or checkpoints[0] != int(rung_timesteps):
        raise ValueError("Promotion checkpoints must begin at the adjudicated rung")
    config.update({
        "initial_checkpoint_path": None,
        "resume_checkpoint_path": record["model"],
        "resume_state_path": record["training_state"],
        "checkpoints": list(checkpoints),
        "bc_pretrain_epochs": 0,
        "run_overfit_milestone": False,
        "run_final_test": False,
        "plateau_min_timesteps": None,
        "plateau_patience_checkpoints": None,
        "plateau_min_delta": None,
    })
    if plateau is not None:
        config.update({
            "plateau_min_timesteps": int(plateau["minimum_timesteps"]),
            "plateau_patience_checkpoints": int(plateau["patience_checkpoints"]),
            "plateau_min_delta": float(plateau["minimum_delta"]),
        })
    if config.get("wandb_run_name"):
        config["wandb_run_name"] = (
            f"{config['wandb_run_name']}-resume-{rung_timesteps}-to-{checkpoints[-1]}"
        )

    # Validate the generated contract now, before a long job is launched.
    from src.benchmarking.phase2 import Phase2Config

    Phase2Config.from_mapping(config)
    return config
