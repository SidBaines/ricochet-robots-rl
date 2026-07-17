"""Prospective successive-halving and plateau-stopping contracts."""

from dataclasses import asdict, replace

import pytest

from src.benchmarking.phase2 import Phase2Config, _plateau_stopping_status
from src.benchmarking.phase5_selection import (
    SuccessiveHalvingRules,
    adjudicate_successive_halving,
    promotion_config_mapping,
)


def _summary(
    rate: float,
    *,
    interval: tuple[float, float] = (0.0, 1.0),
    robot_2: float = 0.20,
    robot_3: float = 0.10,
    no_movement: float = 0.0,
) -> dict:
    return {
        "success_rate": rate,
        "success_rate_puzzle_bootstrap_95": list(interval),
        "by_robots_moved": {
            "2": {"success_rate": robot_2},
            "3": {"success_rate": robot_3},
        },
        "no_movement_rate": no_movement,
        "invalid_action_rate": 0.0,
    }


def _point(timesteps: int, stochastic: float, deterministic: float, **kwargs) -> dict:
    return {
        "actual_timesteps": timesteps,
        "stochastic": _summary(stochastic, **kwargs),
        "deterministic": _summary(deterministic),
    }


def _candidate(
    rates: tuple[float, float, float],
    *,
    final_interval: tuple[float, float],
    final_deterministic: float = 0.34,
) -> dict:
    curve = [
        _point(0, rates[0], 0.32),
        _point(256000, rates[1], 0.33),
        _point(512000, rates[2], final_deterministic, interval=final_interval),
    ]
    return {"seeds": [{"curve": curve, "checkpoints": []}]}


def test_clear_healthy_consistent_leader_is_promoted() -> None:
    candidates = {
        "a": _candidate((0.30, 0.38, 0.45), final_interval=(0.42, 0.48)),
        "b": _candidate((0.30, 0.34, 0.35), final_interval=(0.31, 0.39)),
    }
    decision = adjudicate_successive_halving(
        candidates,
        rung_timesteps=512000,
        target_survivors=1,
        rules=SuccessiveHalvingRules(),
    )
    assert decision["outcome"] == "promote_clear_leader"
    assert decision["selected"] == ["a"]
    assert decision["eliminated"] == ["b"]


@pytest.mark.parametrize(
    ("candidate_a", "reason"),
    [
        (_candidate((0.30, 0.38, 0.45), final_interval=(0.36, 0.50)), "confidence intervals overlap"),
        (_candidate((0.30, 0.33, 0.45), final_interval=(0.42, 0.48)), "did not lead"),
    ],
)
def test_ambiguous_rung_retains_both_candidates(candidate_a, reason) -> None:
    candidates = {
        "a": candidate_a,
        "b": _candidate((0.30, 0.34, 0.35), final_interval=(0.31, 0.39)),
    }
    decision = adjudicate_successive_halving(
        candidates,
        rung_timesteps=512000,
        target_survivors=1,
        rules=SuccessiveHalvingRules(),
    )
    assert decision["outcome"] == "retain_all_healthy_on_ambiguity"
    assert decision["selected"] == ["a", "b"]
    assert any(reason in value for value in decision["ambiguity_reasons"])


def test_health_regression_makes_candidate_ineligible() -> None:
    candidates = {
        "a": _candidate(
            (0.30, 0.38, 0.45),
            final_interval=(0.42, 0.48),
            final_deterministic=0.20,
        ),
        "b": _candidate((0.30, 0.34, 0.35), final_interval=(0.31, 0.39)),
    }
    decision = adjudicate_successive_halving(
        candidates,
        rung_timesteps=512000,
        target_survivors=1,
        rules=SuccessiveHalvingRules(),
    )
    assert decision["selected"] == ["b"]
    assert "a" in decision["eliminated"]
    assert decision["candidates"]["a"]["health_failures"]


def test_promotion_config_uses_exact_resume_and_predeclared_plateau() -> None:
    base = Phase2Config(
        training_seeds=(11,),
        checkpoints=(0, 512000),
        fixed_validation_seeds=True,
        save_checkpoint_weights=True,
        select_checkpoint_by_validation=True,
        run_final_test=False,
    )
    result = {
        "config": asdict(base),
        "seeds": [{
            "curve": [],
            "checkpoints": [{
                "actual_timesteps": 512000,
                "model": "run/checkpoints/step_512000/model.zip",
                "training_state": "run/checkpoints/step_512000/training_state.pkl",
            }],
        }],
    }
    promoted = promotion_config_mapping(
        result,
        candidate_name="a",
        rung_timesteps=512000,
        target_checkpoints=(512000, 1024000, 2048000),
        plateau={
            "minimum_timesteps": 1024000,
            "patience_checkpoints": 2,
            "minimum_delta": 0.01,
        },
    )
    assert promoted["initial_checkpoint_path"] is None
    assert promoted["resume_checkpoint_path"].endswith("model.zip")
    assert promoted["resume_state_path"].endswith("training_state.pkl")
    assert promoted["checkpoints"] == [512000, 1024000, 2048000]
    assert promoted["plateau_min_timesteps"] == 1024000


def test_plateau_rule_is_prospective_and_resets_on_material_improvement() -> None:
    config = Phase2Config(
        training_seeds=(1,),
        checkpoints=(0, 10, 20, 30),
        fixed_validation_seeds=True,
        save_checkpoint_weights=True,
        select_checkpoint_by_validation=True,
        plateau_min_timesteps=10,
        plateau_patience_checkpoints=2,
        plateau_min_delta=0.05,
    )
    curve = [
        {"actual_timesteps": 0, "stochastic": {"success_rate": 0.20}},
        {"actual_timesteps": 10, "stochastic": {"success_rate": 0.22}},
        {"actual_timesteps": 20, "stochastic": {"success_rate": 0.23}},
    ]
    stopped = _plateau_stopping_status(curve, config)
    assert stopped["stopped_early"]
    assert stopped["non_improving_eligible_checkpoints"] == 2

    recovered = _plateau_stopping_status(
        curve + [{"actual_timesteps": 30, "stochastic": {"success_rate": 0.26}}],
        replace(config, plateau_patience_checkpoints=3),
    )
    assert not recovered["stopped_early"]
    assert recovered["best_timesteps"] == 30
    assert recovered["non_improving_eligible_checkpoints"] == 0


def test_plateau_fields_must_be_complete_and_use_fixed_validation() -> None:
    with pytest.raises(ValueError, match="plateau stopping fields",):
        Phase2Config(plateau_min_timesteps=10)
    with pytest.raises(ValueError, match="fixed_validation_seeds"):
        Phase2Config(
            checkpoints=(0, 10),
            save_checkpoint_weights=True,
            select_checkpoint_by_validation=True,
            plateau_min_timesteps=0,
            plateau_patience_checkpoints=1,
            plateau_min_delta=0.01,
        )
