from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest
import torch

import src.groundup.intervene_drc_carry as interventions
from src.groundup.drc import DRCNet, DRCState
from src.groundup.drc_train import DRCTrainConfig, _save_model, build_net, evaluate_drc
from src.groundup.env import EnvConfig, RicochetEnv
from src.groundup.intervene_drc_carry import (
    DEFAULT_TIMING_MOVES,
    PatchContext,
    aggregate_paired_effects,
    evaluate_core_interventions,
    evaluate_matched_patches,
    intervene_checkpoint,
    intervention_specs,
    patch_carry_state,
    run_deterministic_episode,
    select_matched_patch_pair,
    write_json_atomic,
    zero_carry_state,
)
from src.groundup.puzzles import Puzzle


def _open_board(height: int = 3, width: int = 3) -> tuple[np.ndarray, np.ndarray]:
    h_walls = np.zeros((height + 1, width), dtype=bool)
    v_walls = np.zeros((height, width + 1), dtype=bool)
    h_walls[[0, -1], :] = True
    v_walls[:, [0, -1]] = True
    return h_walls, v_walls


def _two_way_puzzle(*, layout_hash: str = "two-way", seed: int = 7) -> Puzzle:
    h_walls, v_walls = _open_board()
    return Puzzle(
        h_walls=h_walls,
        v_walls=v_walls,
        robots=((1, 1),),
        goal=(0, 0),
        target=0,
        optimal_length=2,
        actions=((0, 0), (0, 2)),
        robots_moved=1,
        layout_hash=layout_hash,
        seed=seed,
    )


def _cycling_puzzle(*, layout_hash: str = "cycle", seed: int = 8) -> Puzzle:
    """A lone robot cannot land on its central goal, so it reaches the horizon."""

    h_walls, v_walls = _open_board()
    return Puzzle(
        h_walls=h_walls,
        v_walls=v_walls,
        robots=((1, 1),),
        goal=(1, 1),
        target=0,
        optimal_length=1,
        actions=((0, 0),),
        robots_moved=1,
        layout_hash=layout_hash,
        seed=seed,
    )


def _tiny_net(seed: int = 4, *, horizon: int = 10) -> tuple[DRCNet, DRCTrainConfig]:
    config = DRCTrainConfig(
        seed=seed,
        horizon=horizon,
        shaping_coef=0.0,
        drc_depth=2,
        drc_ticks=1,
        drc_hidden_channels=3,
        drc_encoder_channels=(3,),
        drc_encoder_kernels=(3,),
        drc_encoder_strides=(1,),
        drc_mlp_hidden=12,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = build_net(config, (7, 3, 3), 4)
    with torch.no_grad():
        net.actor.weight.zero_()
        # UP/DOWN alternation on the cycling puzzle; UP then LEFT solves the
        # two-way puzzle because UP is illegal from the top edge.
        net.actor.bias.copy_(torch.tensor([4.0, 2.0, 3.0, 1.0]))
    return net, config


def _env_config(config: DRCTrainConfig) -> EnvConfig:
    return EnvConfig(
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


def _numpy_rng_equal(left, right) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_state_transforms_zero_and_patch_only_requested_values_without_mutation():
    h = torch.arange(3 * 2 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2, 2) + 1
    c = h + 1_000
    state = DRCState(h=h.clone(), c=c.clone())
    original = DRCState(h=state.h.clone(), c=state.c.clone())

    depth = zero_carry_state(state, component="hc", depth=1)
    assert torch.count_nonzero(depth.h[1]) == 0
    assert torch.count_nonzero(depth.c[1]) == 0
    assert torch.equal(depth.h[[0, 2]], original.h[[0, 2]])
    assert torch.equal(depth.c[[0, 2]], original.c[[0, 2]])

    h_only = zero_carry_state(state, component="h")
    c_only = zero_carry_state(state, component="c")
    assert torch.count_nonzero(h_only.h) == 0 and torch.equal(h_only.c, original.c)
    assert torch.count_nonzero(c_only.c) == 0 and torch.equal(c_only.h, original.h)
    assert torch.equal(state.h, original.h) and torch.equal(state.c, original.c)

    donor = DRCState(h=original.h + 10_000, c=original.c + 20_000)
    donor_before = DRCState(h=donor.h.clone(), c=donor.c.clone())
    patched_h = patch_carry_state(state, donor, component="h")
    patched_c = patch_carry_state(state, donor, component="c")
    patched_full = patch_carry_state(state, donor, component="hc")
    assert torch.equal(patched_h.h, donor.h) and torch.equal(patched_h.c, state.c)
    assert torch.equal(patched_c.h, state.h) and torch.equal(patched_c.c, donor.c)
    assert torch.equal(patched_full.h, donor.h) and torch.equal(patched_full.c, donor.c)
    assert patched_full.h.data_ptr() != donor.h.data_ptr()
    assert torch.equal(state.h, original.h) and torch.equal(state.c, original.c)
    assert torch.equal(donor.h, donor_before.h) and torch.equal(donor.c, donor_before.c)

    with pytest.raises(ValueError, match="outside"):
        zero_carry_state(state, depth=3)
    with pytest.raises(ValueError, match="component"):
        patch_carry_state(state, donor, component="bad")


def test_reset_timing_is_exact_once_and_every_move_excludes_move_zero():
    net, config = _tiny_net(horizon=10)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)
    specs = intervention_specs(net.depth)
    by_id = {spec.intervention_id: spec for spec in specs}

    every_move = run_deterministic_episode(
        net,
        puzzle,
        env_config,
        intervention=by_id["reset_all_every_move"],
    )
    assert every_move.moves == 10
    assert every_move.applied_at_completed_moves == tuple(range(1, 10))
    assert 0 not in every_move.applied_at_completed_moves

    for timing in DEFAULT_TIMING_MOVES:
        trace = run_deterministic_episode(
            net,
            puzzle,
            env_config,
            intervention=by_id[f"reset_once_after_{timing}_moves"],
        )
        assert trace.applied_at_completed_moves == (timing,)

    short_net, short_config = _tiny_net(horizon=2)
    progress = []
    short = evaluate_core_interventions(
        short_net,
        [_cycling_puzzle()],
        _env_config(short_config),
        timing_moves=DEFAULT_TIMING_MOVES,
        bootstrap_reps=8,
        progress=progress.append,
    )
    after_two = short["interventions"]["reset_once_after_2_moves"]
    assert after_two["application_counts"]["eligible_pre_intervention"] == 0
    assert after_two["application_counts"]["not_eligible_pre_intervention"] == 1
    assert after_two["paired_outcomes"][0]["intervention_applied"] is False
    assert after_two["effects"]["overall"]["n"] == 1  # intention to treat
    assert after_two["effects"]["overall"]["n_eligible"] == 0
    assert after_two["effects"]["overall"]["eligibility_rate"] == 0.0
    after_two_eligible = after_two["effects"]["eligible_puzzles_descriptive_effects"]
    assert after_two_eligible["overall"]["n"] == 0
    assert after_two_eligible["subset"]["all_selected_rows_intervention_applied"] is None
    assert any("core puzzle-arm evaluations" in message for message in progress)
    assert progress[-1] == "core phase complete"


def test_baseline_matches_existing_deterministic_drc_evaluation_semantics():
    net, config = _tiny_net(horizon=4)
    puzzle = _two_way_puzzle()
    env_config = _env_config(config)

    trace = run_deterministic_episode(net, puzzle, env_config)
    existing = evaluate_drc(
        net,
        [puzzle],
        env_config,
        deterministic=True,
        episodes_per_puzzle=1,
        batch_size=1,
        seed=91,
    )
    core = evaluate_core_interventions(
        net,
        [puzzle],
        env_config,
        bootstrap_reps=8,
    )

    assert trace.success is True
    assert trace.actions == (0, 2)
    assert existing["episodes"] == 1
    assert existing["success_rate"] == 1.0
    assert core["baseline"]["success_rate"] == existing["success_rate"]
    assert core["baseline"]["outcomes"][0]["moves"] == trace.moves
    assert core["baseline"]["outcomes"][0]["actions"] == [0, 2]
    assert {arm["effects"]["estimand_role"] for arm in core["interventions"].values()} == {
        "all_selected_puzzles_itt_estimand"
    }


def test_paired_aggregation_is_stratified_deterministic_and_explicit_when_small():
    rows = [
        {"optimal_length": 1, "robots_moved": 1, "baseline_success": False, "intervention_success": True},
        {"optimal_length": 1, "robots_moved": 1, "baseline_success": True, "intervention_success": False},
        {"optimal_length": 1, "robots_moved": 2, "baseline_success": True, "intervention_success": True},
        {"optimal_length": 2, "robots_moved": 1, "baseline_success": False, "intervention_success": False},
    ]
    kwargs = {
        "bootstrap_seed": 123,
        "bootstrap_reps": 200,
        "seed_namespace": "unit",
        "reference_strata": [(1, 1), (1, 2), (2, 1)],
    }
    first = aggregate_paired_effects(rows, **kwargs)
    second = aggregate_paired_effects(rows, **kwargs)
    assert first == second
    assert first["estimand_role"] == "paired_puzzle_estimand"
    overall = first["overall"]
    assert overall["n"] == 4
    assert overall["baseline_success_rate"] == 0.5
    assert overall["intervention_success_rate"] == 0.5
    assert overall["mean_paired_success_delta"] == 0.0
    assert (overall["improved"], overall["harmed"], overall["tied"]) == (1, 1, 2)
    assert overall["bootstrap_ci"]["unit"] == "puzzle"
    assert overall["bootstrap_ci"]["coverage_scope"] == "pointwise_only"
    assert overall["bootstrap_ci"]["multiplicity_adjustment"] == "none"
    assert overall["bootstrap_ci"]["family_wise_coverage"] is False
    assert overall["bootstrap_ci"]["simultaneous_coverage"] is False
    assert overall["bootstrap_ci"]["zero_exclusion_claim_allowed"] is False
    assert first["by_optimal_length"]["1"]["bootstrap_ci"]["inference_role"] == (interventions.SUBGROUP_INFERENCE_ROLE)
    assert first["by_optimal_length"]["2"]["size_status"] == "single_item"
    assert first["by_optimal_length"]["2"]["bootstrap_ci"]["status"] == "single_item_degenerate"
    missing = first["by_joint_cell"]["2|2"]
    assert missing["n"] == 0 and missing["size_status"] == "empty"
    assert missing["mean_paired_success_delta"] is None
    assert missing["bootstrap_ci"]["lower"] is None
    json.dumps(first, allow_nan=False)

    empty = aggregate_paired_effects([], bootstrap_seed=1, bootstrap_reps=5)
    assert empty["overall"]["n"] == 0
    assert empty["overall"]["baseline_success_rate"] is None


def test_timing_effects_report_unequal_stratum_coverage_and_descriptive_eligible_subset():
    rows = [
        {
            "optimal_length": 1,
            "robots_moved": 1,
            "baseline_success": False,
            "intervention_success": True,
            "eligible_pre_intervention": True,
            "intervention_applied": True,
        },
        {
            "optimal_length": 1,
            "robots_moved": 1,
            "baseline_success": True,
            "intervention_success": False,
            "eligible_pre_intervention": True,
            "intervention_applied": True,
        },
        {
            "optimal_length": 2,
            "robots_moved": 1,
            "baseline_success": False,
            "intervention_success": False,
            "eligible_pre_intervention": True,
            "intervention_applied": True,
        },
        *(
            {
                "optimal_length": 2,
                "robots_moved": 1,
                "baseline_success": success,
                "intervention_success": success,
                "eligible_pre_intervention": False,
                "intervention_applied": False,
            }
            for success in (True, False, True)
        ),
        *(
            {
                "optimal_length": 2,
                "robots_moved": 2,
                "baseline_success": success,
                "intervention_success": success,
                "eligible_pre_intervention": False,
                "intervention_applied": False,
            }
            for success in (False, True)
        ),
    ]
    effects = aggregate_paired_effects(
        rows,
        bootstrap_seed=91,
        bootstrap_reps=100,
        seed_namespace="unequal_timing_coverage",
        reference_strata=[(1, 1), (2, 1), (2, 2)],
        report_eligibility_application=True,
        estimand_role="all_selected_puzzles_itt_estimand",
        overall_inference_role=interventions.LOCALIZATION_INFERENCE_ROLE,
    )

    assert effects["estimand_role"] == "all_selected_puzzles_itt_estimand"
    all_itt_summaries = [
        effects["overall"],
        *effects["by_optimal_length"].values(),
        *effects["by_robots_moved"].values(),
        *effects["by_joint_cell"].values(),
    ]
    coverage_fields = {
        "n_eligible",
        "eligibility_rate",
        "eligibility_denominator_n",
        "n_applied",
        "application_rate",
        "application_denominator_n",
    }
    for summary in all_itt_summaries:
        assert coverage_fields <= set(summary)
        assert summary["eligibility_denominator_n"] == summary["n"]
        assert summary["application_denominator_n"] == summary["n"]

    overall = effects["overall"]
    assert overall["n"] == 8
    assert overall["eligibility_denominator"] == "all_puzzles_in_effect_summary"
    assert overall["eligibility_denominator_n"] == 8
    assert overall["n_eligible"] == 3
    assert overall["eligibility_rate"] == 0.375
    assert overall["application_denominator"] == "all_puzzles_in_effect_summary"
    assert overall["application_denominator_n"] == 8
    assert overall["n_applied"] == 3
    assert overall["application_rate"] == 0.375
    assert overall["application_among_eligible_denominator_n"] == 3
    assert overall["application_rate_among_eligible"] == 1.0
    assert overall["bootstrap_ci"]["inference_role"] == interventions.LOCALIZATION_INFERENCE_ROLE

    short = effects["by_optimal_length"]["1"]
    long = effects["by_optimal_length"]["2"]
    assert (short["n"], short["n_eligible"], short["eligibility_rate"]) == (2, 2, 1.0)
    assert (long["n"], long["n_eligible"], long["eligibility_rate"]) == (6, 1, 1 / 6)
    assert short["bootstrap_ci"]["inference_role"] == interventions.SUBGROUP_INFERENCE_ROLE
    assert effects["by_joint_cell"]["1|1"]["eligibility_rate"] == 1.0
    assert effects["by_joint_cell"]["2|1"]["eligibility_rate"] == 0.25
    assert effects["by_joint_cell"]["2|2"]["eligibility_rate"] == 0.0
    assert effects["by_robots_moved"]["1"]["eligibility_rate"] == 0.5
    assert effects["by_robots_moved"]["2"]["eligibility_rate"] == 0.0

    empty = effects["by_joint_cell"]["1|2"]
    assert empty["n"] == 0
    assert empty["eligibility_denominator_n"] == 0
    assert empty["n_eligible"] == 0 and empty["n_applied"] == 0
    assert empty["eligibility_rate"] is None and empty["application_rate"] is None
    assert empty["application_rate_among_eligible"] is None

    eligible = effects["eligible_puzzles_descriptive_effects"]
    assert eligible["estimand_role"] == "secondary_descriptive_principal_stratum_not_randomized_or_itt"
    assert eligible["subset"]["is_primary_contrast"] is False
    assert eligible["subset"]["is_randomized_or_itt_estimand"] is False
    assert eligible["subset"]["interpretation"] == "descriptive_principal_stratum_only"
    assert eligible["subset"]["all_selected_rows_intervention_applied"] is True
    assert eligible["overall"]["n"] == 3
    assert eligible["by_optimal_length"]["2"]["n"] == 1
    assert eligible["by_joint_cell"]["1|2"]["n"] == 0
    assert eligible["overall"]["bootstrap_ci"]["inference_role"] == (interventions.ELIGIBLE_SUBSET_INFERENCE_ROLE)
    assert eligible["overall"]["bootstrap_ci"]["multiplicity_adjustment"] == "none"
    json.dumps(effects, allow_nan=False)

    invalid = [dict(rows[0], eligible_pre_intervention=False, intervention_applied=True)]
    with pytest.raises(ValueError, match="cannot be applied to an ineligible"):
        aggregate_paired_effects(
            invalid,
            bootstrap_seed=1,
            bootstrap_reps=2,
            report_eligibility_application=True,
        )


def _patch_context(
    net: DRCNet,
    puzzle: Puzzle,
    env_config: EnvConfig,
    *,
    rollout_id: int,
    history: tuple[int, ...],
    suite_puzzle_index: int = 0,
    positions_override: tuple[tuple[int, int], ...] | None = None,
    move_override: int | None = None,
    layout_override: str | None = None,
    fill: float = 0.0,
) -> PatchContext:
    env = RicochetEnv([puzzle], env_config, sampler=None, seed=333)
    obs, _info = env.reset(options={"puzzle_index": 0})
    # Put the environment at a real nonterminal decision point.
    action = 0
    obs, _reward, terminated, truncated, _info = env.step(action)
    assert not (terminated or truncated)
    state = net.initial_state(1)
    state = DRCState(h=torch.full_like(state.h, fill), c=torch.full_like(state.c, fill + 1.0))
    positions = tuple(tuple(position) for position in env._robots)
    return PatchContext(
        suite_puzzle_index=suite_puzzle_index,
        puzzle_seed=puzzle.seed,
        layout_hash=layout_override or puzzle.layout_hash,
        rollout_id=rollout_id,
        rollout_seed=100 + rollout_id,
        move_index=len(history) if move_override is None else move_override,
        positions=positions if positions_override is None else positions_override,
        action_history=history,
        observation=np.asarray(obs).copy(),
        environment_state=copy.deepcopy(env.state_dict()),
        carry=state,
    )


def test_patch_pair_requires_exact_context_and_different_histories_and_selection_is_stable():
    net, config = _tiny_net(horizon=4)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)
    first = _patch_context(net, puzzle, env_config, rollout_id=0, history=(0,), fill=1.0)
    same_history = _patch_context(net, puzzle, env_config, rollout_id=1, history=(0,), fill=2.0)
    donor = _patch_context(net, puzzle, env_config, rollout_id=2, history=(1,), fill=3.0)

    assert select_matched_patch_pair([first, same_history]) is None
    pair = select_matched_patch_pair([donor, same_history, first])
    assert pair is not None
    assert pair.recipient.rollout_id == 0
    assert pair.donor.rollout_id == 2
    reordered = select_matched_patch_pair([first, donor, same_history])
    assert reordered is not None
    assert (reordered.recipient.rollout_id, reordered.donor.rollout_id) == (0, 2)

    different_move = dataclasses_replace(donor, move_index=2)
    different_position = dataclasses_replace(donor, positions=((2, 2),))
    different_puzzle = dataclasses_replace(donor, suite_puzzle_index=1)
    different_layout = dataclasses_replace(donor, layout_hash="other")
    for ineligible in (different_move, different_position, different_puzzle, different_layout):
        assert select_matched_patch_pair([first, ineligible]) is None


def dataclasses_replace(value, **changes):
    # Local wrapper keeps the test import list focused on the scientific API.
    import dataclasses

    return dataclasses.replace(value, **changes)


def test_patch_pair_checks_observation_rng_and_time_values_not_only_hashes(monkeypatch):
    net, config = _tiny_net(horizon=4)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)
    recipient = _patch_context(net, puzzle, env_config, rollout_id=0, history=(0,), fill=1.0)
    donor = _patch_context(net, puzzle, env_config, rollout_id=1, history=(1,), fill=2.0)

    monkeypatch.setattr(interventions, "_array_fingerprint", lambda _value: "forced-collision")
    altered_observation = donor.observation.copy()
    altered_observation.flat[0] = 1.0 - altered_observation.flat[0]
    with pytest.raises(AssertionError, match="byte-identical"):
        select_matched_patch_pair([recipient, dataclasses_replace(donor, observation=altered_observation)])

    monkeypatch.setattr(interventions, "_environment_state_fingerprint", lambda _value: "forced-collision")
    altered_rng_snapshot = copy.deepcopy(donor.environment_state)
    altered_rng_snapshot["rng_state"]["has_uint32"] = 1 - int(altered_rng_snapshot["rng_state"]["has_uint32"])
    with pytest.raises(AssertionError, match="RNG states"):
        select_matched_patch_pair([recipient, dataclasses_replace(donor, environment_state=altered_rng_snapshot)])

    altered_time_snapshot = copy.deepcopy(donor.environment_state)
    altered_time_snapshot["steps"] = donor.move_index + 1
    with pytest.raises(AssertionError, match="steps.*move index"):
        select_matched_patch_pair([recipient, dataclasses_replace(donor, environment_state=altered_time_snapshot)])


def test_patch_runner_uses_identical_suffix_state_requested_components_and_no_leakage(monkeypatch):
    net, config = _tiny_net(horizon=4)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)
    recipient = _patch_context(net, puzzle, env_config, rollout_id=0, history=(0,), fill=0.25)
    donor = _patch_context(net, puzzle, env_config, rollout_id=1, history=(1,), fill=0.75)
    recipient_before = (recipient.carry.h.clone(), recipient.carry.c.clone())
    donor_before = (donor.carry.h.clone(), donor.carry.c.clone())
    recipient_environment_before = copy.deepcopy(recipient.environment_state)
    donor_environment_before = copy.deepcopy(donor.environment_state)
    recipient_observation_before = recipient.observation.copy()
    donor_observation_before = donor.observation.copy()

    def fake_contexts(*_args, **_kwargs):
        return [recipient, donor], [
            {"rollout_id": 0, "rollout_seed": 100, "moves": 1, "success": False, "action_history_sha256": "a"},
            {"rollout_id": 1, "rollout_seed": 101, "moves": 1, "success": False, "action_history_sha256": "b"},
        ]

    real_suffix = interventions._run_deterministic_suffix
    suffix_inputs = []

    def record_suffix_inputs(*args, environment_state, carry, **kwargs):
        suffix_inputs.append((environment_state, carry))
        return real_suffix(*args, environment_state=environment_state, carry=carry, **kwargs)

    monkeypatch.setattr(interventions, "_collect_stochastic_contexts", fake_contexts)
    monkeypatch.setattr(interventions, "_run_deterministic_suffix", record_suffix_inputs)
    result = evaluate_matched_patches(
        net,
        [puzzle],
        env_config,
        patch_rollouts=2,
        patch_seed=5,
        bootstrap_reps=8,
    )
    assert result["status"] == "complete"
    assert result["eligibility"]["eligible_puzzles"] == 1
    assert result["selection"]["outcome_fields_used"] is False
    assert result["primary_estimand"] == ("mean(recipient_patched_suffix_success - recipient_natural_suffix_success)")
    assert result["donor_natural_control"]["status"] == "passed_for_all_eligible_contexts"
    assert len(suffix_inputs) == 5
    assert suffix_inputs[0][0] is recipient.environment_state
    assert suffix_inputs[0][1] is recipient.carry
    assert suffix_inputs[1][0] is donor.environment_state
    assert suffix_inputs[1][1] is donor.carry
    for component in ("hc", "h", "c"):
        arm = result["patches"][component]
        assert arm["effects"]["overall"]["n"] == 1
        assert arm["effects"]["estimand"] == result["primary_estimand"]
        assert arm["effects"]["estimand_role"] == "paired_recipient_suffix_estimand"
        expected_role = (
            interventions.PRIMARY_OVERALL_INFERENCE_ROLE
            if component == "hc"
            else interventions.LOCALIZATION_INFERENCE_ROLE
        )
        assert arm["effects"]["overall"]["bootstrap_ci"]["inference_role"] == expected_role
        assert arm["effects"]["overall"]["bootstrap_ci"]["multiplicity_adjustment"] == "none"
        record = arm["paired_outcomes"][0]
        assert record["context"]["recipient_action_prefix"] == [0]
        assert record["context"]["donor_action_prefix"] == [1]
        assert (
            record["context"]["environment_state_sha256_by_role"]["recipient"]
            == record["context"]["environment_state_sha256_by_role"]["donor"]
        )
        assert (
            record["context"]["environment_rng_state_sha256_by_role"]["recipient"]
            == record["context"]["environment_rng_state_sha256_by_role"]["donor"]
        )
        assert (
            record["context"]["observation_sha256_by_role"]["recipient"]
            == record["context"]["observation_sha256_by_role"]["donor"]
        )
        assert (
            record["natural_suffix"]["initial_environment_sha256"]
            == record["patched_suffix"]["initial_environment_sha256"]
        )
        assert (
            record["natural_suffix"]["initial_observation_sha256"]
            == record["patched_suffix"]["initial_observation_sha256"]
        )
        assert record["suffix_roles"]["natural_suffix"] == "recipient.environment_state + recipient.carry"
        assert record["baseline_success"] == record["natural_suffix"]["success"]
        assert record["intervention_success"] == record["patched_suffix"]["success"]
        assert record["recipient_natural_suffix_success"] == record["baseline_success"]
        assert record["recipient_patched_suffix_success"] == record["intervention_success"]
        assert record["donor_natural_suffix_success"] == record["donor_natural_suffix"]["success"]
        assert record["verification"]["environment_snapshots_exactly_equal"] is True
        assert record["verification"]["environment_rng_states_exactly_equal"] is True
        assert record["verification"]["environment_time_steps_exactly_equal"] is True
        assert record["verification"]["saved_observations_byte_identical"] is True
        assert record["verification"]["recipient_saved_observation_reproduced_exactly"] is True
        assert record["verification"]["donor_saved_observation_reproduced_exactly"] is True
        assert record["donor_natural_hc_transplant_control"]["passed"] is True
        assert record["verification"]["only_requested_components_written"] is True
        assert record["verification"]["no_state_reuse_between_suffix_runs"] is True
        assert record["verification"]["source_environment_snapshots_unmutated_after_all_suffixes"] is True
        assert record["verification"]["source_observations_unmutated_after_all_suffixes"] is True
    hc_record = result["patches"]["hc"]["paired_outcomes"][0]
    assert hc_record["donor_natural_suffix"]["actions"] == hc_record["patched_suffix"]["actions"]
    assert hc_record["donor_natural_suffix"]["success"] == hc_record["patched_suffix"]["success"]
    assert result["patches"]["h"]["paired_outcomes"][0]["verification"]["unrequested_components_equal_recipient"]
    assert torch.equal(recipient.carry.h, recipient_before[0])
    assert torch.equal(recipient.carry.c, recipient_before[1])
    assert torch.equal(donor.carry.h, donor_before[0])
    assert torch.equal(donor.carry.c, donor_before[1])
    assert interventions._json_values_exactly_equal(recipient.environment_state, recipient_environment_before)
    assert interventions._json_values_exactly_equal(donor.environment_state, donor_environment_before)
    assert np.array_equal(recipient.observation, recipient_observation_before)
    assert np.array_equal(donor.observation, donor_observation_before)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mismatch_field", ["actions", "success"])
def test_patch_runner_rejects_donor_natural_hc_control_mismatch(monkeypatch, mismatch_field):
    net, config = _tiny_net(horizon=4)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)
    recipient = _patch_context(net, puzzle, env_config, rollout_id=0, history=(0,), fill=0.25)
    donor = _patch_context(net, puzzle, env_config, rollout_id=1, history=(1,), fill=0.75)

    monkeypatch.setattr(
        interventions,
        "_collect_stochastic_contexts",
        lambda *_args, **_kwargs: (
            [recipient, donor],
            [
                {"rollout_id": 0, "rollout_seed": 100, "moves": 1, "success": False},
                {"rollout_id": 1, "rollout_seed": 101, "moves": 1, "success": False},
            ],
        ),
    )
    real_suffix = interventions._run_deterministic_suffix

    def corrupt_donor_natural(*args, environment_state, carry, **kwargs):
        result = real_suffix(*args, environment_state=environment_state, carry=carry, **kwargs)
        if environment_state is donor.environment_state and carry is donor.carry:
            result = dict(result)
            if mismatch_field == "actions":
                result["actions"] = [int(result["actions"][0]) + 1, *result["actions"][1:]]
            else:
                result["success"] = not bool(result["success"])
        return result

    monkeypatch.setattr(interventions, "_run_deterministic_suffix", corrupt_donor_natural)
    with pytest.raises(AssertionError, match="Donor-natural suffix.*transplant differ"):
        evaluate_matched_patches(
            net,
            [puzzle],
            env_config,
            patch_rollouts=2,
            bootstrap_reps=8,
        )


def test_patch_disabled_and_enabled_no_match_are_not_reported_as_zero_effects(monkeypatch):
    net, config = _tiny_net(horizon=3)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)

    progress = []
    disabled = evaluate_matched_patches(net, [puzzle], env_config, patch_rollouts=0, progress=progress.append)
    assert disabled["status"] == "disabled"
    assert disabled["eligibility"]["eligible_puzzles"] is None
    assert disabled["donor_natural_control"]["status"] == "not_evaluated"
    assert disabled["patches"]["hc"]["effects"] is None
    assert progress == ["patch phase disabled (patch_rollouts=0)"]

    monkeypatch.setattr(
        interventions,
        "_collect_stochastic_contexts",
        lambda *_args, **_kwargs: (
            [],
            [
                {"rollout_id": 0, "rollout_seed": 1, "moves": 2, "success": False, "action_history_sha256": "x"},
                {"rollout_id": 1, "rollout_seed": 2, "moves": 2, "success": True, "action_history_sha256": "y"},
            ],
        ),
    )
    no_match = evaluate_matched_patches(
        net,
        [puzzle],
        env_config,
        patch_rollouts=2,
        bootstrap_reps=8,
    )
    assert no_match["status"] == "enabled_no_matches"
    assert no_match["eligibility"]["eligible_puzzles"] == 0
    assert no_match["eligibility"]["no_match_puzzles"] == 1
    assert no_match["donor_natural_control"]["status"] == "no_eligible_contexts"
    assert no_match["patches"]["hc"]["effects"]["overall"]["n"] == 0
    assert no_match["patches"]["hc"]["effects"]["overall"]["mean_paired_success_delta"] is None


def test_rng_streams_and_model_mode_are_preserved_for_core_and_stochastic_patching():
    net, config = _tiny_net(horizon=2)
    puzzle = _cycling_puzzle()
    env_config = _env_config(config)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    net.train(False)

    evaluate_core_interventions(
        net,
        [puzzle],
        env_config,
        timing_moves=(1, 2),
        bootstrap_seed=7,
        bootstrap_reps=8,
    )
    evaluate_matched_patches(
        net,
        [puzzle],
        env_config,
        patch_rollouts=2,
        patch_seed=8,
        bootstrap_seed=9,
        bootstrap_reps=8,
    )
    assert net.training is False
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert _numpy_rng_equal(np.random.get_state(), numpy_before)

    net.train(True)
    evaluate_core_interventions(net, [puzzle], env_config, timing_moves=(1,), bootstrap_reps=4)
    assert net.training is True


def test_model_mode_is_restored_when_patch_collection_fails(monkeypatch):
    net, config = _tiny_net(horizon=2)
    net.train(False)

    def fail_collection(*_args, **_kwargs):
        raise RuntimeError("controlled collection failure")

    monkeypatch.setattr(interventions, "_collect_stochastic_contexts", fail_collection)
    with pytest.raises(RuntimeError, match="controlled collection failure"):
        evaluate_matched_patches(
            net,
            [_cycling_puzzle()],
            _env_config(config),
            patch_rollouts=2,
            bootstrap_reps=4,
        )
    assert net.training is False


@pytest.mark.parametrize("initial_training", [True, False])
@pytest.mark.parametrize(
    ("failure_point", "failure_marker"),
    [
        ("start", "core phase started"),
        ("progress", "core puzzle-arm evaluations"),
        ("end", "core phase complete"),
    ],
)
def test_core_progress_callback_failures_restore_model_mode(initial_training, failure_point, failure_marker):
    net, config = _tiny_net(horizon=1)
    net.train(initial_training)
    messages = []

    def failing_progress(message):
        messages.append(message)
        if failure_marker in message:
            raise RuntimeError(f"controlled {failure_point} callback failure")

    with pytest.raises(RuntimeError, match=f"controlled {failure_point} callback failure"):
        evaluate_core_interventions(
            net,
            [_cycling_puzzle()],
            _env_config(config),
            timing_moves=(1,),
            bootstrap_reps=4,
            progress=failing_progress,
        )

    assert any(failure_marker in message for message in messages)
    assert net.training is initial_training


@pytest.mark.parametrize("initial_training", [True, False])
@pytest.mark.parametrize(
    ("failure_point", "failure_marker"),
    [
        ("start", "patch phase started"),
        ("progress", "patch puzzles"),
        ("end", "patch phase complete"),
    ],
)
def test_patch_progress_callback_failures_restore_model_mode(
    monkeypatch,
    initial_training,
    failure_point,
    failure_marker,
):
    net, config = _tiny_net(horizon=1)
    net.train(initial_training)
    messages = []
    monkeypatch.setattr(interventions, "_collect_stochastic_contexts", lambda *_args, **_kwargs: ([], []))

    def failing_progress(message):
        messages.append(message)
        if failure_marker in message:
            raise RuntimeError(f"controlled {failure_point} callback failure")

    with pytest.raises(RuntimeError, match=f"controlled {failure_point} callback failure"):
        evaluate_matched_patches(
            net,
            [_cycling_puzzle()],
            _env_config(config),
            patch_rollouts=2,
            bootstrap_reps=4,
            progress=failing_progress,
        )

    assert any(failure_marker in message for message in messages)
    assert net.training is initial_training


def test_cli_rejects_test_split_and_nonpositive_suite_and_exposes_disable_switch():
    parser = interventions._build_argument_parser()
    args = parser.parse_args(
        ["--model", "model.pt", "--bank", "bank.parquet", "--output", "result.json", "--patch-rollouts", "0"]
    )
    assert args.split_name == "val"
    assert args.per_stratum > 0
    assert args.patch_rollouts == 0
    assert args.quiet_progress is False
    assert tuple(args.timing_moves) == DEFAULT_TIMING_MOVES
    quiet_args = parser.parse_args(
        [
            "--model",
            "model.pt",
            "--bank",
            "bank.parquet",
            "--output",
            "result.json",
            "--quiet-progress",
        ]
    )
    assert quiet_args.quiet_progress is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "model.pt",
                "--bank",
                "bank.parquet",
                "--output",
                "result.json",
                "--split",
                "test",
            ]
        )
    with pytest.raises(ValueError, match="restricted"):
        intervene_checkpoint("missing.pt", "missing.parquet", split_name="test")
    with pytest.raises(ValueError, match="positive"):
        intervene_checkpoint("missing.pt", "missing.parquet", per_stratum=0)
    with pytest.raises(ValueError, match="at least 2"):
        intervene_checkpoint("missing.pt", "missing.parquet", patch_rollouts=1)


def test_atomic_output_refuses_overwrite_force_replaces_and_failures_preserve_old_file(tmp_path):
    path = tmp_path / "nested" / "result.json"
    write_json_atomic(path, {"version": 1})
    assert json.loads(path.read_text()) == {"version": 1}
    with pytest.raises(FileExistsError, match="--force"):
        write_json_atomic(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}
    write_json_atomic(path, {"version": 2}, force=True)
    assert json.loads(path.read_text()) == {"version": 2}

    with pytest.raises(ValueError):
        write_json_atomic(path, {"bad": float("nan")}, force=True)
    assert json.loads(path.read_text()) == {"version": 2}
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_cli_progress_uses_stderr_preserves_concise_stdout_and_can_be_quiet(tmp_path, monkeypatch, capsys):
    callbacks = []

    def fake_intervene_checkpoint(*_args, progress=None, **_kwargs):
        callbacks.append(progress)
        if progress is not None:
            progress("core puzzle-arm evaluations: 1/1 puzzle evaluations")
        return {
            "suite": {"count": 1},
            "core": {"baseline": {"success_rate": 0.5}},
            "patching": {"status": "disabled"},
        }

    monkeypatch.setattr(interventions, "intervene_checkpoint", fake_intervene_checkpoint)
    first_output = tmp_path / "with-progress.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intervene_drc_carry",
            "--model",
            "model.pt",
            "--bank",
            "bank.parquet",
            "--output",
            str(first_output),
        ],
    )
    interventions.main()
    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert stdout_payload == {
        "baseline_success_rate": 0.5,
        "output": str(first_output.resolve()),
        "patch_status": "disabled",
        "suite_puzzles": 1,
    }
    assert captured.out.count("\n") == 1
    assert "[drc-carry] core puzzle-arm evaluations: 1/1" in captured.err
    assert "[drc-carry] publishing atomic result" in captured.err
    assert "[drc-carry] run complete" in captured.err
    assert callable(callbacks[0])
    assert json.loads(first_output.read_text()) == {
        "core": {"baseline": {"success_rate": 0.5}},
        "patching": {"status": "disabled"},
        "suite": {"count": 1},
    }

    quiet_output = tmp_path / "quiet.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intervene_drc_carry",
            "--model",
            "model.pt",
            "--bank",
            "bank.parquet",
            "--output",
            str(quiet_output),
            "--quiet-progress",
        ],
    )
    interventions.main()
    quiet_captured = capsys.readouterr()
    assert json.loads(quiet_captured.out)["output"] == str(quiet_output.resolve())
    assert quiet_captured.err == ""
    assert callbacks[1] is None


def test_checkpoint_runner_records_complete_provenance_and_stable_scientific_payload(tmp_path, monkeypatch):
    net, config = _tiny_net(seed=29, horizon=3)
    checkpoint_path = tmp_path / "origin" / "model_1234.pt"
    checkpoint_path.parent.mkdir()
    _save_model(
        net,
        config,
        checkpoint_path,
        metadata={"timesteps": 1234, "kind": "periodic_snapshot"},
    )
    bank_path = tmp_path / "bank.parquet"
    bank_path.write_bytes(b"controlled-test-bank")
    puzzles = [
        _two_way_puzzle(layout_hash="layout-a", seed=1),
        _two_way_puzzle(layout_hash="layout-b", seed=2),
    ]
    monkeypatch.setattr(interventions, "load_puzzles", lambda *_args, **_kwargs: puzzles)

    kwargs = {
        "split_name": "val",
        "suite_seed": 11,
        "per_stratum": 1,
        "timing_moves": (1, 2),
        "rollout_seed": 3,
        "bootstrap_seed": 4,
        "bootstrap_reps": 8,
        "patch_rollouts": 0,
        "patch_seed": 5,
        "device": "cpu",
    }
    first = intervene_checkpoint(checkpoint_path, bank_path, **kwargs)
    second = intervene_checkpoint(checkpoint_path, bank_path, **kwargs)
    assert first["schema_version"] == 2
    assert first["protocol_version"] == "drc-carry-causal-interventions-v2"
    assert first["checkpoint"]["absolute_path"] == str(checkpoint_path.resolve())
    assert len(first["checkpoint"]["sha256"]) == 64
    assert first["checkpoint"]["originating_run"] == config.run_name
    assert first["checkpoint"]["timesteps"] == 1234
    assert first["checkpoint"]["checkpoint_kind"] == "periodic_snapshot"
    assert first["checkpoint"]["stored_config"]["horizon"] == 3
    assert len(first["checkpoint"]["config_fingerprint"]) == 64
    assert len(first["checkpoint"]["architecture_fingerprint"]) == 64
    assert first["bank"]["absolute_path"] == str(bank_path.resolve())
    assert first["bank"]["split_contract"]["split_seed"] == config.split_seed
    assert first["suite"]["source_split"] == "val"
    assert first["suite"]["per_stratum"] == 1
    assert first["suite"]["count"] == 1
    assert first["suite"]["per_cell_counts"] == {"2|1": 1}
    assert first["settings"]["bootstrap_unit"] == "puzzle"
    assert first["inference"]["status"] == "estimation_only_with_unadjusted_pointwise_intervals"
    interval_contract = first["inference"]["interval_contract"]
    assert interval_contract["coverage_scope"] == "pointwise_only"
    assert interval_contract["multiplicity_adjustment"] == "none"
    assert interval_contract["family_wise_coverage"] is False
    assert interval_contract["simultaneous_coverage"] is False
    localization_role = first["inference"]["role_definitions"][interventions.LOCALIZATION_INFERENCE_ROLE]
    assert localization_role["descriptive_only"] is True
    assert localization_role["hypothesis_generating"] is True
    assert localization_role["confirmatory"] is False
    assert (
        first["inference"]["claim_policy"]["localization_or_subgroup_claim_from_interval_excluding_zero_allowed"]
        is False
    )
    assert first["patching"]["status"] == "disabled"
    assert "full_reset_every_move" in first["localization"]
    assert set(first["localization"]["per_depth"]) == {"0", "1"}
    full_reset = first["core"]["interventions"]["reset_all_every_move"]["effects"]
    assert full_reset["estimand_role"] == "all_selected_puzzles_itt_estimand"
    assert full_reset["overall"]["bootstrap_ci"]["inference_role"] == (interventions.PRIMARY_OVERALL_INFERENCE_ROLE)
    assert full_reset["by_optimal_length"]["2"]["bootstrap_ci"]["inference_role"] == (
        interventions.SUBGROUP_INFERENCE_ROLE
    )
    depth_zero = first["core"]["interventions"]["reset_depth_0_every_move"]["effects"]
    assert depth_zero["overall"]["bootstrap_ci"]["inference_role"] == (interventions.LOCALIZATION_INFERENCE_ROLE)
    timing_one = first["core"]["interventions"]["reset_once_after_1_moves"]["effects"]
    assert timing_one["estimand_role"] == "all_selected_puzzles_itt_estimand"
    assert timing_one["overall"]["eligibility_denominator_n"] == 1
    assert timing_one["overall"]["n_eligible"] == 1
    assert timing_one["overall"]["n_applied"] == 1
    assert timing_one["overall"]["bootstrap_ci"]["inference_role"] == (interventions.LOCALIZATION_INFERENCE_ROLE)
    timing_eligible = timing_one["eligible_puzzles_descriptive_effects"]
    assert timing_eligible["subset"]["is_randomized_or_itt_estimand"] is False
    assert timing_eligible["overall"]["bootstrap_ci"]["inference_role"] == (
        interventions.ELIGIBLE_SUBSET_INFERENCE_ROLE
    )
    assert first["localization"]["timing"]["1"]["n_eligible"] == 1
    assert (
        first["localization"]["timing"]["1"]["eligible_puzzles_descriptive_effect"]["estimand_role"]
        == "secondary_descriptive_principal_stratum_not_randomized_or_itt"
    )
    assert set(first["git"]["implementation_file_sha256"]) >= {
        "src/groundup/intervene_drc_carry.py",
        "src/groundup/probe_drc_carry.py",
    }
    json.dumps(first, allow_nan=False)

    stable_first = {key: value for key, value in first.items() if key not in {"runtime", "git"}}
    stable_second = {key: value for key, value in second.items() if key not in {"runtime", "git"}}
    assert stable_first == stable_second
