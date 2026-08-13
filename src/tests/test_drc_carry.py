from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest
import torch

from src.groundup import probe_drc_carry as probe_module
from src.groundup.drc import DRCNet, DRCState
from src.groundup.drc_carry import (
    BOS_ACTION,
    CAPTURE_PROTOCOL_VERSION,
    DATASET_SCHEMA_VERSION,
    ExactStateOracle,
    SolverContract,
    capture_carry_trajectories,
    load_carry_dataset,
    save_carry_dataset,
    validate_carry_dataset,
)
from src.groundup.drc_train import DRCTrainConfig, _save_model, build_net
from src.groundup.env import EnvConfig, RicochetEnv
from src.groundup.probe_drc_carry import (
    DEFAULT_MAX_CARRY_BYTES,
    PROBE_PROTOCOL_VERSION,
    RESULT_SCHEMA_VERSION,
    ProbeTarget,
    _atomic_json,
    _build_argument_parser,
    _result_content_fingerprint,
    _with_result_content_fingerprint,
    build_random_control,
    carry_payload_preflight,
    fit_probe_family,
    load_or_capture_dataset,
    load_schema2_checkpoint,
    run_controlled_probes,
    shuffle_training_labels,
    stable_layout_split,
    suite_provenance,
)
from src.groundup.puzzles import Puzzle


def _open_board(height: int = 3, width: int = 3) -> tuple[np.ndarray, np.ndarray]:
    h_walls = np.zeros((height + 1, width), dtype=bool)
    v_walls = np.zeros((height, width + 1), dtype=bool)
    h_walls[[0, -1], :] = True
    v_walls[:, [0, -1]] = True
    return h_walls, v_walls


def _two_way_puzzle(*, layout_hash: str = "two-way") -> Puzzle:
    """The target can solve UP->LEFT or LEFT->UP, both in two moves."""
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
        seed=7,
    )


def _tiny_net(seed: int = 4) -> tuple[DRCNet, DRCTrainConfig]:
    config = DRCTrainConfig(
        seed=seed,
        horizon=4,
        shaping_coef=0.0,
        drc_depth=1,
        drc_ticks=2,
        drc_hidden_channels=3,
        drc_encoder_channels=(3,),
        drc_encoder_kernels=(3,),
        drc_encoder_strides=(1,),
        drc_mlp_hidden=12,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = build_net(config, (7, 3, 3), 4)
    # A state-independent priority gives the exact solving path UP, LEFT while
    # leaving the recurrent stack nontrivial for carry-threading assertions.
    with torch.no_grad():
        net.actor.weight.zero_()
        net.actor.bias.copy_(torch.tensor([3.0, 0.0, 2.0, 1.0]))
    return net, config


def test_exact_solver_unions_all_optimal_first_actions_and_caps_are_unknown():
    puzzle = _two_way_puzzle()
    oracle = ExactStateOracle(puzzle, SolverContract(max_depth=8, max_nodes=10_000))

    labels = oracle.label(puzzle.robots)

    assert labels.solve.status == "known"
    assert labels.solve.solvable is True
    assert labels.solve.distance == 2
    assert np.flatnonzero(labels.optimal_action_mask).tolist() == [0, 2]
    assert labels.reducing_known
    assert np.array_equal(labels.reducing_action_mask, labels.optimal_action_mask)

    goal_labels = oracle.label(((0, 0),))
    assert goal_labels.solve.status == "known"
    assert goal_labels.solve.distance == 0
    assert not goal_labels.optimal_action_mask.any()
    assert goal_labels.reducing_known
    assert not goal_labels.reducing_action_mask.any()

    depth_capped = ExactStateOracle(puzzle, SolverContract(max_depth=1, max_nodes=10_000)).solve(puzzle.robots)
    node_capped = ExactStateOracle(puzzle, SolverContract(max_depth=8, max_nodes=1)).solve(puzzle.robots)
    for result, reason in ((depth_capped, "max_depth"), (node_capped, "max_nodes")):
        assert result.status == "unknown"
        assert result.solvable is None
        assert result.distance is None
        assert result.censor_reason == reason
        assert result.optimal_action_bits == 0

    h_walls, v_walls = _open_board()
    truly_unsolvable = Puzzle(
        h_walls=h_walls,
        v_walls=v_walls,
        robots=((0, 0),),
        goal=(1, 1),  # a lone robot can stop only at a boundary on this board
        target=0,
        optimal_length=1,
        actions=((0, 1),),
        robots_moved=1,
        layout_hash="exhausted-unsolvable",
        seed=8,
    )
    exhausted = ExactStateOracle(truly_unsolvable, SolverContract(max_depth=8, max_nodes=1_000)).solve(
        truly_unsolvable.robots
    )
    assert exhausted.status == "known"
    assert exhausted.solvable is False
    assert exhausted.distance is None
    assert exhausted.censor_reason is None


def test_solver_transitions_and_masks_match_canonical_environment():
    h_walls, v_walls = _open_board(4, 4)
    h_walls[2, 1] = True
    v_walls[1, 3] = True
    puzzle = Puzzle(
        h_walls=h_walls,
        v_walls=v_walls,
        robots=((3, 3), (1, 1), (0, 2)),
        goal=(0, 0),
        target=1,
        optimal_length=2,
        actions=((1, 0), (1, 2)),
        robots_moved=1,
        layout_hash="canonical-parity",
        seed=9,
    )
    env = RicochetEnv([puzzle], EnvConfig(canonical_identity=True, shaping_coef=0.0, horizon=8))
    env.reset(options={"puzzle_index": 0})
    oracle = ExactStateOracle(puzzle, SolverContract(max_depth=4, max_nodes=10_000))

    positions = tuple(env._robots)
    assert positions[0] == puzzle.robots[puzzle.target]
    assert np.array_equal(oracle.legal_action_mask(positions), env.action_masks())
    for action in range(env.action_space.n):
        expected = list(positions)
        rid, direction = divmod(action, 4)
        expected[rid] = env._slide_from(expected, rid, direction, puzzle)
        assert oracle.transition(positions, action) == tuple(expected)
    non_target_action = next(int(action) for action in np.flatnonzero(env.action_masks()) if action // 4 > 0)
    expected_after_step = oracle.transition(positions, non_target_action)
    env.step(non_target_action)
    assert tuple(env._robots) == expected_after_step
    assert np.array_equal(oracle.legal_action_mask(expected_after_step), env.action_masks())


def test_capture_threads_exact_forward_carries_and_backfills_labels():
    puzzle = _two_way_puzzle()
    net, config = _tiny_net()
    random_net = build_random_control(config, net.observation_shape, net.n_actions, seed=123)
    progress_messages: list[str] = []
    arrays, metadata = capture_carry_trajectories(
        net,
        random_net,
        [puzzle],
        EnvConfig(
            horizon=4,
            shaping_coef=0.0,
            time_channel=True,
            canonical_identity=True,
        ),
        solver_contract=SolverContract(max_depth=8, max_nodes=10_000),
        rollout_seed=17,
        recent_action_lags=3,
        carry_dtype=np.float32,
        progress=progress_messages.append,
    )

    assert arrays["chosen_action"].tolist() == [0, 2]
    assert arrays["future_success"].tolist() == [True, True]
    assert arrays["immediate_success"].tolist() == [False, True]
    assert arrays["landing_cell"].tolist() == [1, 0]
    assert arrays["slide_length"].tolist() == [1, 1]
    assert arrays["moved"].tolist() == [True, True]
    assert arrays["recent_actions"][:, 0].tolist() == [BOS_ACTION(4), 0]
    assert arrays["recent_actions"].tolist() == [[4, 5, 5], [0, 4, 5]]
    assert arrays["move_index"].tolist() == [0, 1]
    assert np.allclose(arrays["horizon_progress"], [0.0, 0.25])
    assert np.all(arrays["exact_label_known"])
    assert arrays["exact_distance"].tolist() == [2, 1]

    assert np.count_nonzero(arrays["trained_carry_in_h"][0]) == 0
    assert np.array_equal(arrays["trained_carry_in_h"][1], arrays["trained_carry_out_h"][0])
    assert np.array_equal(arrays["trained_carry_in_c"][1], arrays["trained_carry_out_c"][0])
    with torch.no_grad():
        obs = torch.as_tensor(arrays["obs"][0:1])
        state = DRCState(
            h=torch.as_tensor(arrays["trained_carry_in_h"][0:1]).movedim(0, 1),
            c=torch.as_tensor(arrays["trained_carry_in_c"][0:1]).movedim(0, 1),
        )
        mask = torch.as_tensor(arrays["action_mask"][0:1])
        logits, _value, state_out = net(obs, state, action_mask=mask)
    assert np.array_equal(logits.numpy()[0], arrays["policy_logits"][0])
    assert int(logits.argmax(-1)) == arrays["chosen_action"][0]
    assert np.allclose(state_out.h[:, 0].numpy(), arrays["trained_carry_out_h"][0])
    assert metadata["episodes"]["successes"] == 1
    assert metadata["schema_version"] == DATASET_SCHEMA_VERSION == 2
    assert metadata["capture_protocol"] == CAPTURE_PROTOCOL_VERSION == "drc-carry-trajectory-v2"
    assert progress_messages == ["Capturing puzzle 1/1"]
    assert metadata["solver_coverage"]["exact_known_rows"] == 2
    assert metadata["solver_coverage"]["exact_fraction"] == 1.0
    assert metadata["solver_coverage"]["root_nodes_discovered"] > 0
    assert metadata["solver_coverage"]["censor_reasons"] == {
        "max_depth": 0,
        "max_nodes": 0,
    }
    validate_carry_dataset(arrays, metadata)

    failed, failed_metadata = capture_carry_trajectories(
        net,
        random_net,
        [puzzle],
        EnvConfig(horizon=1, shaping_coef=0.0),
        solver_contract=SolverContract(max_depth=8, max_nodes=10_000),
        carry_dtype=np.float32,
    )
    assert failed["chosen_action"].tolist() == [0]
    assert failed["future_success"].tolist() == [False]
    assert failed["immediate_success"].tolist() == [False]
    assert failed_metadata["episodes"] == {"successes": 0, "failures": 1}

    censored, censored_metadata = capture_carry_trajectories(
        net,
        random_net,
        [puzzle],
        EnvConfig(horizon=4, shaping_coef=0.0),
        solver_contract=SolverContract(max_depth=1, max_nodes=10_000),
        carry_dtype=np.float32,
    )
    assert len(censored["chosen_action"]) == 2  # censored row was retained
    assert censored["exact_label_known"].tolist() == [False, True]
    assert np.isnan(censored["exact_distance"][0])
    assert censored["exact_distance"][1] == 1
    assert censored_metadata["solver_coverage"]["exact_known_rows"] == 1
    assert censored_metadata["solver_coverage"]["exact_total_rows"] == 2
    assert censored_metadata["solver_coverage"]["censor_reasons"]["max_depth"] == 1


def test_random_control_is_matched_reproducible_and_cannot_choose_actions():
    puzzle = _two_way_puzzle()
    net, config = _tiny_net()
    rng_before = torch.random.get_rng_state().clone()
    random_a = build_random_control(config, net.observation_shape, net.n_actions, seed=88)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    random_b = build_random_control(config, net.observation_shape, net.n_actions, seed=88)
    assert list(net.state_dict()) == list(random_a.state_dict())
    assert all(
        left.shape == right.shape for left, right in zip(net.state_dict().values(), random_a.state_dict().values())
    )
    assert all(
        torch.equal(left, right) for left, right in zip(random_a.state_dict().values(), random_b.state_dict().values())
    )
    # Make the random actor strongly prefer a different legal action. Capture
    # must still step only the trained actor's actions.
    with torch.no_grad():
        random_a.actor.weight.zero_()
        random_a.actor.bias.copy_(torch.tensor([0.0, 9.0, 0.0, 0.0]))
        random_b.load_state_dict(random_a.state_dict())

    kwargs = dict(
        puzzles=[puzzle],
        env_config=EnvConfig(horizon=4, shaping_coef=0.0),
        solver_contract=SolverContract(max_depth=8, max_nodes=10_000),
        rollout_seed=3,
        carry_dtype=np.float32,
    )
    random_observations: list[np.ndarray] = []
    hook = random_a.register_forward_pre_hook(
        lambda _module, inputs: random_observations.append(inputs[0].detach().cpu().numpy()[0].copy())
    )
    first, first_meta = capture_carry_trajectories(net, random_a, **kwargs)
    hook.remove()
    second, _ = capture_carry_trajectories(net, random_b, **kwargs)
    assert first["chosen_action"].tolist() == [0, 2]
    assert np.array_equal(first["obs"], second["obs"])
    assert np.array_equal(first["chosen_action"], second["chosen_action"])
    assert np.array_equal(first["random_carry_out_h"], second["random_carry_out_h"])
    assert np.array_equal(first["random_carry_out_c"], second["random_carry_out_c"])
    assert np.array_equal(np.asarray(random_observations), first["obs"])
    assert first_meta["random_control"]["chooses_actions"] is False
    assert first_meta["random_control"]["trajectory_source"] == "trained_policy"


def test_schema2_checkpoint_loading_is_strict_and_records_full_provenance(tmp_path):
    net, config = _tiny_net(seed=29)
    checkpoint = tmp_path / "origin" / "model_1234.pt"
    checkpoint.parent.mkdir()
    _save_model(
        net,
        config,
        checkpoint,
        metadata={"timesteps": 1234, "kind": "periodic_snapshot"},
    )
    rng_before = torch.random.get_rng_state().clone()
    loaded, loaded_config, payload, provenance = load_schema2_checkpoint(checkpoint, device="cpu")
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert loaded_config == config
    assert payload["schema_version"] == 2
    assert all(torch.equal(loaded.state_dict()[name], value) for name, value in net.state_dict().items())
    assert provenance["absolute_path"] == str(checkpoint.resolve())
    assert len(provenance["sha256"]) == 64
    assert provenance["originating_run"] == config.run_name
    assert provenance["timesteps"] == 1234
    assert provenance["checkpoint_kind"] == "periodic_snapshot"
    assert len(provenance["config_fingerprint"]) == 64
    assert len(provenance["architecture_fingerprint"]) == 64

    forged = dict(payload)
    forged["config_fingerprint"] = "0" * 64
    forged_path = tmp_path / "forged.pt"
    torch.save(forged, forged_path)
    with pytest.raises(ValueError, match="config_fingerprint"):
        load_schema2_checkpoint(forged_path)

    legacy_path = tmp_path / "legacy.pt"
    torch.save({**payload, "schema_version": 1}, legacy_path)
    with pytest.raises(ValueError, match="supported schema-2"):
        load_schema2_checkpoint(legacy_path)


def test_cli_maps_required_paths_to_orchestrator_interface():
    assert RESULT_SCHEMA_VERSION == 2
    assert PROBE_PROTOCOL_VERSION == "drc-carry-controlled-probes-v2"
    args = _build_argument_parser().parse_args(
        [
            "--model",
            "model.pt",
            "--bank",
            "bank.parquet",
            "--dataset",
            "carry.npz",
            "--output",
            "result.json",
        ]
    )
    arguments = vars(args).copy()
    assert arguments.pop("output") == "result.json"
    assert arguments["model_path"] == "model.pt"
    assert arguments["bank_path"] == "bank.parquet"
    assert arguments["dataset_path"] == "carry.npz"
    assert arguments["split_name"] == "val"
    assert arguments["max_carry_bytes"] == DEFAULT_MAX_CARRY_BYTES
    assert arguments["allow_large_dataset"] is False
    assert arguments.pop("print_json") is False
    assert "model" not in arguments and "bank" not in arguments and "dataset" not in arguments


def test_layout_split_has_no_leakage_and_is_row_order_stable():
    layouts = np.asarray(["a", "a", "b", "c", "c", "d", "e", "e"])
    first = stable_layout_split(layouts, seed=19, test_fraction=0.4, min_layouts_per_partition=1)
    order = np.asarray([7, 2, 5, 0, 3, 1, 6, 4])
    second = stable_layout_split(layouts[order], seed=19, test_fraction=0.4, min_layouts_per_partition=1)

    train_layouts = set(layouts[first.train_indices])
    test_layouts = set(layouts[first.test_indices])
    assert train_layouts.isdisjoint(test_layouts)
    assert train_layouts == set(layouts[order][second.train_indices])
    assert test_layouts == set(layouts[order][second.test_indices])
    assert first.fingerprint == second.fingerprint
    with pytest.raises(ValueError, match="at least two distinct layouts"):
        stable_layout_split(np.asarray(["only", "only"]), seed=1)


def _minimal_dataset() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    puzzle = _two_way_puzzle()
    net, config = _tiny_net()
    random_net = build_random_control(config, net.observation_shape, net.n_actions, seed=5)
    arrays, capture = capture_carry_trajectories(
        net,
        random_net,
        [puzzle],
        EnvConfig(horizon=4, shaping_coef=0.0),
        solver_contract=SolverContract(max_depth=8, max_nodes=10_000),
        carry_dtype=np.float32,
    )
    metadata = {"schema_version": 1, "contract": {"checkpoint_sha256": "abc"}, **capture}
    return arrays, metadata


def test_saved_dataset_fingerprint_and_contract_reject_mutation(tmp_path):
    arrays, metadata = _minimal_dataset()
    with pytest.raises(ValueError, match="Unsupported carry dataset schema"):
        validate_carry_dataset(arrays, {**metadata, "schema_version": 1})
    with pytest.raises(ValueError, match="Unsupported carry capture protocol"):
        validate_carry_dataset(arrays, {**metadata, "capture_protocol": "drc-carry-trajectory-v1"})
    path = tmp_path / "carry.npz"
    saved = save_carry_dataset(path, arrays, metadata)
    loaded, loaded_meta = load_carry_dataset(path, expected_contract={"checkpoint_sha256": "abc"})
    assert loaded_meta["content_sha256"] == saved["content_sha256"]
    assert np.array_equal(loaded["chosen_action"], arrays["chosen_action"])
    with pytest.raises(ValueError, match="contract mismatch"):
        load_carry_dataset(path, expected_contract={"checkpoint_sha256": "different"})

    with np.load(path, allow_pickle=False) as payload:
        corrupt = {name: payload[name].copy() for name in payload.files if name != "metadata"}
        raw_meta = json.loads(str(payload["metadata"]))
    corrupt["chosen_action"][0] = 3
    np.savez_compressed(
        path,
        **corrupt,
        metadata=np.asarray(json.dumps(raw_meta, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_carry_dataset(path)


def test_dataset_reuse_binds_realized_random_control_weights(tmp_path):
    puzzle = _two_way_puzzle()
    net, config = _tiny_net()
    random_net = build_random_control(config, net.observation_shape, net.n_actions, seed=41)
    dataset = tmp_path / "bound-random-control.npz"
    suite = suite_provenance([puzzle], split="val", seed=3, per_stratum=1)
    kwargs = {
        "path": dataset,
        "net": net,
        "puzzles": [puzzle],
        "env_config": EnvConfig(horizon=4, shaping_coef=0.0),
        "checkpoint_provenance": {
            "absolute_path": "/checkpoint.pt",
            "sha256": "1" * 64,
            "config_fingerprint": "2" * 64,
            "architecture_fingerprint": "3" * 64,
        },
        "bank_provenance": {"absolute_path": "/bank.parquet", "sha256": "4" * 64},
        "suite": suite,
        "solver_contract": SolverContract(max_depth=8, max_nodes=10_000),
        "rollout_seed": 0,
        "random_network_seed": 41,
        "recent_action_lags": 3,
        "carry_dtype": "float32",
        "force": False,
    }

    _arrays, first_metadata = load_or_capture_dataset(random_net=random_net, **kwargs)
    exact_clone = build_random_control(config, net.observation_shape, net.n_actions, seed=41)
    _reused, reused_metadata = load_or_capture_dataset(random_net=exact_clone, **kwargs)

    contract_control = first_metadata["contract"]["random_control"]
    assert contract_control["state_dict_sha256"] == first_metadata["random_control"]["state_dict_sha256"]
    assert contract_control["architecture_fingerprint"] == first_metadata["random_control"]["architecture_fingerprint"]
    assert contract_control["architecture_signature"] == first_metadata["random_control"]["architecture_signature"]
    assert reused_metadata["content_sha256"] == first_metadata["content_sha256"]

    same_seed_different_weights = build_random_control(config, net.observation_shape, net.n_actions, seed=41)
    with torch.no_grad():
        first_parameter = next(same_seed_different_weights.parameters())
        first_parameter.view(-1)[0].add_(1.0)
    with pytest.raises(ValueError, match=r"contract\.random_control\.state_dict_sha256"):
        load_or_capture_dataset(random_net=same_seed_different_weights, **kwargs)


def test_shuffled_control_changes_only_training_labels():
    values = np.arange(12, dtype=np.int64)
    applicable = np.ones(12, dtype=bool)
    targets = {
        "one": ProbeTarget("one", "multiclass", values, applicable, 12, "solo"),
        "action": ProbeTarget("action", "multiclass", values.copy(), applicable, 12, "action_tuple"),
        "robot": ProbeTarget("robot", "multiclass", (values % 3), applicable, 3, "action_tuple"),
    }
    train = np.arange(8)
    test = np.arange(8, 12)
    original = {name: target.values.copy() for name, target in targets.items()}
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    features_before = features.copy()
    train_before, test_before = train.copy(), test.copy()

    shuffled, audit = shuffle_training_labels(targets, train, seed=91)

    assert np.array_equal(features, features_before)
    assert np.array_equal(train, train_before)
    assert np.array_equal(test, test_before)
    for name in targets:
        assert np.array_equal(shuffled[name].values[test], original[name][test])
    assert not np.array_equal(shuffled["one"].values[train], original["one"][train])
    # Structured intended-action targets share an exact row permutation.
    assert np.array_equal(shuffled["robot"].values[train], shuffled["action"].values[train] % 3)
    assert audit["test_labels_untouched"] is True


def test_controlled_probe_output_is_explicitly_nested_and_nulls_are_explained():
    puzzles = [_two_way_puzzle(layout_hash=f"layout-{index}") for index in range(4)]
    net, config = _tiny_net()
    random_net = build_random_control(config, net.observation_shape, net.n_actions, seed=77)
    arrays, metadata = capture_carry_trajectories(
        net,
        random_net,
        puzzles,
        EnvConfig(horizon=4, shaping_coef=0.0),
        solver_contract=SolverContract(max_depth=8, max_nodes=10_000),
        carry_dtype=np.float32,
    )
    split = stable_layout_split(
        arrays["layout_hash"],
        seed=4,
        test_fraction=0.5,
        min_layouts_per_partition=1,
    )
    results, shuffle_audit, fit_manifest = run_controlled_probes(
        arrays,
        metadata,
        split=split,
        probe_seed=3,
        shuffle_seed=5,
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        hidden_size=8,
        device="cpu",
    )
    assert set(results) == {"h_final", "c_final", "h_c_final"}
    for representation in results.values():
        assert set(representation) == {"linear", "mlp"}
        for family in representation.values():
            assert set(family) == {"trained", "random_network", "shuffled_labels"}
            assert "exact_distance" in family["trained"]
            assert "previous_action" in family["trained"]
            assert "landing_cell" in family["trained"]
    undefined = results["h_final"]["linear"]["trained"]["moved"]["metrics"]
    assert undefined["balanced_accuracy"]["value"] is None
    assert undefined["balanced_accuracy"]["reason"]
    assert undefined["roc_auc"]["value"] is None
    trained_target = results["h_final"]["mlp"]["trained"]["future_success"]
    shuffled_target = results["h_final"]["mlp"]["shuffled_labels"]["future_success"]
    assert trained_target["label_sources"] == {"train": "original_labels", "test": "original_labels"}
    assert shuffled_target["label_sources"] == {
        "train": "shuffled_training_labels",
        "test": "original_labels",
    }
    assert trained_target["train_metrics"]
    assert trained_target["target_summary"]["train"]["class_histogram"] == {"0": 0, "1": 4}
    assert trained_target["no_skill_baseline"]["strategy"] == "training_majority_class"
    diagnostics = trained_target["training_diagnostics"]
    assert diagnostics["loss_curve_summary"]["finite"] is True
    assert diagnostics["loss_curve_summary"]["epochs"] == 2
    assert diagnostics["loss_curve_summary"]["batches"] > 0
    assert len(diagnostics["initial_state_sha256"]) == 64
    assert len(diagnostics["final_state_sha256"]) == 64
    assert len(diagnostics["test_prediction_sha256"]) == 64
    paired = results["h_final"]["mlp"]
    for control in ("random_network", "shuffled_labels"):
        paired_diagnostics = paired[control]["future_success"]["training_diagnostics"]
        assert paired_diagnostics["seed"] == diagnostics["seed"]
        assert paired_diagnostics["initial_state_sha256"] == diagnostics["initial_state_sha256"]
        assert paired_diagnostics["minibatch_order_sha256"] == diagnostics["minibatch_order_sha256"]
    assert shuffle_audit["test_labels_untouched"] is True
    assert fit_manifest["h_final"]["linear"]["trained"]["seed"] is not None
    json.dumps(
        {"results": results, "shuffle": shuffle_audit, "fits": fit_manifest},
        allow_nan=False,
    )

    class_starved = dict(arrays)
    class_starved["future_success"] = arrays["future_success"].copy()
    class_starved["future_success"][split.train_indices] = True
    class_starved["future_success"][split.test_indices] = False
    with pytest.raises(ValueError, match="only 1 training class"):
        run_controlled_probes(
            class_starved,
            metadata,
            split=split,
            probe_seed=3,
            shuffle_seed=5,
            epochs=1,
            batch_size=8,
            learning_rate=1e-2,
            hidden_size=8,
            device="cpu",
        )


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_probe_families_learn_signal_and_shuffled_test_labels_stay_real(family):
    rng = np.random.default_rng(44)
    n = 240
    # Each layout owns two rows, preventing state-level leakage.
    layouts = np.repeat([f"layout-{i:03d}" for i in range(n // 2)], 2)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    target = ProbeTarget("binary_signal", "binary", y, np.ones(n, dtype=bool), 1, "binary_signal")
    targets = {target.name: target}
    split = stable_layout_split(layouts, seed=5, test_fraction=0.25, min_layouts_per_partition=2)
    kwargs = dict(
        family=family,
        split=split,
        epochs=50,
        batch_size=64,
        learning_rate=0.03,
        hidden_size=16,
        seed=8,
        device="cpu",
    )
    torch_rng_before = torch.random.get_rng_state().clone()
    trained = fit_probe_family(x, targets, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)
    shuffled_targets, _audit = shuffle_training_labels(targets, split.train_indices, seed=33)
    shuffled = fit_probe_family(x, shuffled_targets, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)

    trained_acc = trained["targets"]["binary_signal"]["metrics"]["accuracy"]["value"]
    shuffled_acc = shuffled["targets"]["binary_signal"]["metrics"]["accuracy"]["value"]
    assert trained_acc >= 0.9
    assert shuffled_acc < 0.75
    # The control object itself retains original labels in the test rows; a
    # deceptively high score against permuted test labels cannot be reported.
    assert np.array_equal(shuffled_targets["binary_signal"].values[split.test_indices], y[split.test_indices])


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_each_target_probe_is_fully_isolated_from_other_labels(family):
    rng = np.random.default_rng(812)
    n = 96
    features = rng.normal(size=(n, 5)).astype(np.float32)
    layouts = np.asarray([f"layout-{index // 2:03d}" for index in range(n)])
    split = stable_layout_split(layouts, seed=7, test_fraction=0.25, min_layouts_per_partition=2)
    applicable = np.ones(n, dtype=bool)
    target_b_values = (features[:, 1] - 0.25 * features[:, 2] > 0).astype(np.int64)
    first_targets = {
        "target_a": ProbeTarget(
            "target_a",
            "binary",
            (features[:, 0] > 0).astype(np.int64),
            applicable,
            1,
            "target_a",
        ),
        "target_b": ProbeTarget(
            "target_b",
            "binary",
            target_b_values,
            applicable,
            1,
            "target_b",
        ),
    }
    changed_targets = dict(first_targets)
    changed_targets["target_a"] = ProbeTarget(
        "target_a",
        "binary",
        1 - first_targets["target_a"].values,
        applicable,
        1,
        "target_a",
    )
    kwargs = {
        "family": family,
        "split": split,
        "epochs": 8,
        "batch_size": 19,
        "learning_rate": 0.02,
        "hidden_size": 11,
        "seed": 991,
        "device": "cpu",
    }

    first = fit_probe_family(features, first_targets, **kwargs)["targets"]["target_b"]
    changed = fit_probe_family(features, changed_targets, **kwargs)["targets"]["target_b"]

    assert (
        first["training_diagnostics"]["initial_state_sha256"] == changed["training_diagnostics"]["initial_state_sha256"]
    )
    assert first["training_diagnostics"]["final_state_sha256"] == changed["training_diagnostics"]["final_state_sha256"]
    assert (
        first["training_diagnostics"]["train_prediction_sha256"]
        == changed["training_diagnostics"]["train_prediction_sha256"]
    )
    assert (
        first["training_diagnostics"]["test_prediction_sha256"]
        == changed["training_diagnostics"]["test_prediction_sha256"]
    )
    assert first["train_metrics"] == changed["train_metrics"]
    assert first["metrics"] == changed["metrics"]


def test_target_summaries_and_no_skill_baselines_cover_every_target_kind():
    rng = np.random.default_rng(83)
    features = rng.normal(size=(12, 3)).astype(np.float32)
    split = probe_module.LayoutSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        train_layouts=("train",),
        test_layouts=("test",),
        fingerprint="unit-test",
        manifest={},
    )
    applicable = np.ones(12, dtype=bool)
    regression_values = np.asarray([1, 2, 3, np.nan, 5, 6, 7, 8, 10, 10, 10, np.nan], dtype=np.float32)
    regression_applicable = applicable.copy()
    multilabel_values = np.asarray(
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]] * 3,
        dtype=np.int64,
    )
    targets = {
        "binary": ProbeTarget(
            "binary", "binary", np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]), applicable, 1, "binary"
        ),
        "multiclass": ProbeTarget(
            "multiclass",
            "multiclass",
            np.asarray([0, 1, 1, 1, 0, 1, 0, 1, 2, 2, 2, 2]),
            applicable,
            3,
            "multiclass",
        ),
        "regression": ProbeTarget(
            "regression", "regression", regression_values, regression_applicable, 1, "regression"
        ),
        "multilabel": ProbeTarget("multilabel", "multilabel", multilabel_values, applicable, 3, "multilabel"),
    }

    fitted = fit_probe_family(
        features,
        targets,
        family="linear",
        split=split,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        seed=13,
        device="cpu",
    )["targets"]

    assert fitted["binary"]["target_summary"]["train"]["class_histogram"] == {"0": 5, "1": 3}
    assert fitted["multiclass"]["target_summary"]["test"]["class_histogram"] == {
        "0": 0,
        "1": 0,
        "2": 4,
    }
    regression_summary = fitted["regression"]["target_summary"]
    assert regression_summary["train"]["n_applicable_finite"] == 7
    assert regression_summary["test"]["n_applicable_finite"] == 3
    assert regression_summary["test"]["mean"]["value"] == 10.0
    assert regression_summary["test"]["std"]["value"] == 0.0
    assert regression_summary["test"]["median"]["value"] == 10.0
    multilabel_summary = fitted["multilabel"]["target_summary"]["train"]
    assert multilabel_summary["positive_count"] == {"0": 4, "1": 4, "2": 2}
    assert multilabel_summary["prevalence"] == {"0": 0.5, "1": 0.5, "2": 0.25}
    assert multilabel_summary["cardinality"]["mean"]["value"] == 1.25
    assert fitted["regression"]["no_skill_baseline"]["strategy"] == "training_mean"
    assert fitted["regression"]["no_skill_baseline"]["metrics"]["r2"]["value"] is None
    assert fitted["regression"]["no_skill_baseline"]["metrics"]["r2"]["reason"]
    assert fitted["multilabel"]["no_skill_baseline"]["strategy"] == "per_label_training_prevalence"
    assert fitted["multilabel"]["no_skill_baseline"]["predicted_prevalence"] == {
        "0": 0.5,
        "1": 0.5,
        "2": 0.25,
    }
    for result in fitted.values():
        assert result["train_metrics"]
        assert result["no_skill_baseline"]["metrics"]
        assert result["training_diagnostics"]["loss_curve_summary"]["finite"] is True
    json.dumps(fitted, allow_nan=False)


def test_result_content_fingerprint_excludes_only_declared_volatile_fields(tmp_path):
    payload = {
        "schema_version": 2,
        "git": {"head": "abc", "tracked_worktree_dirty": False},
        "dataset": {
            "content_sha256": "1" * 64,
            "contract_sha256": "2" * 64,
            "generation_git": {"head": "def", "tracked_worktree_dirty": False},
        },
        "runtime": {"torch_version": "old", "numpy_version": "old"},
        "results": {"h_final": {"score": 0.75}},
    }
    finalized = _with_result_content_fingerprint(payload)
    assert finalized["result_content_fingerprint"]["sha256"] == _result_content_fingerprint(finalized)
    assert _with_result_content_fingerprint(finalized) == finalized

    volatile_changes = copy.deepcopy(finalized)
    volatile_changes["git"]["tracked_worktree_dirty"] = True
    volatile_changes["dataset"]["generation_git"]["tracked_worktree_dirty"] = True
    volatile_changes["dataset"]["content_sha256"] = "9" * 64
    volatile_changes["runtime"]["torch_version"] = "new"
    assert _result_content_fingerprint(volatile_changes) == _result_content_fingerprint(finalized)

    meaningful_change = copy.deepcopy(finalized)
    meaningful_change["results"]["h_final"]["score"] = 0.5
    assert _result_content_fingerprint(meaningful_change) != _result_content_fingerprint(finalized)

    output = tmp_path / "result.json"
    _atomic_json(output, finalized)
    reloaded = json.loads(output.read_text())
    assert reloaded["result_content_fingerprint"] == finalized["result_content_fingerprint"]
    assert _result_content_fingerprint(reloaded) == finalized["result_content_fingerprint"]["sha256"]


def test_carry_payload_preflight_uses_full_carry_shape_and_requires_override():
    net, _config = _tiny_net()
    expected_bytes = 2 * 4 * 1 * 3 * 3 * 3 * np.dtype(np.float32).itemsize * 8
    estimate = carry_payload_preflight(
        net,
        puzzle_count=2,
        horizon=4,
        carry_dtype="float32",
        max_carry_bytes=expected_bytes,
        allow_large_dataset=False,
    )
    assert estimate["upper_bound_rows"] == 8
    assert estimate["carry_shape_per_row"] == [1, 3, 3, 3]
    assert estimate["carry_array_count"] == 8
    assert estimate["estimated_carry_bytes"] == expected_bytes
    assert estimate["within_limit"] is True

    with pytest.raises(ValueError, match="--allow-large-dataset"):
        carry_payload_preflight(
            net,
            puzzle_count=2,
            horizon=4,
            carry_dtype="float32",
            max_carry_bytes=expected_bytes - 1,
            allow_large_dataset=False,
        )
    with pytest.warns(RuntimeWarning, match="carry payload"):
        overridden = carry_payload_preflight(
            net,
            puzzle_count=2,
            horizon=4,
            carry_dtype="float32",
            max_carry_bytes=expected_bytes - 1,
            allow_large_dataset=True,
        )
    assert overridden["override_used"] is True


def test_cli_prints_concise_default_and_full_json_only_on_request(tmp_path, monkeypatch, capsys):
    def fake_probe_checkpoint(**kwargs):
        kwargs["progress"]("fitting isolated probes")
        return _with_result_content_fingerprint({"schema_version": 2, "results": {"ok": True}})

    monkeypatch.setattr(probe_module, "probe_checkpoint", fake_probe_checkpoint)
    common = [
        "probe_drc_carry",
        "--model",
        str(tmp_path / "model.pt"),
        "--bank",
        str(tmp_path / "bank.parquet"),
        "--dataset",
        str(tmp_path / "carry.npz"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    monkeypatch.setattr(sys, "argv", common)
    probe_module.main()
    captured = capsys.readouterr()
    assert captured.out.startswith("Wrote ")
    assert "result_content_sha256=" in captured.out
    assert '"results"' not in captured.out
    assert "fitting isolated probes" in captured.err

    monkeypatch.setattr(sys, "argv", [*common, "--output", str(tmp_path / "printed.json"), "--print-json"])
    probe_module.main()
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["results"] == {"ok": True}
