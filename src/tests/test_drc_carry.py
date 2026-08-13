from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.groundup.drc import DRCNet, DRCState
from src.groundup.drc_carry import (
    BOS_ACTION,
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
    ProbeTarget,
    _build_argument_parser,
    build_random_control,
    fit_probe_family,
    load_schema2_checkpoint,
    run_controlled_probes,
    shuffle_training_labels,
    stable_layout_split,
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
