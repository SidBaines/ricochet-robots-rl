import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from src.benchmarking.phase2 import (
    Phase2Config,
    _build_model,
    _make_cumulative_schedule_callback,
    _maybe_init_wandb,
    _make_wandb_rollout_callback,
    _puzzle_clustered_bootstrap,
    build_confirmation_manifest,
    run_phase2_benchmark,
)
from src.env.puzzle_bank import PuzzleBank
from src.env.ricochet_env import RicochetRobotsEnv


def _tiny_config() -> Phase2Config:
    return Phase2Config(
        fixture_attempts=400,
        heldout_fraction=0.5,
        suite_per_stratum=1,
        training_seeds=(3,),
        checkpoints=(0, 8),
        n_envs=1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        validation_episodes_per_puzzle=1,
        final_stochastic_episodes_per_puzzle=2,
        baseline_episodes_per_puzzle=2,
        overfit_puzzles_per_stratum=1,
        overfit_stochastic_episodes_per_puzzle=2,
        save_checkpoint_weights=True,
        select_checkpoint_by_validation=True,
        validation_bootstrap_samples=100,
        fixed_validation_seeds=True,
        device="cpu",
    )


def test_phase2_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown Phase 2 configuration keys"):
        Phase2Config.from_mapping({"training_seeds": [1], "unknown": True})


def test_phase2_config_validates_explicit_policy_and_checkpoint_controls() -> None:
    with pytest.raises(ValueError, match="policy_activation"):
        Phase2Config(policy_activation="gelu")
    with pytest.raises(ValueError, match="bc_action_head_gain"):
        Phase2Config(bc_action_head_gain=0)
    with pytest.raises(ValueError, match="requires save_checkpoint_weights"):
        Phase2Config(select_checkpoint_by_validation=True)
    with pytest.raises(ValueError, match="evaluation_batch_size"):
        Phase2Config(evaluation_batch_size=0)
    with pytest.raises(ValueError, match="bc_precision"):
        Phase2Config(bc_precision="tf32")


def test_puzzle_bootstrap_clusters_repeated_episodes() -> None:
    records = [
        {"puzzle_id": "a", "success": True},
        {"puzzle_id": "a", "success": True},
        {"puzzle_id": "b", "success": False},
        {"puzzle_id": "b", "success": False},
    ]
    interval = _puzzle_clustered_bootstrap(records, samples=500, seed=7)
    assert interval is not None
    assert interval[0] == 0.0
    assert interval[1] == 1.0


def test_wandb_callback_forwards_train_and_health_metrics() -> None:
    from stable_baselines3.common.env_util import make_vec_env

    config = replace(
        _tiny_config(),
        obs_mode="image",
        policy="grid_cnn",
        features_dim=32,
        conv_channels=8,
        n_conv_layers=1,
        save_checkpoint_weights=False,
        select_checkpoint_by_validation=False,
    )
    vec_env = make_vec_env(
        lambda: RicochetRobotsEnv(**config.environment_spec(123).to_env_kwargs()),
        n_envs=1,
        seed=123,
    )
    model = _build_model(vec_env, config, seed=123, device="cpu")
    obs = np.stack([vec_env.reset()[0]])

    class FakeRun:
        def __init__(self) -> None:
            self.payloads = []

        def log(self, payload) -> None:
            self.payloads.append(dict(payload))

    run = FakeRun()
    callback = _make_wandb_rollout_callback(
        run,
        None,
        health_observations=obs,
        log_every_rollouts=1,
    )
    model.learn(total_timesteps=16, callback=callback)
    assert any("train/approx_kl" in payload for payload in run.payloads)
    assert any("train/explained_variance" in payload for payload in run.payloads)
    assert any("health/logit_between_state_variance" in payload for payload in run.payloads)
    vec_env.close()


def test_wandb_metric_patterns_use_supported_suffix_globs(monkeypatch, tmp_path) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.patterns = []

        def define_metric(self, name, **kwargs) -> None:
            if "*" in name and not name.endswith("/*"):
                raise AssertionError(f"unsupported non-suffix glob: {name}")
            self.patterns.append((name, kwargs))

    run = FakeRun()

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            return run

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    config = replace(_tiny_config(), wandb_project="test-project")
    assert _maybe_init_wandb(config, seed=3, output_dir=tmp_path) is run
    assert ("eval_val_stoch/*", {"step_metric": "train/timesteps"}) in run.patterns


def test_cumulative_optimizer_schedule_reaches_declared_endpoints() -> None:
    from stable_baselines3.common.env_util import make_vec_env

    config = replace(
        _tiny_config(),
        obs_mode="image",
        policy="grid_cnn",
        features_dim=32,
        conv_channels=8,
        n_conv_layers=1,
        checkpoints=(0, 16),
        learning_rate=3e-4,
        learning_rate_final=3e-5,
        ent_coef=0.01,
        ent_coef_final=0.001,
        save_checkpoint_weights=False,
        select_checkpoint_by_validation=False,
    )
    vec_env = make_vec_env(
        lambda: RicochetRobotsEnv(**config.environment_spec(321).to_env_kwargs()),
        n_envs=1,
        seed=321,
    )
    model = _build_model(vec_env, config, seed=321, device="cpu")
    model.learn(total_timesteps=16, callback=_make_cumulative_schedule_callback(config))
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert model.ent_coef == pytest.approx(0.001)
    vec_env.close()


@pytest.mark.integration
@pytest.mark.training
def test_tiny_phase2_benchmark_is_disjoint_replayable_and_complete(tmp_path) -> None:
    config = _tiny_config()
    bank_dir = tmp_path / "bank"
    output_dir = tmp_path / "output"

    result = run_phase2_benchmark(config, bank_dir=bank_dir, output_dir=output_dir)
    replay = run_phase2_benchmark(config, bank_dir=bank_dir, output_dir=tmp_path / "replay")

    memberships = result["split_manifest"]["memberships"]
    train = set(memberships["train"])
    validation = set(memberships["validation"])
    test = set(memberships["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert result["split_manifest"]["fingerprint"] == replay["split_manifest"]["fingerprint"]
    assert [point["stochastic"]["successes"] for point in result["overfit"]["curve"]] == [
        point["stochastic"]["successes"] for point in replay["overfit"]["curve"]
    ]
    assert [point["stochastic"]["successes"] for point in result["seeds"][0]["curve"]] == [
        point["stochastic"]["successes"] for point in replay["seeds"][0]["curve"]
    ]
    assert result["baselines"]["oracle"]["summary"]["success_rate"] == 1.0
    assert set(result["baselines"]["oracle"]["summary"]["by_optimal_length"]) == {"1", "2"}
    assert len(result["overfit"]["puzzle_ids"]) == 2
    assert result["overfit"]["actual_timesteps"] == 8
    assert result["seeds"][0]["actual_timesteps"] == 8
    assert (output_dir / "overfit" / "model.zip").exists()
    assert (output_dir / "seed_3" / "model.zip").exists()
    assert (output_dir / "seed_3" / "checkpoints" / "step_0" / "model.zip").exists()
    assert (output_dir / "seed_3" / "checkpoints" / "step_8" / "evaluation.json").exists()
    assert result["seeds"][0]["selected_checkpoint_timesteps"] in {0, 8}
    assert result["seeds"][0]["policy"]["effective_construction"]["activation"] == "tanh"
    assert "success_rate_puzzle_bootstrap_95" in result["seeds"][0]["curve"][0]["stochastic"]
    assert (output_dir / "report.md").exists()
    persisted = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1

    confirmation = build_confirmation_manifest(
        PuzzleBank(bank_dir),
        result["split_manifest"],
        tmp_path / "confirmation.json",
        config,
        max_per_stratum=1,
    )
    selected = set(confirmation["selected_puzzle_ids"])
    assert selected.isdisjoint(result["split_manifest"]["test_suite"])
    exposed_layouts = set(confirmation["excluded_layout_ids"])
    assert set(confirmation["selected_layout_ids"]).isdisjoint(exposed_layouts)
    assert len(confirmation["selected_layout_ids"]) == len(set(confirmation["selected_layout_ids"]))


@pytest.mark.integration
@pytest.mark.training
def test_validation_only_run_keeps_final_test_sealed(tmp_path) -> None:
    config = replace(_tiny_config(), run_final_test=False)
    result = run_phase2_benchmark(config, bank_dir=tmp_path / "bank", output_dir=tmp_path / "output")
    seed_result = result["seeds"][0]
    assert seed_result["test_status"] == "sealed_not_run"
    assert seed_result["test_selected"] is None
    assert result["heldout_milestone"]["observed"] is None
    assert "Final test remained sealed" in (tmp_path / "output" / "report.md").read_text(encoding="utf-8")


@pytest.mark.integration
@pytest.mark.training
def test_initial_checkpoint_is_loaded_exactly_at_step_zero(tmp_path) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    base = _tiny_config()
    vec_env = make_vec_env(
        lambda: RicochetRobotsEnv(**base.environment_spec(77).to_env_kwargs()),
        n_envs=1,
        seed=77,
    )
    initial = _build_model(vec_env, base, seed=77, device="cpu")
    initial_path = tmp_path / "initial_model"
    initial.save(initial_path)
    vec_env.close()

    config = replace(base, initial_checkpoint_path=str(initial_path.with_suffix(".zip")))
    result = run_phase2_benchmark(config, bank_dir=tmp_path / "bank", output_dir=tmp_path / "output")
    loaded = PPO.load(result["seeds"][0]["checkpoints"][0]["model"], device="cpu")
    reference = PPO.load(initial_path.with_suffix(".zip"), device="cpu")
    for observed, expected in zip(loaded.policy.parameters(), reference.policy.parameters()):
        assert np.array_equal(observed.detach().numpy(), expected.detach().numpy())
