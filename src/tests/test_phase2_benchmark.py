import json

import pytest

from src.benchmarking.phase2 import Phase2Config, run_phase2_benchmark


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
        device="cpu",
    )


def test_phase2_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown Phase 2 configuration keys"):
        Phase2Config.from_mapping({"training_seeds": [1], "unknown": True})


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
    assert (output_dir / "report.md").exists()
    persisted = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
