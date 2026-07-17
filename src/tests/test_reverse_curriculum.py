from dataclasses import replace

import pytest

from src.benchmarking.phase2 import Phase2Config, _fixture_attempt, _reverse_curriculum_levels
from src.env.criteria_env import BankCurriculumManager, layout_at_start_distance
from src.env.curriculum import BankCurriculumWrapper
from src.env.puzzle_bank import PuzzleBank, create_fixed_layout_from_seed
from src.env.ricochet_env import RicochetRobotsEnv


def _config(**overrides) -> Phase2Config:
    values = dict(
        height=8,
        width=8,
        num_robots=2,
        obs_mode="image",
        policy="grid_cnn",
        max_optimal_length=4,
        solver_max_depth=4,
        curriculum_start_distances=(1, 2, 3),
        curriculum_frontier_thresholds={1: 0.8, 2: 0.7, 3: 0.7},
        reverse_final_full_starts=True,
        curriculum_frontier_window=4,
        curriculum_min_episodes=1,
        curriculum_check_frequency=1,
        training_seeds=(3,),
        checkpoints=(0, 8),
        n_envs=1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
    )
    values.update(overrides)
    return Phase2Config(**values)


def _puzzles(config: Phase2Config, count: int = 12):
    puzzles = [
        puzzle
        for seed in range(2000)
        if (puzzle := _fixture_attempt((seed, config))) is not None
    ]
    assert len(puzzles) >= count
    return puzzles[:count]


def test_reverse_start_suffix_solves_exactly_and_full_distance_is_original() -> None:
    config = _config()
    puzzle = next(puzzle for puzzle in _puzzles(config, 20) if puzzle.optimal_length >= 3)
    original = create_fixed_layout_from_seed(puzzle)
    for distance in range(1, puzzle.optimal_length + 1):
        offset = layout_at_start_distance(puzzle, original, distance)
        env = RicochetRobotsEnv(**config.environment_spec(7).to_env_kwargs(fixed_layout=offset))
        env.reset()
        terminated = truncated = False
        for index, action in enumerate(puzzle.actions_encoded[-distance:], start=1):
            _, _, terminated, truncated, _ = env.step(int(action))
            assert not truncated
            assert terminated is (index == distance)
        assert terminated
        if distance == puzzle.optimal_length:
            assert offset.robot_positions == original.robot_positions


def test_reverse_level_construction_and_validation() -> None:
    config = _config()
    levels = _reverse_curriculum_levels(config)
    assert len(levels) == 4
    assert [band["band_key"] for band in levels[1]["bands"]] == ["d1", "d2"]
    assert levels[1]["bands"][0]["weight"] == 1.0
    assert levels[1]["bands"][1]["weight"] == config.curriculum_frontier_weight
    assert levels[1]["frontier_threshold"] == 0.7
    assert levels[-1]["name"] == "reverse-final-original-starts"
    assert all("start_distance" not in band for band in levels[-1]["bands"])

    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(curriculum_stage_max_lengths=(2, 4))
    with pytest.raises(ValueError, match="Missing calibrated"):
        _config(curriculum_frontier_thresholds={1: 0.8})


def test_frontier_gate_cannot_be_hidden_by_easy_band_success(tmp_path) -> None:
    config = _config(reverse_final_full_starts=False)
    bank = PuzzleBank(tmp_path / "bank", run_id="frontier-gate")
    manager = BankCurriculumManager(
        bank,
        _reverse_curriculum_levels(config),
        success_rate_threshold=0.7,
        min_episodes_per_level=1,
        success_rate_window_size=20,
        advancement_check_frequency=1,
        verbose=False,
    )
    for _ in range(10):
        manager.record_episode_result(True, "easy")
    for result in (True, True, False, False):
        manager.record_episode_result(result, "d1")
    assert manager.current_level == 0
    assert manager.get_success_rate() > 0.7
    assert manager.get_frontier_success_rate() == 0.5
    for result in (True, True, True, True):
        manager.record_episode_result(result, "d1")
    assert manager.current_level == 1
    assert manager.transitions[-1]["reason"] == "mastery"


def test_wrapper_preserves_band_key_and_records_frontier_result(tmp_path) -> None:
    config = _config(curriculum_start_distances=(1,), curriculum_frontier_thresholds={1: 0.8})
    puzzles = _puzzles(config)
    bank = PuzzleBank(tmp_path / "bank", run_id="reverse-wrapper")
    bank.add_puzzles(puzzles)
    manager = BankCurriculumManager(
        bank,
        _reverse_curriculum_levels(config),
        success_rate_threshold=0.7,
        min_episodes_per_level=100,
        success_rate_window_size=4,
        advancement_check_frequency=1,
        verbose=False,
    )
    wrapper = BankCurriculumWrapper(
        bank,
        manager,
        environment_spec=config.environment_spec(17),
        split="all",
        track_curriculum=True,
        verbose=False,
    )
    _, reset_info = wrapper.reset(seed=17)
    assert reset_info["curriculum_band"] == "d1"
    assert reset_info["start_distance"] == 1
    puzzle = reset_info["bank_metadata"]
    _, _, terminated, truncated, terminal_info = wrapper.step(int(puzzle.actions_encoded[-1]))
    assert terminated and not truncated
    assert terminal_info["curriculum_band"] == "d1"
    assert manager.get_band_success_rates()["d1"] == 1.0
    wrapper.close()
