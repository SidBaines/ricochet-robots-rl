import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from src.env.criteria_env import CriteriaFilteredEnv, PuzzleCriteria
from src.env.criteria_env import BankCurriculumManager
from src.env.curriculum import (
    BankCurriculumWrapper,
    CurriculumConfig,
    CurriculumLevel,
    CurriculumManager,
    CurriculumWrapper,
)
from src.env.environment_spec import EnvironmentSpec
from src.env.puzzle_bank import (
    PuzzleBank,
    PuzzleMetadata,
    SolverKey,
    SpecKey,
    compute_board_hash,
    compute_layout_hash,
)
from src.env.ricochet_env import generate_board_from_spec_with_seed
from src.scripts.train_agent import (
    apply_initialization_overrides,
    build_training_parser,
    callback_frequency,
    effective_environment_spec,
    flatten_structured_config,
    make_env_factory,
    make_bank_curriculum_env_factory,
    make_trajectory_env_factory,
    select_policy,
    validate_and_describe_model,
)
from src.training.run_manifest import create_run_manifest, write_run_manifest


def _args(**overrides):
    args = build_training_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("mode", ["random", "v0", "v1"])
def test_environment_spec_survives_standard_factories(mode: str) -> None:
    args = _args(
        env_mode=mode,
        include_noop=True,
        step_penalty=-0.25,
        goal_reward=7.0,
        noop_penalty=-0.5,
        max_steps=33,
        seed=19,
        obs_mode="rgb_cell_image",
        cell_obs_pixel_size=40,
    )
    env = make_env_factory(args)()

    assert env.include_noop is True
    assert env.step_penalty == -0.25
    assert env.goal_reward == 7.0
    assert env.noop_penalty == -0.5
    assert env.max_steps == 33
    assert env.observation_space.shape == (3, 40, 40)
    assert env.action_space.n == env.num_robots * 4 + 1


def test_bank_environment_receives_complete_environment_spec(tmp_path: Path) -> None:
    spec_key = SpecKey(4, 4, 2, 0, 0)
    seed = 1234
    board = generate_board_from_spec_with_seed(
        height=4,
        width=4,
        num_robots=2,
        edge_t_per_quadrant=0,
        central_l_per_quadrant=0,
        seed=seed,
    )
    metadata = PuzzleMetadata(
        seed=seed,
        spec_key=spec_key,
        solver_key=SolverKey(),
        board_hash=compute_board_hash(board),
        optimal_length=1,
        robots_moved=1,
        nodes_expanded=0,
        solve_time_ms=0,
        solved_within_limits=True,
        hit_depth_limit=False,
        hit_node_limit=False,
        actions_encoded=[],
        created_at=0.0,
        run_id="test",
        board_seed=seed,
        layout_hash=compute_layout_hash(board),
    )
    bank = PuzzleBank(tmp_path / "bank")
    bank.add_puzzles([metadata])
    environment_spec = EnvironmentSpec(
        height=4,
        width=4,
        num_robots=2,
        include_noop=True,
        step_penalty=-0.25,
        goal_reward=7.0,
        noop_penalty=-0.5,
        max_steps=33,
        seed=19,
        obs_mode="rgb_cell_image",
        channels_first=True,
        cell_obs_pixel_size=40,
    )
    env = CriteriaFilteredEnv(
        bank,
        PuzzleCriteria(spec_key, 1, 1, 1, 1),
        environment_spec=environment_spec,
    )
    env.reset(seed=91)
    inner = env._current_env
    assert inner is not None

    assert inner.include_noop is True
    assert inner.step_penalty == -0.25
    assert inner.goal_reward == 7.0
    assert inner.noop_penalty == -0.5
    assert inner.max_steps == 33
    assert inner.observation_space.shape == (3, 40, 40)
    assert inner.action_space.n == 9

    level = {
        "name": "test",
        "spec_key": spec_key,
        "min_optimal_length": 1,
        "max_optimal_length": 1,
        "min_robots_moved": 1,
        "max_robots_moved": 1,
    }
    manager = BankCurriculumManager(bank, [level], verbose=False)
    evaluation = BankCurriculumWrapper(
        bank,
        manager,
        environment_spec=environment_spec,
        split="all",
        track_curriculum=False,
        verbose=False,
    )
    evaluation.reset(seed=91)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = evaluation.step(8)

    assert truncated is True
    assert manager.episode_count == 0
    assert manager.success_rate_window == []

    second_level = dict(level, name="test-2")
    sync_manager = BankCurriculumManager(bank, [level, second_level], verbose=False)
    read_only = BankCurriculumWrapper(
        bank,
        sync_manager,
        environment_spec=environment_spec,
        split="all",
        track_curriculum=False,
        verbose=False,
    )
    read_only.sync_curriculum_level(1)
    assert sync_manager.current_level == 1
    assert read_only._last_known_level == 1

    shared_manager = BankCurriculumManager(bank, [level, second_level], verbose=False)
    worker_a = BankCurriculumWrapper(bank, shared_manager, environment_spec=environment_spec, verbose=False)
    worker_b = BankCurriculumWrapper(bank, shared_manager, environment_spec=environment_spec, verbose=False)
    worker_a.reset(seed=101)
    worker_b.reset(seed=202)
    shared_manager.current_level = 1
    worker_a._create_current_env()
    worker_b._create_current_env()

    assert worker_a._current_env.sampler.seed == 101
    assert worker_b._current_env.sampler.seed == 202

    trajectory_args = _args(
        height=4,
        width=4,
        num_robots=1,
        obs_mode="rgb_cell_image",
        cell_obs_pixel_size=40,
        bank_eval_fraction=0.999999,
        bank_split_seed=0,
        bank_split_key="layout_hash",
    )
    trajectory_factory = make_trajectory_env_factory(trajectory_args, manager)
    trajectory_env = trajectory_factory()
    next_trajectory_env = trajectory_factory()

    assert trajectory_env.curriculum_manager is not manager
    assert trajectory_env.track_curriculum is False
    assert trajectory_env.split == "eval"
    assert trajectory_env.action_space.n == 8
    assert trajectory_env._worker_seed != next_trajectory_env._worker_seed

    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level": 0,
                        "name": "sparse",
                        "spec_key": {
                            "height": 4,
                            "width": 4,
                            "num_robots": 2,
                            "edge_t_per_quadrant": 0,
                            "central_l_per_quadrant": 0,
                            "generator_rules_version": "1.0",
                        },
                        "min_optimal_length": 1,
                        "max_optimal_length": 1,
                        "min_robots_moved": 1,
                        "max_robots_moved": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sparse_args = _args(
        bank_dir=str(tmp_path / "bank"),
        curriculum_config=str(curriculum_path),
        curriculum_initial_level=0,
        bank_eval_fraction=0.5,
        bank_split_seed=0,
        bank_split_key="layout_hash",
        height=4,
        width=4,
        num_robots=2,
    )
    with pytest.raises(ValueError, match="Bank split coverage is empty"):
        make_bank_curriculum_env_factory(sparse_args)


def test_structured_config_rejects_unknown_and_unsupported_model_keys() -> None:
    with pytest.raises(ValueError, match="Unknown env configuration keys"):
        flatten_structured_config({"env": {"max_stepz": 10}})
    with pytest.raises(ValueError, match="Unsupported model.type"):
        flatten_structured_config({"model": {"type": "resnet"}})
    with pytest.raises(ValueError, match="model.params is not supported"):
        flatten_structured_config({"model": {"type": "stock_cnn", "params": {"width": 32}}})
    with pytest.raises(ValueError, match="Unknown initialization configuration keys"):
        apply_initialization_overrides(_args(), {"mode": "fresh", "typo": True})


def test_default_config_resolves_to_supported_baseline() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "train_defaults.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    flat = flatten_structured_config(raw)

    assert raw["initialization"]["mode"] == "fresh"
    assert flat["model_type"] == "stock_cnn"
    assert flat["obs_mode"] == "rgb_cell_image"
    assert flat["cell_obs_pixel_size"] == 40
    assert flat["bank_eval_fraction"] == 0.2


def test_policy_selection_is_explicit() -> None:
    image_space = spaces.Box(0, 255, shape=(3, 40, 40), dtype=np.uint8)
    policy, kwargs = select_policy(_args(model_type="stock_cnn", obs_mode="rgb_cell_image"), image_space)
    assert policy == "CnnPolicy"
    assert kwargs["normalize_images"] is True

    symbolic_space = spaces.Box(0, 1, shape=(8,), dtype=np.float32)
    policy, _ = select_policy(_args(model_type="mlp", obs_mode="symbolic"), symbolic_space)
    assert policy == "MlpPolicy"

    with pytest.raises(ValueError, match="Unsupported legacy model flags"):
        select_policy(_args(model_type="stock_cnn", obs_mode="rgb_cell_image", resnet=True), image_space)
    with pytest.raises(ValueError, match="flattened RGB inputs"):
        select_policy(_args(model_type="mlp", obs_mode="rgb_cell_image"), image_space)


def test_effective_environment_spec_records_curriculum_geometry() -> None:
    args = _args(height=8, width=8, num_robots=2)
    spec_key = SpecKey(16, 16, 4, 2, 2)

    class Manager:
        def get_current_criteria(self):
            return PuzzleCriteria(spec_key)

    resolved = effective_environment_spec(args, Manager())
    assert (resolved.height, resolved.width, resolved.num_robots) == (16, 16, 4)


def test_online_eval_wrapper_syncs_level_without_tracking_results() -> None:
    levels = [CurriculumLevel(i, f"level-{i}", 4, 4, 1, 0, 0, i + 1, 5, 100, "test") for i in range(2)]
    manager = CurriculumManager(CurriculumConfig(levels), verbose=False)
    spec = EnvironmentSpec(height=4, width=4, num_robots=1, max_steps=5)
    wrapper = CurriculumWrapper(
        base_env_factory=lambda: None,
        curriculum_manager=manager,
        environment_spec=spec,
        track_curriculum=False,
        verbose=False,
    )

    wrapper.sync_curriculum_level(1)

    assert manager.current_level == 1
    assert manager.episode_count == 0
    assert wrapper._last_known_level == 1


def test_callback_frequency_uses_aggregate_timesteps() -> None:
    assert callback_frequency(1_000, 1) == 1_000
    assert callback_frequency(1_000, 8) == 125
    assert callback_frequency(1_001, 8) == 126
    assert callback_frequency(0, 8) == 0


def test_vecnormalize_sync_copies_current_statistics() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

    train = VecNormalize(DummyVecEnv([make_env_factory(_args(obs_mode="symbolic"))]))
    evaluation = VecNormalize(
        DummyVecEnv([make_env_factory(_args(obs_mode="symbolic"))]),
        training=False,
        norm_reward=False,
    )
    train.obs_rms.mean[...] = 3.5
    original_eval_stats = evaluation.obs_rms

    sync_envs_normalization(train, evaluation)

    np.testing.assert_allclose(evaluation.obs_rms.mean, train.obs_rms.mean)
    assert evaluation.obs_rms is not train.obs_rms
    assert evaluation.obs_rms is not original_eval_stats
    assert evaluation.training is False
    assert evaluation.norm_reward is False
    train.close()
    evaluation.close()


def test_run_manifest_is_json_and_records_resolved_contract(tmp_path: Path) -> None:
    spec = EnvironmentSpec(obs_mode="rgb_cell_image", cell_obs_pixel_size=40)
    observation_space = spaces.Box(0, 255, shape=(3, 40, 40), dtype=np.uint8)
    action_space = spaces.Discrete(8)
    manifest = create_run_manifest(
        project_root=Path(__file__).resolve().parents[2],
        requested_environment_spec=spec,
        effective_environment_spec=spec,
        model_type="stock_cnn",
        policy="CnnPolicy",
        policy_kwargs={"normalize_images": True},
        actual_model={
            "algorithm_class": "stable_baselines3.ppo.PPO",
            "policy_class": "stable_baselines3.common.policies.ActorCriticCnnPolicy",
            "features_extractor_class": "stable_baselines3.common.torch_layers.NatureCNN",
            "actual_policy_kwargs": {"normalize_images": True},
            "checkpoint_sha256": None,
        },
        observation_space=observation_space,
        action_space=action_space,
        initial_timesteps=0,
        requested_additional_timesteps=256,
        target_total_timesteps=256,
        seed=5,
        resolved_config={"model_type": "stock_cnn"},
        bank_split={"training_split": "train", "evaluation_split": "eval"},
    )
    path = tmp_path / "run_manifest.json"
    write_run_manifest(path, manifest)
    loaded = json.loads(path.read_text())

    assert loaded["schema_version"] == 2
    assert loaded["environment"]["cell_obs_pixel_size"] == 40
    assert loaded["model"]["type"] == "stock_cnn"
    assert loaded["model"]["requested_policy"] == "CnnPolicy"
    assert loaded["model"]["features_extractor_class"].endswith("NatureCNN")
    assert loaded["timesteps"] == {
        "final_total": None,
        "initial": 0,
        "requested_additional": 256,
        "target_total": 256,
    }
    assert loaded["git"]["commit"]


def test_actual_model_contract_is_validated_and_checkpoint_is_fingerprinted(tmp_path: Path) -> None:
    NatureCNN = type("NatureCNN", (), {})
    UnsupportedExtractor = type("UnsupportedExtractor", (), {})
    Policy = type("ActorCriticCnnPolicy", (), {})
    Model = type("PPO", (), {})
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint contents")

    policy = Policy()
    policy.features_extractor = NatureCNN()
    model = Model()
    model.policy = policy
    model.policy_kwargs = {"normalize_images": True}

    summary = validate_and_describe_model(model, "stock_cnn", checkpoint_path=str(checkpoint))

    assert summary["features_extractor_class"].endswith("NatureCNN")
    assert len(summary["checkpoint_sha256"]) == 64

    policy.features_extractor = UnsupportedExtractor()
    with pytest.raises(ValueError, match="Model contract mismatch"):
        validate_and_describe_model(model, "stock_cnn")
