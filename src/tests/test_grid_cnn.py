"""Contract tests for the GridPlanesCnn extractor and its Phase 2 wiring."""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from src.benchmarking.phase2 import Phase2Config
from src.models.grid_cnn import GridPlanesCnn, GridPlanesResNet


def _planes_space(channels: int = 8, height: int = 8, width: int = 8) -> Box:
    return Box(low=0.0, high=1.0, shape=(channels, height, width), dtype=np.float32)


def test_grid_cnn_preserves_batch_and_feature_dims() -> None:
    space = _planes_space()
    extractor = GridPlanesCnn(space, features_dim=256, conv_channels=32, n_conv_layers=3)
    batch = torch.zeros((5, *space.shape), dtype=torch.float32)
    features = extractor(batch)
    assert features.shape == (5, 256)
    assert torch.isfinite(features).all()


def test_grid_cnn_distinguishes_robot_positions() -> None:
    """Two boards differing only in one robot cell must map to distinct features."""
    space = _planes_space()
    extractor = GridPlanesCnn(space, features_dim=64, conv_channels=16, n_conv_layers=2)
    board_a = torch.zeros((1, *space.shape), dtype=torch.float32)
    board_b = board_a.clone()
    board_a[0, 6, 2, 3] = 1.0
    board_b[0, 6, 5, 5] = 1.0
    features_a = extractor(board_a)
    features_b = extractor(board_b)
    assert not torch.allclose(features_a, features_b)


def test_grid_cnn_rejects_non_spatial_observations() -> None:
    with pytest.raises(ValueError):
        GridPlanesCnn(Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32))


def test_phase2_config_rejects_mismatched_policy_and_obs_mode() -> None:
    with pytest.raises(ValueError, match="grid_cnn requires"):
        Phase2Config(policy="grid_cnn", obs_mode="rgb_cell_image")
    with pytest.raises(ValueError, match="stock_cnn requires"):
        Phase2Config(policy="stock_cnn", obs_mode="image")
    with pytest.raises(ValueError, match="Unsupported policy"):
        Phase2Config(policy="resnet")


def test_phase2_config_accepts_grid_cnn_on_planes() -> None:
    config = Phase2Config(policy="grid_cnn", obs_mode="image")
    spec = config.environment_spec(seed=0)
    assert spec.obs_mode == "image"


def test_grid_resnet_preserves_batch_and_feature_dims() -> None:
    space = _planes_space(channels=10, height=16, width=16)
    extractor = GridPlanesResNet(space, features_dim=512, conv_channels=64, n_blocks=3)
    batch = torch.zeros((4, *space.shape), dtype=torch.float32)
    features = extractor(batch)
    assert features.shape == (4, 512)
    assert torch.isfinite(features).all()


def test_grid_resnet_distinguishes_robot_positions() -> None:
    space = _planes_space(channels=10, height=16, width=16)
    extractor = GridPlanesResNet(space, features_dim=64, conv_channels=32, n_blocks=2)
    board_a = torch.zeros((1, *space.shape), dtype=torch.float32)
    board_b = board_a.clone()
    board_a[0, 6, 2, 3] = 1.0
    board_b[0, 6, 12, 9] = 1.0
    assert not torch.allclose(extractor(board_a), extractor(board_b))


def test_phase2_config_accepts_grid_resnet_on_planes() -> None:
    config = Phase2Config(policy="grid_resnet", obs_mode="image")
    assert config.environment_spec(seed=0).obs_mode == "image"
    with pytest.raises(ValueError, match="grid_resnet requires"):
        Phase2Config(policy="grid_resnet", obs_mode="rgb_cell_image")


def test_curriculum_stall_safety_valve_forces_advancement() -> None:
    from src.env.criteria_env import BankCurriculumManager

    levels = [{"name": f"level-{i}", "spec_key": None} for i in range(2)]
    manager = BankCurriculumManager(
        bank=None,
        curriculum_levels=levels,
        success_rate_threshold=1.0,
        min_episodes_per_level=10_000,
        success_rate_window_size=100,
        advancement_check_frequency=10,
        max_episodes_per_level=50,
        verbose=False,
    )
    for _ in range(49):
        manager.record_episode_result(False)
    assert manager.get_current_level() == 0
    manager.record_episode_result(False)
    assert manager.get_current_level() == 1
    # Final level: the valve must not advance past the end.
    for _ in range(200):
        manager.record_episode_result(False)
    assert manager.get_current_level() == 1


def test_train_agent_select_policy_supports_grid_cnn() -> None:
    from types import SimpleNamespace

    from src.scripts.train_agent import select_policy

    args = SimpleNamespace(model_type="grid_cnn", obs_mode="image")
    policy, policy_kwargs = select_policy(args, _planes_space())
    assert policy == "MlpPolicy"
    assert policy_kwargs["features_extractor_class"] is GridPlanesCnn
    assert policy_kwargs["normalize_images"] is False

    with pytest.raises(ValueError, match="grid_cnn requires"):
        select_policy(SimpleNamespace(model_type="grid_cnn", obs_mode="rgb_cell_image"), _planes_space())
