"""Contract tests for the GridPlanesCnn extractor and its Phase 2 wiring."""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from src.benchmarking.phase2 import Phase2Config
from src.models.grid_cnn import GridPlanesCnn


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
