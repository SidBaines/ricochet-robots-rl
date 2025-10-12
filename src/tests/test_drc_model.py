from __future__ import annotations

import torch
import pytest


def test_drc_features_extractor_shapes_cpu():
    from gymnasium.spaces import Box
    from src.models.convlstm import ConvLSTMFeaturesExtractor

    C_in, H, W = 8, 10, 10
    obs_space = Box(low=0.0, high=1.0, shape=(C_in, H, W))
    extractor = ConvLSTMFeaturesExtractor(
        observation_space=obs_space,
        features_dim=128,
        conv_channels=16,
        lstm_channels=32,
        num_lstm_layers=3,
        num_repeats=3,
        use_pool_and_inject=True,
    )
    extractor.eval()

    B = 4
    obs = torch.rand(B, C_in, H, W)
    features, state = extractor(obs, None)

    assert features.shape == (B, 128)
    assert isinstance(state, list)
    assert len(state) == 3
    for h, c in state:
        assert h.shape == (B, 32, H, W)
        assert c.shape == (B, 32, H, W)


def test_drc_state_persistence_and_repeats_effect():
    from gymnasium.spaces import Box
    from src.models.convlstm import ConvLSTMFeaturesExtractor

    torch.manual_seed(0)
    C_in, H, W = 6, 8, 8
    obs_space = Box(low=0.0, high=1.0, shape=(C_in, H, W))
    extractor = ConvLSTMFeaturesExtractor(
        observation_space=obs_space,
        features_dim=64,
        conv_channels=8,
        lstm_channels=16,
        num_lstm_layers=2,
        num_repeats=2,
        use_pool_and_inject=True,
    )
    extractor.train(False)

    B = 2
    obs = torch.rand(B, C_in, H, W)
    feat1, state1 = extractor(obs, None)
    feat2, state2 = extractor(obs, state1)

    # Features should be tensors with same shape, and states should update (not be identical zeros)
    assert feat1.shape == feat2.shape == (B, 64)
    assert len(state1) == len(state2) == 2

    # Hidden states change across calls (most entries non-zero and not exactly equal)
    diffs = []
    for (h1, c1), (h2, c2) in zip(state1, state2):
        diffs.append(torch.mean(torch.abs(h2 - h1)).item())
        diffs.append(torch.mean(torch.abs(c2 - c1)).item())
    assert any(d > 0.0 for d in diffs)


def test_drc_gradient_flow():
    from gymnasium.spaces import Box
    from src.models.convlstm import ConvLSTMFeaturesExtractor

    C_in, H, W = 5, 6, 6
    obs_space = Box(low=0.0, high=1.0, shape=(C_in, H, W))
    extractor = ConvLSTMFeaturesExtractor(
        observation_space=obs_space,
        features_dim=32,
        conv_channels=8,
        lstm_channels=8,
        num_lstm_layers=2,
        num_repeats=1,
        use_pool_and_inject=True,
    )
    extractor.train(True)

    B = 3
    obs = torch.rand(B, C_in, H, W)
    features, state = extractor(obs, None)
    loss = features.sum()
    loss.backward()

    # Ensure some parameters received gradients
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in extractor.parameters())
    assert has_grad


def test_drcstack_shapes_direct():
    from src.models.convlstm import DRCStack

    B, Ce, C, H, W = 2, 8, 16, 7, 7
    drc = DRCStack(
        encoded_channels=Ce,
        hidden_channels=C,
        num_layers=3,
        repeats_per_step=2,
        kernel_size=3,
        use_boundary_channel=True,
        use_pool_and_inject=True,
    )
    E = torch.randn(B, Ce, H, W)
    h_top, (h_list, c_list) = drc(E, None, repeats_override=None)
    assert h_top.shape == (B, C, H, W)
    assert len(h_list) == len(c_list) == 3
    for h, c in zip(h_list, c_list):
        assert h.shape == (B, C, H, W)
        assert c.shape == (B, C, H, W)


