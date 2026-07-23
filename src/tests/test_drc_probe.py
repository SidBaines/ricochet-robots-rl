from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from src.groundup.collate_drc_probes import load_probe_series, write_outputs
from src.groundup.drc import DRCNet
from src.groundup.drc_train import DRCTrainConfig, _save_model
from src.groundup.probe_drc_landing import (
    _array_fingerprint,
    extract_zero_state_features,
    load_drc_checkpoint,
)
from src.groundup.probe_landing import train_probe


def probe_net(seed: int = 0) -> DRCNet:
    torch.manual_seed(seed)
    return DRCNet(
        observation_shape=(8, 6, 6),
        n_actions=8,
        depth=2,
        ticks=3,
        hidden_channels=8,
        encoder_channels=(8, 8),
        encoder_kernels=(3, 3),
        encoder_strides=(1, 1),
        mlp_hidden=32,
    )


def test_tick_trace_reconstructs_forward_without_changing_outputs():
    net = probe_net()
    obs = torch.rand(5, 8, 6, 6)
    state = net.initial_state(5)
    with torch.no_grad():
        logits, values, new_state = net(obs, state)
        encoded, ticks, readout, traced_state = net.recurrent_trace(obs, state)
        hidden = F.relu(net.mlp(readout.flatten(1)))

    assert encoded.shape == (5, 8, 6, 6)
    assert len(ticks) == 3
    assert all(tick.shape == encoded.shape for tick in ticks)
    assert torch.equal(logits, net.actor(hidden))
    assert torch.equal(values, net.critic(hidden).squeeze(-1))
    assert torch.equal(new_state.h, traced_state.h)
    assert torch.equal(new_state.c, traced_state.c)

    features = extract_zero_state_features(net, obs.numpy(), device="cpu", batch_size=2)
    assert list(features) == ["encoder", "tick_1", "tick_2", "tick_3", "readout"]
    assert all(value.shape == encoded.shape for value in features.values())


def test_checkpoint_reconstruction_and_random_control_are_reproducible(tmp_path):
    config = DRCTrainConfig(
        seed=17,
        drc_depth=2,
        drc_ticks=3,
        drc_hidden_channels=8,
        drc_encoder_channels=(8, 8),
        drc_encoder_kernels=(3, 3),
        drc_encoder_strides=(1, 1),
        drc_mlp_hidden=32,
    )
    net = probe_net(seed=17)
    with torch.no_grad():
        next(net.parameters()).add_(1.0)
    path = tmp_path / "model_5000000.pt"
    _save_model(net, config, path, metadata={"timesteps": 5_000_000})

    loaded, _, payload = load_drc_checkpoint(path, device="cpu")
    control1, _, _ = load_drc_checkpoint(path, device="cpu", random_init=True)
    control2, _, _ = load_drc_checkpoint(path, device="cpu", random_init=True)
    assert payload["metadata"]["timesteps"] == 5_000_000
    assert torch.equal(next(loaded.parameters()), next(net.parameters()))
    assert torch.equal(next(control1.parameters()), next(control2.parameters()))
    assert not torch.equal(next(control1.parameters()), next(loaded.parameters()))


def test_linear_probe_recovers_synthetic_spatial_landing_code():
    generator = np.random.default_rng(4)
    n, actions, height, width = 512, 4, 4, 4
    labels = generator.integers(0, height * width, size=(n, actions), dtype=np.int64)
    features = torch.zeros(n, actions, height, width)
    for row in range(n):
        for action in range(actions):
            features[row, action].flatten()[labels[row, action]] = 1.0
    lengths = np.ones_like(labels)
    result = train_probe(
        features,
        labels,
        lengths,
        epochs=100,
        lr=5e-2,
        seed=3,
        device="cpu",
    )
    assert result["overall_acc"] > 0.98
    assert result["by_slide_length"]["short(1-3)"]["acc"] > 0.98


def test_dataset_fingerprint_and_probe_collation(tmp_path):
    arrays = {
        "obs": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
        "labels": np.arange(4, dtype=np.int64).reshape(2, 2),
        "lengths": np.ones((2, 2), dtype=np.int64),
    }
    metadata = {"schema_version": 1, "seed": 0}
    first = _array_fingerprint(arrays, metadata)
    second = _array_fingerprint(arrays, metadata)
    assert first == second
    arrays["obs"][0, 0, 0, 0] += 1
    assert _array_fingerprint(arrays, metadata) != first

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    representations = {}
    for name, value in (("encoder", 0.05), ("tick_1", 0.08), ("tick_2", 0.12), ("tick_3", 0.2), ("readout", 0.18)):
        representations[name] = {
            "overall_acc": value + 0.1,
            "by_slide_length": {
                "blocked(0)": {"acc": value + 0.2, "n": 10},
                "short(1-3)": {"acc": value + 0.1, "n": 10},
                "mid(4-7)": {"acc": value, "n": 10},
                "long(8+)": {"acc": value / 2, "n": 10},
            },
            "n_test_states": 4,
        }
    for steps in (5_000_000, 10_000_000):
        row = {
            "schema_version": 1,
            "probe_type": "drc_landing_zero_state",
            "model": f"model_{steps}.pt",
            "timesteps": steps,
            "random_init_control": False,
            "dataset_sha256": "fixed",
            "representations": representations,
        }
        (run_dir / f"probe_t{steps}.json").write_text(json.dumps(row))
    (run_dir / "curve.jsonl").write_text(
        "\n".join(
            json.dumps({"timesteps": steps, "val_det": {"success_rate": success}})
            for steps, success in ((5_000_000, 0.4), (10_000_000, 0.5))
        )
        + "\n"
    )

    rows = load_probe_series(run_dir)
    summary, chart = write_outputs(run_dir, rows)
    assert len(rows) == 2
    assert "tick_3 long" in summary.read_text()
    assert chart.exists() and chart.stat().st_size > 0
