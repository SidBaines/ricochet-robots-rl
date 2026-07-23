"""Tick-wise landing-cell probes for helper-free DRC checkpoints.

The primary probe starts every sampled observation from zero recurrent state,
then decodes slide landing cells from:

* the observation encoder output;
* the final-depth hidden map after every DRC tick; and
* the encoder-skip map consumed by the faithful flattened readout.

This separates information already present in local perceptual features from
information made linearly available by repeated computation. Probes measure
decodability, not causal use.

Usage:
    .venv-phase2/bin/python -m src.groundup.probe_drc_landing \
        --model artifacts/groundup/drc33_p1_12x12_planefree/model_5000000.pt \
        --dataset artifacts/groundup/drc33_p1_12x12_planefree/probe_dataset.npz \
        --output artifacts/groundup/drc33_p1_12x12_planefree/probe_t5000000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .drc import DRCNet
from .drc_train import DRCTrainConfig, build_net
from .env import EnvConfig, RicochetEnv
from .probe_landing import collect_states, train_probe
from .puzzles import load_puzzles


SCHEMA_VERSION = 1


def _array_fingerprint(arrays: Dict[str, np.ndarray], metadata: Dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _env_config(config: DRCTrainConfig) -> EnvConfig:
    return EnvConfig(
        horizon=config.horizon,
        step_penalty=config.step_penalty,
        goal_reward=config.goal_reward,
        shaping_coef=config.shaping_coef,
        gamma=config.gamma,
        time_channel=config.time_channel,
        canonical_identity=config.canonical_identity,
        stop_planes=config.stop_planes,
        distance_plane=config.distance_plane,
    )


def load_or_create_dataset(
    path: str | Path,
    *,
    config: DRCTrainConfig,
    split: str,
    n_states: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Create once, then require every checkpoint to use the exact same data."""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            obs = payload["obs"]
            labels = payload["labels"]
            lengths = payload["lengths"]
            metadata = json.loads(str(payload["metadata"]))
        arrays = {"obs": obs, "labels": labels, "lengths": lengths}
        actual = _array_fingerprint(arrays, {k: v for k, v in metadata.items() if k != "sha256"})
        if actual != metadata.get("sha256"):
            raise ValueError(f"Probe dataset fingerprint mismatch: {path}")
        expected = {
            "bank_path": config.bank_path,
            "split": split,
            "n_states": int(n_states),
            "seed": int(seed),
        }
        changed = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        if changed:
            raise ValueError(f"Probe dataset contract changed in fields: {changed}")
        return obs, labels, lengths, metadata

    puzzles = load_puzzles(
        config.bank_path,
        split,
        eval_fraction=config.eval_fraction,
        split_seed=config.split_seed,
        val_test_seed=config.val_test_seed,
    )
    env = RicochetEnv(puzzles, _env_config(config), seed=seed)
    if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
        raise ValueError("DRC landing probe currently requires a tensor observation")
    obs, labels, lengths = collect_states(env, n_states, seed)
    metadata: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "bank_path": config.bank_path,
        "split": split,
        "n_states": int(n_states),
        "seed": int(seed),
        "observation_shape": list(obs.shape[1:]),
        "label_shape": list(labels.shape[1:]),
    }
    arrays = {"obs": obs, "labels": labels, "lengths": lengths}
    metadata["sha256"] = _array_fingerprint(arrays, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=obs,
        labels=labels,
        lengths=lengths,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return obs, labels, lengths, metadata


@torch.no_grad()
def extract_zero_state_features(
    net: DRCNet,
    obs: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> Dict[str, torch.Tensor]:
    """Return encoder, per-tick, and readout spatial maps on a zero state."""
    target = torch.device(device)
    net = net.to(target)
    was_training = net.training
    net.eval()
    chunks: Dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(obs), batch_size):
        batch = torch.as_tensor(obs[start : start + batch_size], device=target)
        encoded, ticks, readout_map, _state = net.recurrent_trace(batch, state=None)
        features = {
            "encoder": encoded,
            **{f"tick_{index}": value for index, value in enumerate(ticks, start=1)},
            "readout": readout_map,
        }
        for name, value in features.items():
            chunks.setdefault(name, []).append(value.detach().cpu())
    net.train(was_training)
    return {name: torch.cat(values) for name, values in chunks.items()}


def load_drc_checkpoint(
    path: str | Path,
    *,
    device: str,
    random_init: bool = False,
) -> tuple[DRCNet, DRCTrainConfig, Dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in payload or "observation_shape" not in payload or "n_actions" not in payload:
        raise ValueError(f"Checkpoint lacks the DRC reconstruction contract: {path}")
    config = DRCTrainConfig(**payload["config"])
    torch.manual_seed(config.seed)
    net = build_net(config, tuple(payload["observation_shape"]), int(payload["n_actions"]))
    if not random_init:
        net.load_state_dict(payload["state_dict"])
    net.to(device)
    return net, config, payload


def probe_checkpoint(
    model_path: str | Path,
    dataset_path: str | Path,
    *,
    split: str = "val",
    n_states: int = 16_000,
    seed: int = 0,
    device: str = "cpu",
    random_init: bool = False,
    probe_epochs: int = 30,
) -> Dict[str, object]:
    net, config, payload = load_drc_checkpoint(
        model_path,
        device=device,
        random_init=random_init,
    )
    obs, labels, lengths, dataset_metadata = load_or_create_dataset(
        dataset_path,
        config=config,
        split=split,
        n_states=n_states,
        seed=seed,
    )
    if tuple(obs.shape[1:]) != tuple(net.observation_shape):
        raise ValueError(
            f"Probe dataset observation shape {obs.shape[1:]} != "
            f"checkpoint {net.observation_shape}"
        )
    features = extract_zero_state_features(net, obs, device=device)
    for name, value in features.items():
        if value.shape[-2:] != obs.shape[-2:]:
            raise ValueError(
                f"Representation {name} grid {tuple(value.shape[-2:])} does not align "
                f"with landing labels {tuple(obs.shape[-2:])}"
            )
    representations = {
        name: train_probe(
            value,
            labels,
            lengths,
            epochs=probe_epochs,
            seed=seed,
            device=device,
        )
        for name, value in features.items()
    }
    metadata = payload.get("metadata", {})
    timesteps = metadata.get("timesteps")
    if timesteps is None:
        match = re.search(r"(\d+)(?=\.pt$)", str(model_path))
        timesteps = int(match.group(1)) if match else None
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_type": "drc_landing_zero_state",
        "model": str(model_path),
        "timesteps": timesteps,
        "random_init_control": bool(random_init),
        "split": split,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_metadata["sha256"],
        "n_states": int(len(obs)),
        "representations": representations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-states", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = probe_checkpoint(
        args.model,
        args.dataset,
        split=args.split,
        n_states=args.n_states,
        seed=args.seed,
        device=args.device,
        random_init=args.random_init,
        probe_epochs=args.probe_epochs,
    )
    text = json.dumps(result, indent=2)
    Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
