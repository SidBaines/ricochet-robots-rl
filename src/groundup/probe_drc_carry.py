"""Controlled probes over recurrent carries from real DRC policy rollouts.

The main comparison is trained versus architecture-matched random-network
carries versus shuffled *training* labels, crossed with linear and one-hidden-
layer MLP probes.  Features are recurrent carries only; observations are
retained for provenance/random unrolling but never enter a probe.

Example::

    .venv-phase2/bin/python -m src.groundup.probe_drc_carry \
      --model model_190000000.pt --bank puzzles.parquet \
      --dataset carry_190000000.npz --output carry_190000000.json \
      --split val --suite-seed 9 --per-stratum 8 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .drc import DRCNet
from .drc_carry import (
    CAPTURE_PROTOCOL_VERSION,
    DATASET_SCHEMA_VERSION,
    SolverContract,
    capture_carry_trajectories,
    load_carry_dataset,
    save_carry_dataset,
)
from .drc_train import (
    DRCTrainConfig,
    _file_sha256,
    _puzzle_fingerprint,
    _sha256_json,
    build_net,
)
from .env import EnvConfig
from .puzzles import Puzzle, joint_stratified_subset, load_puzzles


RESULT_SCHEMA_VERSION = 1
PROBE_PROTOCOL_VERSION = "drc-carry-controlled-probes-v1"


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _network_architecture_contract(net: DRCNet) -> Dict[str, object]:
    return {
        "class": f"{type(net).__module__}.{type(net).__qualname__}",
        "observation_shape": list(net.observation_shape),
        "n_actions": int(net.n_actions),
        "depth": int(net.depth),
        "ticks": int(net.ticks),
        "hidden_channels": int(net.hidden_channels),
        "encoder_relu": bool(net.encoder_relu),
        "encoder_skip": bool(net.encoder_skip),
        "encoder": [
            {
                "in_channels": int(layer.in_channels),
                "out_channels": int(layer.out_channels),
                "kernel_size": list(layer.kernel_size),
                "stride": list(layer.stride),
            }
            for layer in net.encoder
        ],
        "cells": [
            {
                "input_channels": int(cell.input_channels),
                "hidden_channels": int(cell.hidden_channels),
                "kernel_size": int(cell.kernel_size),
                "pool_and_inject": bool(cell.pool_and_inject),
                "pool_projection": str(cell.pool_projection),
                "output_activation": str(cell.output_activation),
                "forget_bias": float(cell.forget_bias),
                "fence": str(cell.fence),
            }
            for cell in net.cells
        ],
        "mlp_in_features": int(net.mlp.in_features),
        "mlp_out_features": int(net.mlp.out_features),
        "state_dict_shapes": {name: list(value.shape) for name, value in net.state_dict().items()},
    }


def build_random_control(
    config: DRCTrainConfig,
    observation_shape: Sequence[int],
    n_actions: int,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> DRCNet:
    """Build a deterministic, architecture-identical random DRC locally seeded."""

    # Module initialization uses PyTorch's global generator.  fork_rng restores
    # it on exit, so running a probe cannot perturb a training process's stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        net = build_net(config, tuple(map(int, observation_shape)), int(n_actions))
    net.to(torch.device(device))
    net._carry_control_seed = int(seed)  # provenance only; not part of state_dict
    return net


def load_schema2_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[DRCNet, DRCTrainConfig, Dict[str, object], Dict[str, object]]:
    """Strictly reconstruct one supported schema-2 ``drc_model`` checkpoint."""

    checkpoint = Path(path).resolve()
    checkpoint_sha256 = _file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError(f"Checkpoint changed while it was being loaded: {checkpoint}")
    if payload.get("kind") != "drc_model" or int(payload.get("schema_version", -1)) != 2:
        raise ValueError(f"Not a supported schema-2 DRC model checkpoint: {checkpoint}")
    required = {"state_dict", "config", "observation_shape", "n_actions"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Schema-2 checkpoint missing reconstruction fields: {missing}")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("Schema-2 checkpoint config must be a JSON-like object")
    stored_config = dict(payload["config"])
    computed_config_fingerprint = _sha256_json(stored_config)
    recorded_config_fingerprint = payload.get("config_fingerprint")
    if recorded_config_fingerprint is not None and recorded_config_fingerprint != computed_config_fingerprint:
        raise ValueError("Checkpoint config_fingerprint does not match stored config")
    try:
        config = DRCTrainConfig(**stored_config)
    except TypeError as exc:
        raise ValueError(f"Unsupported schema-2 DRC config: {exc}") from exc

    with torch.random.fork_rng(devices=[]):
        # Initialization is overwritten; seed only the CPU generator and keep
        # every caller/device RNG stream unchanged.
        torch.random.default_generator.manual_seed(0)
        net = build_net(
            config,
            tuple(int(value) for value in payload["observation_shape"]),
            int(payload["n_actions"]),
        )
    try:
        net.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ValueError(f"Checkpoint tensors do not match reconstructed DRC: {exc}") from exc
    net.to(torch.device(device))

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    timesteps = metadata.get("timesteps")
    if timesteps is None:
        match = re.search(r"(\d+)(?=\.pt$)", checkpoint.name)
        timesteps = int(match.group(1)) if match else None
    architecture_contract = _network_architecture_contract(net)
    provenance = {
        "absolute_path": str(checkpoint),
        "sha256": checkpoint_sha256,
        "format_kind": "drc_model",
        "schema_version": 2,
        "checkpoint_kind": metadata.get("kind", checkpoint.stem),
        "originating_run": stored_config.get("run_name") or checkpoint.parent.name,
        "timesteps": None if timesteps is None else int(timesteps),
        "config_fingerprint": computed_config_fingerprint,
        "architecture_fingerprint": _sha256_json(architecture_contract),
        "architecture_contract": architecture_contract,
    }
    return net, config, payload, provenance


def env_config_from_checkpoint(config: DRCTrainConfig) -> EnvConfig:
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


def suite_provenance(puzzles: Sequence[Puzzle], *, split: str, seed: int, per_stratum: int) -> Dict[str, object]:
    counts = Counter((p.optimal_length, p.robots_moved) for p in puzzles)
    return {
        "selection": "joint_stratified_subset",
        "source_split": split,
        "seed": int(seed),
        "per_stratum": int(per_stratum),
        "count": int(len(puzzles)),
        "per_cell_counts": {f"{length}|{robots}": int(count) for (length, robots), count in sorted(counts.items())},
        # This is the existing stable repository convention, not a new identity.
        "fingerprint": _puzzle_fingerprint(puzzles),
    }


@dataclass(frozen=True)
class LayoutSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_layouts: tuple[str, ...]
    test_layouts: tuple[str, ...]
    fingerprint: str
    manifest: Dict[str, object]


def stable_layout_split(
    layout_hashes: Sequence[str] | np.ndarray,
    *,
    seed: int,
    test_fraction: float = 0.25,
    min_layouts_per_partition: int = 2,
) -> LayoutSplit:
    """Deterministically split whole layouts, invariant to dataset row order."""

    hashes = np.asarray(layout_hashes).astype(str)
    if hashes.ndim != 1:
        raise ValueError("layout_hash must be a one-dimensional row-group array")
    unique = sorted(set(hashes.tolist()))
    if len(unique) < 2:
        raise ValueError("Need at least two distinct layouts for a leakage-free split")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if min_layouts_per_partition < 1:
        raise ValueError("min_layouts_per_partition must be at least one")
    minimum_total = 2 * min_layouts_per_partition
    if len(unique) < minimum_total:
        raise ValueError(
            f"Need at least {minimum_total} layouts for "
            f"{min_layouts_per_partition} layouts in each partition; got {len(unique)}"
        )

    ordered = sorted(
        unique,
        key=lambda layout: (
            hashlib.sha256(f"{int(seed)}:{layout}".encode("utf-8")).digest(),
            layout,
        ),
    )
    requested = int(round(len(unique) * float(test_fraction)))
    n_test = min(
        len(unique) - min_layouts_per_partition,
        max(min_layouts_per_partition, requested),
    )
    test_layouts = tuple(sorted(ordered[:n_test]))
    train_layouts = tuple(sorted(set(unique) - set(test_layouts)))
    train_set, test_set = set(train_layouts), set(test_layouts)
    if train_set & test_set:
        raise AssertionError("Layout leakage in split construction")
    train_indices = np.flatnonzero(np.isin(hashes, list(train_set))).astype(np.int64)
    test_indices = np.flatnonzero(np.isin(hashes, list(test_set))).astype(np.int64)
    if len(train_indices) + len(test_indices) != len(hashes):
        raise AssertionError("Some carry rows were not assigned to a layout partition")

    assignments = [(layout, "test" if layout in test_set else "train") for layout in unique]
    fingerprint = _sha256_json(
        {
            "protocol": "stable-layout-hash-order-v1",
            "seed": int(seed),
            "test_fraction": float(test_fraction),
            "min_layouts_per_partition": int(min_layouts_per_partition),
            "assignments": assignments,
        }
    )
    manifest = {
        "protocol": "stable-layout-hash-order-v1",
        "seed": int(seed),
        "requested_test_fraction": float(test_fraction),
        "fingerprint": fingerprint,
        "train_layout_count": len(train_layouts),
        "test_layout_count": len(test_layouts),
        "train_row_count": int(len(train_indices)),
        "test_row_count": int(len(test_indices)),
        "train_layouts": list(train_layouts),
        "test_layouts": list(test_layouts),
        "layout_leakage": False,
    }
    return LayoutSplit(
        train_indices,
        test_indices,
        train_layouts,
        test_layouts,
        fingerprint,
        manifest,
    )


@dataclass(frozen=True)
class ProbeTarget:
    name: str
    kind: str
    values: np.ndarray
    applicable: np.ndarray
    n_outputs: int
    permutation_group: str

    def __post_init__(self) -> None:
        if self.kind not in {"regression", "multiclass", "binary", "multilabel"}:
            raise ValueError(f"Unknown probe target kind {self.kind!r}")
        values = np.asarray(self.values)
        applicable = np.asarray(self.applicable)
        if applicable.ndim != 1 or len(values) != len(applicable):
            raise ValueError(f"Target {self.name!r} has misaligned values/applicability")
        if self.kind == "multilabel" and (values.ndim != 2 or values.shape[1] != self.n_outputs):
            raise ValueError(f"Multilabel target {self.name!r} has wrong output shape")
        if self.kind != "multilabel" and values.ndim != 1:
            raise ValueError(f"Scalar target {self.name!r} must be one-dimensional")
        if self.n_outputs < 1:
            raise ValueError(f"Target {self.name!r} has no outputs")


def build_probe_targets(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> Dict[str, ProbeTarget]:
    n = len(arrays["chosen_action"])
    all_rows = np.ones(n, dtype=bool)
    n_actions = int(np.asarray(arrays["action_mask"]).shape[1])
    n_robots = n_actions // 4
    width = int(np.asarray(arrays["obs"]).shape[-1])
    height = int(np.asarray(arrays["obs"]).shape[-2])
    exact = np.asarray(arrays["exact_label_known"], dtype=bool)
    reducing = np.asarray(arrays["reducing_label_known"], dtype=bool)
    recent = np.asarray(arrays["recent_actions"])

    def target(
        name: str,
        kind: str,
        values: np.ndarray,
        applicable: np.ndarray,
        n_outputs: int,
        group: Optional[str] = None,
    ) -> ProbeTarget:
        return ProbeTarget(
            name,
            kind,
            np.asarray(values),
            np.asarray(applicable, dtype=bool),
            int(n_outputs),
            group or name,
        )

    targets: Dict[str, ProbeTarget] = {
        "exact_distance": target("exact_distance", "regression", arrays["exact_distance"], exact, 1),
        "optimal_action_set": target(
            "optimal_action_set",
            "multilabel",
            arrays["optimal_action_mask"],
            exact,
            n_actions,
            "exact_action_sets",
        ),
        "reducing_action_set": target(
            "reducing_action_set",
            "multilabel",
            arrays["reducing_action_mask"],
            reducing,
            n_actions,
            # Coverage may be smaller than optimal_action_set when an
            # independent bounded successor check hits max_nodes, so this is
            # independently permuted rather than silently moving its mask.
            "reducing_action_set",
        ),
        "future_success": target("future_success", "binary", arrays["future_success"], all_rows, 1),
        "intended_action": target(
            "intended_action",
            "multiclass",
            arrays["chosen_action"],
            all_rows,
            n_actions,
            "intended_action_tuple",
        ),
        "intended_robot": target(
            "intended_robot",
            "multiclass",
            arrays["intended_robot"],
            all_rows,
            n_robots,
            "intended_action_tuple",
        ),
        "intended_direction": target(
            "intended_direction",
            "multiclass",
            arrays["intended_direction"],
            all_rows,
            4,
            "intended_action_tuple",
        ),
        "horizon_progress": target("horizon_progress", "regression", arrays["horizon_progress"], all_rows, 1),
        "move_index": target("move_index", "regression", arrays["move_index"], all_rows, 1),
        "optimal_distance_progress": target(
            "optimal_distance_progress",
            "regression",
            arrays["optimal_distance_progress"],
            exact,
            1,
        ),
        "landing_cell": target(
            "landing_cell",
            "multiclass",
            arrays["landing_cell"],
            all_rows,
            height * width,
            "chosen_move_outcome",
        ),
        "slide_length": target(
            "slide_length",
            "regression",
            arrays["slide_length"],
            all_rows,
            1,
            "chosen_move_outcome",
        ),
        "moved": target("moved", "binary", arrays["moved"], all_rows, 1, "chosen_move_outcome"),
        "immediate_success": target(
            "immediate_success",
            "binary",
            arrays["immediate_success"],
            all_rows,
            1,
            "chosen_move_outcome",
        ),
    }
    action_classes = n_actions + 2  # actions + explicit BOS + MISSING padding
    for lag in range(recent.shape[1]):
        name = "previous_action" if lag == 0 else f"recent_action_lag_{lag + 1}"
        targets[name] = target(
            name,
            "multiclass",
            recent[:, lag],
            all_rows,
            action_classes,
            f"recent_action_lag_{lag + 1}",
        )
    return targets


def shuffle_training_labels(
    targets: Mapping[str, ProbeTarget],
    train_indices: Sequence[int] | np.ndarray,
    *,
    seed: int,
) -> tuple[Dict[str, ProbeTarget], Dict[str, object]]:
    """Permute only applicable training labels; all non-training rows stay exact.

    Targets in the same ``permutation_group`` use a shared row permutation.
    This preserves intended-action/robot/direction tuples and chosen-move
    outcome tuples. The two exact action masks are independent because their
    availability can differ. Other targets have deterministic independent
    permutations derived from their target/group name.
    """

    if not targets:
        raise ValueError("At least one target is required for a shuffled control")
    train = np.asarray(train_indices, dtype=np.int64)
    if train.ndim != 1 or len(np.unique(train)) != len(train):
        raise ValueError("train_indices must be a unique one-dimensional index set")
    n_rows = len(next(iter(targets.values())).applicable)
    if np.any(train < 0) or np.any(train >= n_rows):
        raise ValueError("train_indices contains a row outside the target arrays")
    if any(len(target.applicable) != n_rows for target in targets.values()):
        raise ValueError("Shuffled-control targets have inconsistent row counts")
    shuffled = {name: replace(target, values=np.asarray(target.values).copy()) for name, target in targets.items()}
    grouped: Dict[str, list[str]] = {}
    for name, target in targets.items():
        grouped.setdefault(target.permutation_group, []).append(name)

    audit_groups: Dict[str, object] = {}
    for group, names in sorted(grouped.items()):
        reference_mask = np.asarray(targets[names[0]].applicable, dtype=bool)
        for name in names[1:]:
            if not np.array_equal(reference_mask, np.asarray(targets[name].applicable, dtype=bool)):
                raise ValueError(f"Structured shuffle group {group!r} has inconsistent applicability masks")
        applicable_train = train[reference_mask[train]]
        rng = np.random.default_rng(_stable_seed(seed, "shuffle", group))
        permutation = rng.permutation(len(applicable_train))
        source = applicable_train[permutation]
        changed: Dict[str, int] = {}
        for name in names:
            values = shuffled[name].values
            original = np.asarray(targets[name].values)
            values[applicable_train] = original[source]
            if values.ndim == 1:
                changed[name] = int(np.sum(values[applicable_train] != original[applicable_train]))
            else:
                changed[name] = int(np.sum(np.any(values[applicable_train] != original[applicable_train], axis=1)))
        audit_groups[group] = {
            "targets": sorted(names),
            "applicable_training_rows": int(len(applicable_train)),
            "permutation_sha256": hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest(),
            "changed_rows": changed,
        }

    train_set = set(train.tolist())
    outside = np.asarray(
        [index for index in range(len(next(iter(targets.values())).applicable)) if index not in train_set],
        dtype=np.int64,
    )
    for name, target in targets.items():
        if not np.array_equal(
            shuffled[name].values[outside],
            np.asarray(target.values)[outside],
            equal_nan=True,
        ):
            raise AssertionError("Shuffled control modified a non-training label")
    return shuffled, {
        "protocol": "per-target-training-only-permutation-v1",
        "seed": int(seed),
        "test_labels_untouched": True,
        "features_untouched": True,
        "split_untouched": True,
        "structured_groups_share_permutation": True,
        "groups": audit_groups,
    }


class _MultiTaskProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        targets: Mapping[str, ProbeTarget],
        *,
        family: str,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if family == "linear":
            self.backbone = nn.Identity()
            width = input_dim
        elif family == "mlp":
            if hidden_size < 1:
                raise ValueError("hidden_size must be positive")
            self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ReLU())
            width = hidden_size
        else:
            raise ValueError(f"Probe family must be 'linear' or 'mlp', got {family!r}")
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    width,
                    target.n_outputs if target.kind in {"multiclass", "multilabel"} else 1,
                )
                for name, target in targets.items()
            }
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {name: head(hidden) for name, head in self.heads.items()}


def _metric(value: Optional[float], reason: Optional[str] = None) -> Dict[str, object]:
    if value is None:
        if not reason:
            raise ValueError("Undefined metrics require an explicit reason")
        return {"value": None, "reason": reason}
    return {"value": float(value), "reason": None}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    if len(y_true) < 2:
        return _metric(None, "fewer than two applicable test rows")
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total <= 0.0:
        return _metric(None, "test target has zero variance")
    residual = float(np.sum((y_true - y_pred) ** 2))
    return _metric(1.0 - residual / total)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return _metric(None, "test target contains only one class")
    recalls = [float(np.mean(y_pred[y_true == cls] == cls)) for cls in classes]
    return _metric(float(np.mean(recalls)))


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    y_true = y_true.astype(bool)
    positives, negatives = int(y_true.sum()), int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return _metric(None, "ROC-AUC requires both test classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    auc = (float(ranks[y_true].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    return _metric(auc)


def _target_metrics(
    target: ProbeTarget, y_true: np.ndarray, raw_prediction: np.ndarray
) -> Dict[str, Dict[str, object]]:
    if target.kind == "regression":
        prediction = raw_prediction.reshape(-1)
        truth = y_true.astype(np.float64).reshape(-1)
        return {
            "mae": _metric(float(np.mean(np.abs(prediction - truth)))),
            "r2": _r2(truth, prediction),
        }
    if target.kind == "multiclass":
        prediction = raw_prediction.argmax(axis=1)
        truth = y_true.astype(np.int64)
        return {"accuracy": _metric(float(np.mean(prediction == truth)))}
    if target.kind == "binary":
        scores = raw_prediction.reshape(-1)
        prediction = (scores >= 0.0).astype(np.int64)
        truth = y_true.astype(np.int64).reshape(-1)
        return {
            "accuracy": _metric(float(np.mean(prediction == truth))),
            "balanced_accuracy": _balanced_accuracy(truth, prediction),
            "roc_auc": _roc_auc(truth, scores),
        }
    truth = y_true.astype(bool)
    prediction = raw_prediction >= 0.0
    positive_rows = truth.any(axis=1)
    if positive_rows.any():
        top = raw_prediction.argmax(axis=1)
        top_in_set = truth[np.arange(len(truth)), top]
        top_metric = _metric(float(top_in_set[positive_rows].mean()))
    else:
        top_metric = _metric(None, "no test row has a positive action in its exact set")
    true_positive = int(np.sum(prediction & truth))
    false_positive = int(np.sum(prediction & ~truth))
    false_negative = int(np.sum(~prediction & truth))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = (
        _metric(2 * true_positive / denominator)
        if denominator
        else _metric(None, "multilabel F1 has no positive truth or prediction")
    )
    return {
        "top1_in_set": top_metric,
        "multilabel_f1": f1,
        "exact_set_accuracy": _metric(float(np.mean(np.all(prediction == truth, axis=1)))),
    }


def _undefined_metrics(target: ProbeTarget, reason: str) -> Dict[str, Dict[str, object]]:
    if target.kind == "regression":
        names = ("mae", "r2")
    elif target.kind == "multiclass":
        names = ("accuracy",)
    elif target.kind == "binary":
        names = ("accuracy", "balanced_accuracy", "roc_auc")
    else:
        names = ("top1_in_set", "multilabel_f1", "exact_set_accuracy")
    return {name: _metric(None, reason) for name in names}


def fit_probe_family(
    features: np.ndarray,
    targets: Mapping[str, ProbeTarget],
    *,
    family: str,
    split: LayoutSplit,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    evaluation_targets: Optional[Mapping[str, ProbeTarget]] = None,
) -> Dict[str, object]:
    """Fit one shared representation with separate typed heads."""

    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Probe epochs/batch_size/lr must all be positive")
    x = np.asarray(features)
    if x.ndim < 2:
        raise ValueError("Probe features must have a row and at least one feature dimension")
    x = x.reshape(len(x), -1).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Probe features contain non-finite values")
    if any(len(target.values) != len(x) for target in targets.values()):
        raise ValueError("Probe target row count does not match features")
    evaluation_targets = targets if evaluation_targets is None else evaluation_targets
    if set(evaluation_targets) != set(targets):
        raise ValueError("Training and evaluation target sets differ")
    for name, target in targets.items():
        evaluation_target = evaluation_targets[name]
        if len(evaluation_target.values) != len(x):
            raise ValueError(f"Evaluation target {name!r} row count does not match features")
        if (target.kind, target.n_outputs) != (evaluation_target.kind, evaluation_target.n_outputs):
            raise ValueError(f"Training/evaluation contracts differ for target {name!r}")
    train = np.asarray(split.train_indices, dtype=np.int64)
    test = np.asarray(split.test_indices, dtype=np.int64)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Both layout partitions need at least one row")
    if np.any(train < 0) or np.any(test < 0) or np.any(train >= len(x)) or np.any(test >= len(x)):
        raise ValueError("Layout split contains a row outside the feature array")
    if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
        raise ValueError("Layout split indices must be unique within each partition")
    if np.intersect1d(train, test).size:
        raise ValueError("Layout split train/test row indices overlap")

    mean = x[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalised = (x - mean) / std
    target_device = torch.device(device)
    x_tensor = torch.as_tensor(normalised, device=target_device)
    train_tensors: Dict[str, torch.Tensor] = {}
    applicable_tensors: Dict[str, torch.Tensor] = {}
    regression_scale: Dict[str, tuple[float, float]] = {}
    for name, target in targets.items():
        values = np.asarray(target.values)
        applicable = np.asarray(target.applicable, dtype=bool)
        applicable_train = train[applicable[train]]
        if target.kind == "regression":
            finite = applicable & np.isfinite(values)
            applicable = finite
            applicable_train = train[applicable[train]]
            if len(applicable_train):
                centre = float(np.mean(values[applicable_train]))
                scale = float(np.std(values[applicable_train]))
                if scale < 1e-6:
                    scale = 1.0
            else:
                centre, scale = 0.0, 1.0
            regression_scale[name] = (centre, scale)
            encoded = (values.astype(np.float32) - centre) / scale
            encoded[~np.isfinite(encoded)] = 0.0
            train_tensors[name] = torch.as_tensor(encoded, device=target_device)
        elif target.kind == "multilabel":
            train_tensors[name] = torch.as_tensor(values.astype(np.float32), device=target_device)
        elif target.kind == "binary":
            train_tensors[name] = torch.as_tensor(values.astype(np.float32), device=target_device)
        else:
            if applicable.any() and (
                int(values[applicable].min()) < 0 or int(values[applicable].max()) >= target.n_outputs
            ):
                raise ValueError(f"Target {name!r} has a class outside its vocabulary")
            train_tensors[name] = torch.as_tensor(values.astype(np.int64), device=target_device)
        applicable_tensors[name] = torch.as_tensor(applicable, device=target_device)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        model = _MultiTaskProbe(x.shape[1], targets, family=family, hidden_size=hidden_size)
    model.to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(int(seed))
    final_losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
        epoch_losses: list[float] = []
        order = rng.permutation(train)
        for start in range(0, len(order), batch_size):
            batch_np = order[start : start + batch_size]
            batch = torch.as_tensor(batch_np, device=target_device)
            outputs = model(x_tensor[batch])
            losses: list[torch.Tensor] = []
            for name, target in targets.items():
                applicable = applicable_tensors[name][batch]
                if not bool(applicable.any()):
                    continue
                prediction = outputs[name][applicable]
                truth = train_tensors[name][batch][applicable]
                if target.kind == "regression":
                    loss = F.mse_loss(prediction.squeeze(-1), truth)
                elif target.kind == "multiclass":
                    loss = F.cross_entropy(prediction, truth)
                elif target.kind == "binary":
                    loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), truth)
                else:
                    loss = F.binary_cross_entropy_with_logits(prediction, truth)
                losses.append(loss)
            if not losses:
                continue
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu()))
        final_losses = epoch_losses
    if final_losses and not np.isfinite(final_losses).all():
        raise RuntimeError("Probe training produced a non-finite loss")

    model.eval()
    with torch.no_grad():
        raw = {name: value.detach().cpu().numpy() for name, value in model(x_tensor[test]).items()}
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise RuntimeError("Probe training produced non-finite test predictions")
    target_results: Dict[str, object] = {}
    for name, target in targets.items():
        eval_target = evaluation_targets[name]
        train_mask = np.asarray(target.applicable, dtype=bool)[train]
        test_mask = np.asarray(eval_target.applicable, dtype=bool)[test]
        if target.kind == "regression":
            train_mask &= np.isfinite(np.asarray(target.values)[train])
            test_mask &= np.isfinite(np.asarray(eval_target.values)[test])
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        if n_train == 0:
            status, reason = "skipped", "no applicable training rows"
            metrics = _undefined_metrics(target, reason)
        elif n_test == 0:
            status, reason = "skipped", "no applicable test rows"
            metrics = _undefined_metrics(target, reason)
        else:
            status, reason = "fitted", None
            prediction = raw[name][test_mask]
            if target.kind == "regression":
                centre, scale = regression_scale[name]
                prediction = prediction * scale + centre
            truth = np.asarray(eval_target.values)[test][test_mask]
            metrics = _target_metrics(target, truth, prediction)
        train_values = np.asarray(target.values)[train][train_mask]
        test_values = np.asarray(eval_target.values)[test][test_mask]
        if target.kind in {"binary", "multiclass"}:
            train_class_count = int(len(np.unique(train_values))) if n_train else 0
            test_class_count = int(len(np.unique(test_values))) if n_test else 0
        else:
            train_class_count = None
            test_class_count = None
        target_results[name] = {
            "status": status,
            "reason": reason,
            "kind": target.kind,
            "n_train": n_train,
            "n_test": n_test,
            "train_coverage": float(n_train / len(train)),
            "test_coverage": float(n_test / len(test)),
            "train_class_count": train_class_count,
            "test_class_count": test_class_count,
            "metrics": metrics,
        }
    return {
        "family": family,
        "input_dimension": int(x.shape[1]),
        "hidden_size": int(hidden_size) if family == "mlp" else None,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "final_training_loss": (float(np.mean(final_losses)) if final_losses else None),
        "targets": target_results,
    }


def _final_carry_components(arrays: Mapping[str, np.ndarray], *, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(arrays[f"{prefix}_carry_out_h"])
    c = np.asarray(arrays[f"{prefix}_carry_out_c"])
    if h.shape != c.shape or h.ndim != 5:
        raise ValueError(f"{prefix} carry-out arrays must have shape (N,D,C,H,W)")
    final_h = h[:, -1].reshape(len(h), -1)
    final_c = c[:, -1].reshape(len(c), -1)
    return final_h, final_c


def carry_representations(arrays: Mapping[str, np.ndarray], *, prefix: str) -> Dict[str, np.ndarray]:
    final_h, final_c = _final_carry_components(arrays, prefix=prefix)
    return {
        "h_final": final_h,
        "c_final": final_c,
        "h_c_final": np.concatenate([final_h, final_c], axis=1),
    }


def run_controlled_probes(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    *,
    split: LayoutSplit,
    probe_seed: int,
    shuffle_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    device: str,
) -> tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    targets = build_probe_targets(arrays, metadata)
    for name, target in targets.items():
        if target.kind not in {"binary", "multiclass"}:
            continue
        applicable = np.asarray(target.applicable, dtype=bool)
        values = np.asarray(target.values)
        all_classes = np.unique(values[applicable])
        train_classes = np.unique(values[split.train_indices][applicable[split.train_indices]])
        if len(all_classes) >= 2 and len(train_classes) < 2:
            raise ValueError(
                f"Layout split leaves target {name!r} with only {len(train_classes)} "
                "training class despite multiple classes in the dataset"
            )
    shuffled_targets, shuffle_audit = shuffle_training_labels(targets, split.train_indices, seed=shuffle_seed)
    trained_h, trained_c = _final_carry_components(arrays, prefix="trained")
    random_h, random_c = _final_carry_components(arrays, prefix="random")

    results: Dict[str, object] = {}
    fit_manifest: Dict[str, object] = {}
    for representation in ("h_final", "c_final", "h_c_final"):
        if representation == "h_final":
            trained_features, random_features = trained_h, random_h
        elif representation == "c_final":
            trained_features, random_features = trained_c, random_c
        else:
            # Materialize the doubled representation only while it is fitted.
            trained_features = np.concatenate([trained_h, trained_c], axis=1)
            random_features = np.concatenate([random_h, random_c], axis=1)
        results[representation] = {}
        fit_manifest[representation] = {}
        for family in ("linear", "mlp"):
            results[representation][family] = {}
            fit_manifest[representation][family] = {}
            controls = {
                "trained": (trained_features, targets, targets),
                "random_network": (random_features, targets, targets),
                # The explicit evaluation target map makes untouched test
                # labels a call-site contract, not merely a shuffle helper detail.
                "shuffled_labels": (trained_features, shuffled_targets, targets),
            }
            for control, (features, training_targets, evaluation_targets) in controls.items():
                # Controls with the same representation/family use identical
                # probe initialization and minibatch order; only the declared
                # feature/label control differs.
                local_seed = _stable_seed(probe_seed, representation, family)
                fit = fit_probe_family(
                    features,
                    training_targets,
                    evaluation_targets=evaluation_targets,
                    family=family,
                    split=split,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_size=hidden_size,
                    seed=local_seed,
                    device=device,
                )
                # Keep the required machine-readable path exact:
                # representation -> family -> control -> target.
                results[representation][family][control] = fit["targets"]
                fit_manifest[representation][family][control] = {
                    key: value for key, value in fit.items() if key != "targets"
                }
    return results, shuffle_audit, fit_manifest


def _git_provenance() -> Dict[str, object]:
    repository = Path(__file__).resolve().parents[2]

    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repository,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "head": head,
        "tracked_worktree_dirty": None if status is None else bool(status),
        "implementation_file_sha256": {
            "drc_carry.py": _file_sha256(Path(__file__).with_name("drc_carry.py")),
            "probe_drc_carry.py": _file_sha256(Path(__file__)),
        },
    }


def _dataset_contract(
    *,
    checkpoint: Mapping[str, object],
    bank: Mapping[str, object],
    suite: Mapping[str, object],
    env_config: EnvConfig,
    solver_contract: SolverContract,
    rollout_seed: int,
    random_network_seed: int,
    recent_action_lags: int,
    carry_dtype: str,
) -> Dict[str, object]:
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "capture_protocol": CAPTURE_PROTOCOL_VERSION,
        "checkpoint": {
            key: checkpoint[key]
            for key in (
                "absolute_path",
                "sha256",
                "config_fingerprint",
                "architecture_fingerprint",
            )
        },
        "bank": dict(bank),
        "suite": {
            key: suite[key]
            for key in (
                "selection",
                "source_split",
                "seed",
                "per_stratum",
                "count",
                "per_cell_counts",
                "fingerprint",
            )
        },
        "environment": dataclasses.asdict(env_config),
        "solver": solver_contract.as_dict(),
        "rollout_seed": int(rollout_seed),
        "random_network_seed": int(random_network_seed),
        "recent_action_lags": int(recent_action_lags),
        "carry_dtype": carry_dtype,
    }


def load_or_capture_dataset(
    path: str | Path,
    *,
    net: DRCNet,
    random_net: DRCNet,
    puzzles: Sequence[Puzzle],
    env_config: EnvConfig,
    checkpoint_provenance: Mapping[str, object],
    bank_provenance: Mapping[str, object],
    suite: Mapping[str, object],
    solver_contract: SolverContract,
    rollout_seed: int,
    random_network_seed: int,
    recent_action_lags: int,
    carry_dtype: str,
    force: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    dataset_path = Path(path)
    contract = _dataset_contract(
        checkpoint=checkpoint_provenance,
        bank=bank_provenance,
        suite=suite,
        env_config=env_config,
        solver_contract=solver_contract,
        rollout_seed=rollout_seed,
        random_network_seed=random_network_seed,
        recent_action_lags=recent_action_lags,
        carry_dtype=carry_dtype,
    )
    if dataset_path.exists() and not force:
        return load_carry_dataset(dataset_path, expected_contract=contract)

    dtype = np.float16 if carry_dtype == "float16" else np.float32
    arrays, capture_metadata = capture_carry_trajectories(
        net,
        random_net,
        puzzles,
        env_config,
        solver_contract=solver_contract,
        rollout_seed=rollout_seed,
        recent_action_lags=recent_action_lags,
        carry_dtype=dtype,
    )
    metadata = {
        **capture_metadata,
        "contract": contract,
        "checkpoint": dict(checkpoint_provenance),
        "bank": dict(bank_provenance),
        "suite": dict(suite),
        "git": _git_provenance(),
    }
    saved_metadata = save_carry_dataset(dataset_path, arrays, metadata)
    return arrays, saved_metadata


def probe_checkpoint(
    model_path: str | Path,
    bank_path: str | Path,
    dataset_path: str | Path,
    *,
    split_name: str = "val",
    suite_seed: int = 9,
    per_stratum: int = 8,
    rollout_seed: int = 0,
    solver_max_depth: int = 64,
    solver_max_nodes: int = 2_000_000,
    random_network_seed: int = 1_729,
    layout_split_seed: int = 20_260_813,
    layout_test_fraction: float = 0.25,
    min_layouts_per_partition: int = 2,
    shuffle_seed: int = 31_415,
    probe_seed: int = 27_182,
    probe_epochs: int = 30,
    probe_batch_size: int = 256,
    probe_learning_rate: float = 1e-3,
    probe_hidden_size: int = 64,
    recent_action_lags: int = 3,
    carry_dtype: str = "float16",
    device: str = "cpu",
    force_dataset: bool = False,
) -> Dict[str, object]:
    if split_name not in {"val", "train"}:
        raise ValueError("Carry studies are restricted to validation or training data")
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive for a meaningful suite")
    if carry_dtype not in {"float16", "float32"}:
        raise ValueError("carry_dtype must be float16 or float32")
    model = Path(model_path).resolve()
    bank = Path(bank_path).resolve()
    dataset = Path(dataset_path).resolve()
    if dataset in {model, bank}:
        raise ValueError("Dataset path must differ from the checkpoint and puzzle bank")

    net, config, _payload, checkpoint = load_schema2_checkpoint(model, device=device)
    if not config.canonical_identity:
        raise ValueError("Checkpoint is not from the canonical-identity DRC protocol")
    bank_provenance = {
        "absolute_path": str(bank),
        "sha256": _file_sha256(bank),
        "source_split": split_name,
        "split_contract": {
            "eval_fraction": float(config.eval_fraction),
            "split_seed": int(config.split_seed),
            "val_test_seed": int(config.val_test_seed),
        },
    }
    pool = load_puzzles(
        bank,
        split_name,
        eval_fraction=config.eval_fraction,
        split_seed=config.split_seed,
        val_test_seed=config.val_test_seed,
    )
    if _file_sha256(bank) != bank_provenance["sha256"]:
        raise RuntimeError(f"Puzzle bank changed while it was being loaded: {bank}")
    puzzles = joint_stratified_subset(pool, per_stratum=per_stratum, seed=suite_seed)
    if not puzzles:
        raise ValueError("Joint-stratified suite selection returned no puzzles")
    suite = suite_provenance(puzzles, split=split_name, seed=suite_seed, per_stratum=per_stratum)
    env_config = env_config_from_checkpoint(config)
    solver_contract = SolverContract(max_depth=solver_max_depth, max_nodes=solver_max_nodes)
    random_net = build_random_control(
        config,
        net.observation_shape,
        net.n_actions,
        seed=random_network_seed,
        device=device,
    )
    if _sha256_json(_network_architecture_contract(random_net)) != checkpoint["architecture_fingerprint"]:
        raise AssertionError("Random-control architecture fingerprint differs from checkpoint")
    if all(
        torch.equal(net.state_dict()[name].detach().cpu(), value.detach().cpu())
        for name, value in random_net.state_dict().items()
    ):
        raise ValueError("Random-control weights exactly equal the trained checkpoint; choose a different seed")
    arrays, dataset_metadata = load_or_capture_dataset(
        dataset,
        net=net,
        random_net=random_net,
        puzzles=puzzles,
        env_config=env_config,
        checkpoint_provenance=checkpoint,
        bank_provenance=bank_provenance,
        suite=suite,
        solver_contract=solver_contract,
        rollout_seed=rollout_seed,
        random_network_seed=random_network_seed,
        recent_action_lags=recent_action_lags,
        carry_dtype=carry_dtype,
        force=force_dataset,
    )
    layout_split = stable_layout_split(
        arrays["layout_hash"],
        seed=layout_split_seed,
        test_fraction=layout_test_fraction,
        min_layouts_per_partition=min_layouts_per_partition,
    )
    representations, shuffle_audit, fit_manifest = run_controlled_probes(
        arrays,
        dataset_metadata,
        split=layout_split,
        probe_seed=probe_seed,
        shuffle_seed=shuffle_seed,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        hidden_size=probe_hidden_size,
        device=device,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "study_type": "observational_drc_carry_state_probes",
        "causal_interventions": False,
        "git": _git_provenance(),
        "checkpoint": checkpoint,
        "bank": bank_provenance,
        "suite": suite,
        "dataset": {
            "absolute_path": str(dataset),
            "content_sha256": dataset_metadata["content_sha256"],
            "schema_version": dataset_metadata["schema_version"],
            "n_rows": dataset_metadata["n_rows"],
            "n_episodes": dataset_metadata["n_episodes"],
            "storage": dataset_metadata["storage"],
            "label_schema": dataset_metadata["labels"],
            "capture_protocol": dataset_metadata["capture_protocol"],
            "contract": dataset_metadata["contract"],
            "contract_sha256": _sha256_json(dataset_metadata["contract"]),
            "generation_git": dataset_metadata["git"],
        },
        "random_control": dataset_metadata["random_control"],
        "solver_contract": dataset_metadata["solver_contract"],
        "solver_coverage": dataset_metadata["solver_coverage"],
        "runtime": dataset_metadata["runtime"],
        "layout_split": layout_split.manifest,
        "hyperparameters": {
            "suite_seed": int(suite_seed),
            "per_stratum": int(per_stratum),
            "rollout_seed": int(rollout_seed),
            "solver_max_depth": int(solver_max_depth),
            "solver_max_nodes": int(solver_max_nodes),
            "random_network_seed": int(random_network_seed),
            "layout_split_seed": int(layout_split_seed),
            "layout_test_fraction": float(layout_test_fraction),
            "min_layouts_per_partition": int(min_layouts_per_partition),
            "shuffle_seed": int(shuffle_seed),
            "probe_seed": int(probe_seed),
            "probe_epochs": int(probe_epochs),
            "probe_batch_size": int(probe_batch_size),
            "probe_learning_rate": float(probe_learning_rate),
            "probe_hidden_size": int(probe_hidden_size),
            "recent_action_lags": int(recent_action_lags),
            "carry_dtype": carry_dtype,
            "device": device,
        },
        "shuffled_label_control": shuffle_audit,
        "probe_fit_manifest": fit_manifest,
        "results": representations,
    }


def _atomic_json(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact observational probes of DRC recurrent trajectory carries")
    parser.add_argument("--model", dest="model_path", required=True)
    parser.add_argument("--bank", dest="bank_path", required=True)
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", dest="split_name", choices=("val", "train"), default="val")
    parser.add_argument("--suite-seed", type=int, default=9)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--solver-max-depth", type=int, default=64)
    parser.add_argument("--solver-max-nodes", type=int, default=2_000_000)
    parser.add_argument("--random-network-seed", type=int, default=1_729)
    parser.add_argument("--layout-split-seed", type=int, default=20_260_813)
    parser.add_argument("--layout-test-fraction", type=float, default=0.25)
    parser.add_argument("--min-layouts-per-partition", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int, default=31_415)
    parser.add_argument("--probe-seed", type=int, default=27_182)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-hidden-size", type=int, default=64)
    parser.add_argument("--recent-action-lags", type=int, default=3)
    parser.add_argument("--carry-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-dataset", action="store_true")
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    arguments = vars(args).copy()
    output_path = arguments.pop("output")
    output = Path(output_path).resolve()
    protected_inputs = {
        Path(arguments["model_path"]).resolve(),
        Path(arguments["bank_path"]).resolve(),
        Path(arguments["dataset_path"]).resolve(),
    }
    if output in protected_inputs:
        parser.error("--output must differ from --model, --bank, and --dataset")
    result = probe_checkpoint(**arguments)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "LayoutSplit",
    "ProbeTarget",
    "_build_argument_parser",
    "build_probe_targets",
    "build_random_control",
    "carry_representations",
    "fit_probe_family",
    "load_or_capture_dataset",
    "load_schema2_checkpoint",
    "probe_checkpoint",
    "run_controlled_probes",
    "shuffle_training_labels",
    "stable_layout_split",
    "suite_provenance",
]
