"""Linear probes: can the trunk's activations predict slide landing cells?

For a checkpoint of the flat (GridTrunk) policy, freeze the network, extract
the spatial activations after the residual blocks, and train a linear 1x1-conv
probe to predict, for every (robot, direction), the cell that slide would end
on — i.e. exactly the stop planes the plane-free obs no longer contains.
Accuracy is exact-landing-cell argmax, reported overall and stratified by
slide length (long slides are the relational computation of interest).

Controls: run with --random-init to probe an untrained network of the same
architecture (distinguishes "circuit formed by training" from "linearly
decodable from any conv features"). Probes measure decodability, not use.

Usage:
    .venv-phase2/bin/python -m src.groundup.probe_landing \
        --model artifacts/groundup/p1_12x12/model_5000000.zip \
        --config artifacts/groundup/p1_12x12/config.json \
        --n-states 16000 --device cpu --output probe_5M.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import EnvConfig, RicochetEnv
from .puzzles import load_puzzles


def collect_states(env: RicochetEnv, n_states: int, seed: int):
    """Sample diverse states: random puzzle, random start distance, 0-6 random
    masked steps. Returns (obs array, landing-cell labels, slide lengths)."""
    rng = np.random.default_rng(seed)
    num_actions = env.action_space.n
    obs_list, labels, lengths = [], [], []
    while len(obs_list) < n_states:
        idx = int(rng.integers(len(env.puzzles)))
        obs, _ = env.reset(options={"puzzle_index": idx})
        for _ in range(int(rng.integers(0, 7))):
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            obs, _, term, trunc, _ = env.step(int(legal[rng.integers(len(legal))]))
            if term or trunc:
                break
        else:
            pass
        if env._episode_done:
            continue
        row_labels = np.zeros(num_actions, dtype=np.int64)
        row_lengths = np.zeros(num_actions, dtype=np.int64)
        for action in range(num_actions):
            rid, direction = action // 4, action % 4
            sr, sc = env._slide_from(env._robots, rid, direction, env._puzzle)
            r0, c0 = env._robots[rid]
            row_labels[action] = sr * env.width + sc
            row_lengths[action] = abs(sr - r0) + abs(sc - c0)
        obs_list.append(obs)
        labels.append(row_labels)
        lengths.append(row_lengths)
    return np.stack(obs_list), np.stack(labels), np.stack(lengths)


@torch.no_grad()
def extract_features(policy, obs: np.ndarray, device: str, batch: int = 512) -> torch.Tensor:
    """Spatial activations after the residual blocks of GridTrunk, (N,C,H,W)."""
    fx = policy.features_extractor
    outs = []
    for i in range(0, len(obs), batch):
        x = torch.as_tensor(obs[i : i + batch], device=device)
        outs.append(fx.blocks(fx.stem(x)).cpu())
    return torch.cat(outs)


def train_probe(
    feats: torch.Tensor, labels: np.ndarray, lengths: np.ndarray,
    *, test_frac: float = 0.25, epochs: int = 30, lr: float = 5e-3, seed: int = 0,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    n, channels, height, width = feats.shape
    n_actions = labels.shape[1]
    n_test = int(n * test_frac)
    train_x, test_x = feats[:-n_test].to(device), feats[-n_test:].to(device)
    labels_t = torch.as_tensor(labels)
    train_y, test_y = labels_t[:-n_test].to(device), labels_t[-n_test:].to(device)

    probe = nn.Conv2d(channels, n_actions, kernel_size=1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    batch = 1024
    for _epoch in range(epochs):
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), batch):
            sel = perm[i : i + batch]
            logits = probe(train_x[sel]).flatten(2)  # (B, A, H*W)
            loss = F.cross_entropy(
                logits.reshape(-1, height * width), train_y[sel].reshape(-1)
            )
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        logits = probe(test_x).flatten(2)
        pred = logits.argmax(dim=2).cpu().numpy()
    truth = labels[-n_test:]
    test_lengths = lengths[-n_test:]
    correct = pred == truth
    buckets = {"blocked(0)": test_lengths == 0, "short(1-3)": (test_lengths >= 1) & (test_lengths <= 3),
               "mid(4-7)": (test_lengths >= 4) & (test_lengths <= 7), "long(8+)": test_lengths >= 8}
    return {
        "overall_acc": round(float(correct.mean()), 4),
        "by_slide_length": {
            k: {"acc": round(float(correct[m].mean()), 4) if m.any() else None, "n": int(m.sum())}
            for k, m in buckets.items()
        },
        "n_test_states": int(n_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-states", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-init", action="store_true",
                        help="probe an untrained net of the same architecture (control)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train_config = json.loads(Path(args.config).read_text())
    env_config = EnvConfig(
        horizon=train_config["horizon"], step_penalty=train_config["step_penalty"],
        goal_reward=train_config["goal_reward"], shaping_coef=train_config["shaping_coef"],
        gamma=train_config["gamma"], time_channel=train_config["time_channel"],
        canonical_identity=train_config["canonical_identity"],
        stop_planes=train_config.get("stop_planes", False),
        distance_plane=train_config.get("distance_plane", False),
        rgb_obs=train_config.get("rgb_obs", False),
        rgb_cell_px=train_config.get("rgb_cell_px", 5),
    )
    if env_config.rgb_obs:
        raise SystemExit("probe currently supports plane obs only")
    puzzles = load_puzzles(
        train_config["bank_path"], args.split,
        eval_fraction=train_config["eval_fraction"], split_seed=train_config["split_seed"],
        val_test_seed=train_config["val_test_seed"],
    )
    env = RicochetEnv(puzzles, env_config, seed=args.seed)

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(args.model, device=args.device)
    if args.random_init:
        for module in model.policy.features_extractor.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.reset_parameters()

    obs, labels, lengths = collect_states(env, args.n_states, args.seed)
    feats = extract_features(model.policy, obs, args.device)
    result = train_probe(feats, labels, lengths, seed=args.seed, device=args.device)
    result["model"] = args.model
    result["random_init_control"] = bool(args.random_init)
    result["split"] = args.split
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
