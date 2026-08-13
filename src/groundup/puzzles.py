"""Load solver-labelled puzzle banks into a lean in-memory table.

Reuses the existing parquet banks (verified: stored plans replay to the goal,
BFS lengths re-derive exactly) and reproduces the repo's layout-hash split rule
byte-for-byte so train/held-out isolation matches earlier phases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.env.puzzle_bank import deserialize_layout_blob


@dataclass(frozen=True)
class Puzzle:
    h_walls: np.ndarray  # (H+1, W) bool
    v_walls: np.ndarray  # (H, W+1) bool
    robots: Tuple[Tuple[int, int], ...]  # original bank order
    goal: Tuple[int, int]
    target: int
    optimal_length: int
    actions: Tuple[Tuple[int, int], ...]  # stored optimal plan, (robot_id, direction)
    robots_moved: int
    layout_hash: str
    seed: int

    @property
    def num_robots(self) -> int:
        return len(self.robots)

    @property
    def height(self) -> int:
        return self.h_walls.shape[0] - 1

    @property
    def width(self) -> int:
        return self.h_walls.shape[1]


def _split_fraction(split_seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{split_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def load_puzzles(
    parquet_path: str | Path,
    split: str = "all",
    *,
    eval_fraction: float = 0.25,
    split_seed: int = 20_260_710,
    val_test_seed: int = 20_260_711,
) -> List[Puzzle]:
    """Load a bank partition and return puzzles for the requested split.

    Splits: 'all', 'train', 'val', 'test'. Train vs held-out uses the same
    sha256(layout_hash) rule as ``BankSampler._split_mask``; the held-out rows
    are then subdivided into val/test with ``val_test_seed`` (50/50).
    """
    if split not in {"all", "train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    df = pd.read_parquet(parquet_path)
    puzzles: List[Puzzle] = []
    for row in df.itertuples(index=False):
        layout_hash = str(row.layout_hash)
        if split != "all":
            is_eval = _split_fraction(split_seed, layout_hash) < eval_fraction
            if split == "train":
                if is_eval:
                    continue
            else:
                if not is_eval:
                    continue
                is_test = _split_fraction(val_test_seed, layout_hash) < 0.5
                if split == "val" and is_test:
                    continue
                if split == "test" and not is_test:
                    continue
        blob = row.layout_blob
        if isinstance(blob, str):
            blob = bytes.fromhex(blob)
        layout = deserialize_layout_blob(blob)
        actions = tuple((int(a) // 4, int(a) % 4) for a in row.actions_encoded)
        puzzles.append(
            Puzzle(
                h_walls=np.ascontiguousarray(layout.h_walls, dtype=bool),
                v_walls=np.ascontiguousarray(layout.v_walls, dtype=bool),
                robots=tuple(
                    (int(r), int(c))
                    for _, (r, c) in sorted(layout.robot_positions.items())
                ),
                goal=(int(layout.goal_position[0]), int(layout.goal_position[1])),
                target=int(layout.target_robot),
                optimal_length=int(row.optimal_length),
                actions=actions,
                robots_moved=int(row.robots_moved),
                layout_hash=layout_hash,
                seed=int(row.seed),
            )
        )
    return puzzles


def stratified_subset(
    puzzles: Sequence[Puzzle],
    per_length: int,
    *,
    seed: int = 0,
) -> List[Puzzle]:
    """Deterministic per-optimal-length subset (for fixed eval suites)."""
    rng = np.random.default_rng(seed)
    by_length: Dict[int, List[int]] = {}
    for i, p in enumerate(puzzles):
        by_length.setdefault(p.optimal_length, []).append(i)
    chosen: List[Puzzle] = []
    for length in sorted(by_length):
        idxs = np.array(by_length[length])
        take = min(per_length, len(idxs))
        pick = rng.choice(idxs, size=take, replace=False)
        chosen.extend(puzzles[int(i)] for i in sorted(pick))
    return chosen


def joint_stratified_subset(
    puzzles: Sequence[Puzzle],
    per_stratum: int,
    *,
    seed: int = 0,
) -> List[Puzzle]:
    """Select up to ``per_stratum`` puzzles for each observed hard stratum.

    Strata are the actual ``(optimal_length, robots_moved)`` pairs in
    ``puzzles``. Missing Cartesian-product cells are not invented, and every
    puzzle belongs to exactly one group. Each group gets a stable derived RNG
    seed, so adding or removing a sparse stratum cannot perturb other groups.
    For a fixed input order and seed the returned suite is deterministic.
    """
    if per_stratum < 0:
        raise ValueError(f"per_stratum must be non-negative, got {per_stratum}")

    by_stratum: Dict[Tuple[int, int], List[int]] = {}
    for index, puzzle in enumerate(puzzles):
        stratum = (puzzle.optimal_length, puzzle.robots_moved)
        by_stratum.setdefault(stratum, []).append(index)

    chosen: List[Puzzle] = []
    for stratum in sorted(by_stratum):
        seed_material = f"{seed}:{stratum[0]}:{stratum[1]}".encode("utf-8")
        stratum_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = np.random.default_rng(stratum_seed)
        indices = np.asarray(by_stratum[stratum], dtype=np.int64)
        take = min(per_stratum, len(indices))
        picked = rng.choice(indices, size=take, replace=False)
        chosen.extend(puzzles[int(index)] for index in sorted(picked))
    return chosen
