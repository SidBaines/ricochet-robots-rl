"""Reverse-start frontier curriculum.

Bands are keyed by start distance d (how many optimal slides remain to the
goal). The frontier is the easiest band the agent has not yet mastered; the
sampler concentrates episodes there while retaining earlier bands and mixing
in the full task. There are no mastery deadlocks (the frontier always has
learnable success rates: measured masked-random success is 0.51 at d=1) and no
forced advancement that can certify failure as progress.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .puzzles import Puzzle


class UniformSampler:
    """Full starts, uniform over puzzles (no curriculum)."""

    def __init__(self, puzzles: Sequence[Puzzle]) -> None:
        self.puzzles = list(puzzles)

    def sample(self, rng: np.random.Generator) -> Tuple[Puzzle, int]:
        puzzle = self.puzzles[int(rng.integers(len(self.puzzles)))]
        return puzzle, puzzle.optimal_length

    def record(self, start_distance: int, success: bool, puzzle: Puzzle) -> None:
        pass

    def stats(self) -> Dict[str, object]:
        return {"sampler": "uniform"}

    def state_dict(self) -> Dict[str, object]:
        return {"sampler": "uniform"}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state != {"sampler": "uniform"}:
            raise ValueError(f"Incompatible uniform sampler state: {state}")


class ReverseFrontierSampler:
    """Adaptive mixture over start-distance bands.

    Mixture: ``frontier_p`` on the frontier band, ``retention_p`` uniform over
    mastered bands (d < frontier), ``uniform_p`` uniform over all bands, and
    the remainder on original full starts (d = optimal length).
    """

    def __init__(
        self,
        puzzles: Sequence[Puzzle],
        *,
        tau: float = 0.75,
        ema_beta: float = 0.02,
        min_episodes: int = 200,
        frontier_p: float = 0.5,
        retention_p: float = 0.25,
        uniform_p: float = 0.125,
    ) -> None:
        if not puzzles:
            raise ValueError("Need puzzles")
        if frontier_p + retention_p + uniform_p > 1.0:
            raise ValueError("Mixture probabilities exceed 1")
        self.puzzles = list(puzzles)
        self.tau = float(tau)
        self.ema_beta = float(ema_beta)
        self.min_episodes = int(min_episodes)
        self.frontier_p = float(frontier_p)
        self.retention_p = float(retention_p)
        self.uniform_p = float(uniform_p)
        self.max_distance = max(p.optimal_length for p in self.puzzles)
        # Pool of puzzle indices usable for each start distance (L >= d).
        self._pool_by_distance: List[np.ndarray] = [np.empty(0, dtype=np.int64)]
        for d in range(1, self.max_distance + 1):
            self._pool_by_distance.append(
                np.array(
                    [i for i, p in enumerate(self.puzzles) if p.optimal_length >= d],
                    dtype=np.int64,
                )
            )
        self.ema = np.zeros(self.max_distance + 1, dtype=np.float64)
        self.count = np.zeros(self.max_distance + 1, dtype=np.int64)

    def frontier(self) -> int:
        for d in range(1, self.max_distance + 1):
            if self.count[d] < self.min_episodes or self.ema[d] < self.tau:
                return d
        return self.max_distance

    def sample(self, rng: np.random.Generator) -> Tuple[Puzzle, int]:
        f = self.frontier()
        u = float(rng.random())
        if u < self.frontier_p:
            d = f
        elif u < self.frontier_p + self.retention_p and f > 1:
            d = 1 + int(rng.integers(f - 1))
        elif u < self.frontier_p + self.retention_p + self.uniform_p:
            d = 1 + int(rng.integers(self.max_distance))
        else:
            puzzle = self.puzzles[int(rng.integers(len(self.puzzles)))]
            return puzzle, puzzle.optimal_length
        pool = self._pool_by_distance[d]
        puzzle = self.puzzles[int(pool[int(rng.integers(len(pool)))])]
        return puzzle, d

    def record(self, start_distance: int, success: bool, puzzle: Puzzle) -> None:
        d = min(int(start_distance), self.max_distance)
        if self.count[d] == 0:
            self.ema[d] = float(success)
        else:
            self.ema[d] += self.ema_beta * (float(success) - self.ema[d])
        self.count[d] += 1

    def stats(self) -> Dict[str, object]:
        return {
            "sampler": "reverse_frontier",
            "frontier": self.frontier(),
            "ema": {d: round(float(self.ema[d]), 3) for d in range(1, self.max_distance + 1)},
            "count": {d: int(self.count[d]) for d in range(1, self.max_distance + 1)},
        }

    def state_dict(self) -> Dict[str, object]:
        """Exact adaptive-curriculum state for crash-safe training resume."""
        return {
            "sampler": "reverse_frontier",
            "max_distance": self.max_distance,
            "ema": self.ema.copy(),
            "count": self.count.copy(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("sampler") != "reverse_frontier":
            raise ValueError(f"Incompatible sampler state: {state.get('sampler')}")
        if int(state.get("max_distance", -1)) != self.max_distance:
            raise ValueError(
                f"Sampler max_distance changed: checkpoint={state.get('max_distance')} "
                f"current={self.max_distance}"
            )
        ema = np.asarray(state["ema"], dtype=np.float64)
        count = np.asarray(state["count"], dtype=np.int64)
        if ema.shape != self.ema.shape or count.shape != self.count.shape:
            raise ValueError(
                f"Sampler state shape changed: checkpoint={ema.shape}/{count.shape} "
                f"current={self.ema.shape}/{self.count.shape}"
            )
        self.ema[...] = ema
        self.count[...] = count
