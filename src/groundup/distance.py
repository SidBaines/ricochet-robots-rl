"""Relaxed slide-distance maps used as a potential for reward shaping.

``d[r, c]`` = minimum number of slides for the target robot to travel from
(r, c) to the goal under the optimistic relaxation that it may stop at *any*
cell along a slide ray (walls only, other robots ignored). This is the classic
admissible Ricochet Robots heuristic; verified empirically over the phase4
bank: 0/300 admissibility violations, mean d0/optimal = 0.52.

Cells that cannot reach the goal even under the relaxation (~1.6% of cells on
sparse 16x16 boards) are clamped to (max finite distance + 2) so the potential
stays bounded.
"""

from __future__ import annotations

import numpy as np


def relaxed_distance_map(
    h_walls: np.ndarray, v_walls: np.ndarray, goal: tuple[int, int]
) -> np.ndarray:
    """Reverse BFS from the goal over the 'stop anywhere on a ray' relaxation.

    A cell c gets distance k+1 when some straight, wall-free corridor from c
    contains a cell at distance k (the robot slides through it and could be
    stopped there by a well-placed blocker).
    """
    height, width = h_walls.shape[0] - 1, h_walls.shape[1]
    INF = np.iinfo(np.int32).max
    dist = np.full((height, width), INF, dtype=np.int32)
    gr, gc = goal
    dist[gr, gc] = 0
    frontier = [(gr, gc)]
    level = 0
    while frontier:
        level += 1
        nxt: list[tuple[int, int]] = []
        for fr, fc in frontier:
            # Walk outward from the frontier cell in each direction; every cell
            # in the corridor can reach the frontier cell in one slide.
            r = fr
            while r + 1 < height and not h_walls[r + 1, fc]:
                r += 1
                if dist[r, fc] > level:
                    dist[r, fc] = level
                    nxt.append((r, fc))
            r = fr
            while r > 0 and not h_walls[r, fc]:
                r -= 1
                if dist[r, fc] > level:
                    dist[r, fc] = level
                    nxt.append((r, fc))
            c = fc
            while c + 1 < width and not v_walls[fr, c + 1]:
                c += 1
                if dist[fr, c] > level:
                    dist[fr, c] = level
                    nxt.append((fr, c))
            c = fc
            while c > 0 and not v_walls[fr, c]:
                c -= 1
                if dist[fr, c] > level:
                    dist[fr, c] = level
                    nxt.append((fr, c))
        frontier = nxt
    finite = dist[dist < INF]
    clamp = int(finite.max()) + 2 if finite.size else 2
    dist[dist == INF] = clamp
    return dist
