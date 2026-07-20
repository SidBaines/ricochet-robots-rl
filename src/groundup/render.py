"""Minimal RGB board renderer for pixel observations.

Cell-aligned sprite rendering at ``cell_px`` pixels per board cell, float32 in
[0, 1], channels-first. The static background (grid, walls, goal tint) is
cached per layout; robots are stamped per step.

Colour convention (canonical identity on): robot 0 is always the target and
always red; the goal cell is tinted with the target colour. When the
helper-removal ladder reaches the colour-generalisation step, the colour
assignment (and the tint linking goal to target) becomes the thing the network
must learn — see docs/GroundUpRLRedesign.md §4.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .puzzles import Puzzle

# RGB in [0,1]; index = canonical robot slot (0 = target).
ROBOT_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.90, 0.10, 0.10),  # target: red
    (0.15, 0.35, 0.95),  # blue
    (0.10, 0.75, 0.25),  # green
    (0.95, 0.80, 0.10),  # yellow
    (0.70, 0.20, 0.80),  # purple (spare)
)
BACKGROUND = 1.0
GRID_GRAY = 0.85
WALL_BLACK = 0.0
GOAL_TINT_ALPHA = 0.35

_BG_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def static_background(puzzle: Puzzle, cell_px: int) -> np.ndarray:
    """(3, H*k, W*k) float32: white canvas, faint grid, black walls, goal tint."""
    key = (puzzle.seed, cell_px)
    cached = _BG_CACHE.get(key)
    if cached is not None:
        return cached
    height, width, k = puzzle.height, puzzle.width, cell_px
    img = np.full((3, height * k, width * k), BACKGROUND, dtype=np.float32)

    # Faint grid lines on every cell boundary (1 px).
    for r in range(height + 1):
        y = min(r * k, height * k - 1)
        img[:, y, :] = GRID_GRAY
    for c in range(width + 1):
        x = min(c * k, width * k - 1)
        img[:, :, x] = GRID_GRAY

    # Goal tint (under the walls): blend the target colour into the goal cell.
    gr, gc = puzzle.goal
    tint = np.array(ROBOT_COLORS[0], dtype=np.float32).reshape(3, 1, 1)
    region = img[:, gr * k : (gr + 1) * k, gc * k : (gc + 1) * k]
    img[:, gr * k : (gr + 1) * k, gc * k : (gc + 1) * k] = (
        (1 - GOAL_TINT_ALPHA) * region + GOAL_TINT_ALPHA * tint
    )

    # Walls: black 1-px segments on cell borders (clamped to canvas).
    for r in range(height + 1):
        y = min(r * k, height * k - 1)
        for c in range(width):
            if puzzle.h_walls[r, c]:
                img[:, y, c * k : (c + 1) * k] = WALL_BLACK
    for r in range(height):
        for c in range(width + 1):
            x = min(c * k, width * k - 1)
            if puzzle.v_walls[r, c]:
                img[:, r * k : (r + 1) * k, x] = WALL_BLACK

    img.setflags(write=False)
    _BG_CACHE[key] = img
    return img


def render(puzzle: Puzzle, robots: Sequence[Tuple[int, int]], cell_px: int) -> np.ndarray:
    """Render the current state: cached background + robot sprites.

    ``robots`` are in canonical order (index 0 = target robot, drawn red).
    Robots are filled squares inset by 1 px so walls/grid stay visible.
    """
    k = cell_px
    img = static_background(puzzle, k).copy()
    inset = 1 if k >= 3 else 0
    for slot, (r, c) in enumerate(robots):
        color = np.array(ROBOT_COLORS[slot % len(ROBOT_COLORS)], dtype=np.float32)
        y0, x0 = r * k + inset, c * k + inset
        y1, x1 = (r + 1) * k - inset, (c + 1) * k - inset
        img[:, y0:y1, x0:x1] = color.reshape(3, 1, 1)
    return img
