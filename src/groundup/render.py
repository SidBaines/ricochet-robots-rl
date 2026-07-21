"""Minimal RGB board renderer for pixel observations.

Geometry: the canvas is (H*k+1, W*k+1) pixels for an H x W board at ``k``
pixels per cell, so every boundary line (grid or wall) owns its own 1-px
row/column and every cell interior is exactly (k-1) x (k-1) pixels:

  line r at y = r*k for r in 0..H,  cell r interior rows [r*k+1, (r+1)*k)

Wall segments span both of their corner pixels ([a*k, b*k+1) inclusive), so
meeting walls form closed corners. The goal tint and robot sprites cover the
full cell interior and never touch boundary pixels. (An earlier H*k canvas
clamped the last boundary line into the final cell row/col, which shrank
edge cells, left wall corners open, and offset the sprites — fixed
2026-07-20 after visual review.)

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


def canvas_shape(height: int, width: int, cell_px: int) -> Tuple[int, int]:
    return (height * cell_px + 1, width * cell_px + 1)


def _interior(index: int, cell_px: int) -> slice:
    """Pixel slice of a cell's interior along one axis (excludes both lines)."""
    return slice(index * cell_px + 1, (index + 1) * cell_px)


def static_background(puzzle: Puzzle, cell_px: int) -> np.ndarray:
    """(3, H*k+1, W*k+1) float32: white canvas, grid, closed walls, goal tint."""
    key = (puzzle.seed, cell_px)
    cached = _BG_CACHE.get(key)
    if cached is not None:
        return cached
    height, width, k = puzzle.height, puzzle.width, cell_px
    h_px, w_px = canvas_shape(height, width, k)
    img = np.full((3, h_px, w_px), BACKGROUND, dtype=np.float32)

    # Faint grid lines: every boundary owns its own pixel row/column.
    for r in range(height + 1):
        img[:, r * k, :] = GRID_GRAY
    for c in range(width + 1):
        img[:, :, c * k] = GRID_GRAY

    # Goal tint on the cell interior only (never the boundary pixels).
    gr, gc = puzzle.goal
    tint = np.array(ROBOT_COLORS[0], dtype=np.float32).reshape(3, 1, 1)
    region = img[:, _interior(gr, k), _interior(gc, k)]
    img[:, _interior(gr, k), _interior(gc, k)] = (
        (1 - GOAL_TINT_ALPHA) * region + GOAL_TINT_ALPHA * tint
    )

    # Walls: black segments spanning BOTH corner pixels, so junctions close.
    for r in range(height + 1):
        for c in range(width):
            if puzzle.h_walls[r, c]:
                img[:, r * k, c * k : (c + 1) * k + 1] = WALL_BLACK
    for r in range(height):
        for c in range(width + 1):
            if puzzle.v_walls[r, c]:
                img[:, r * k : (r + 1) * k + 1, c * k] = WALL_BLACK

    img.setflags(write=False)
    _BG_CACHE[key] = img
    return img


def render(puzzle: Puzzle, robots: Sequence[Tuple[int, int]], cell_px: int) -> np.ndarray:
    """Render the current state: cached background + robot sprites.

    ``robots`` are in canonical order (index 0 = target robot, drawn red).
    Sprites fill the entire (k-1) x (k-1) cell interior.
    """
    k = cell_px
    img = static_background(puzzle, k).copy()
    for slot, (r, c) in enumerate(robots):
        color = np.array(ROBOT_COLORS[slot % len(ROBOT_COLORS)], dtype=np.float32)
        img[:, _interior(r, k), _interior(c, k)] = color.reshape(3, 1, 1)
    return img
