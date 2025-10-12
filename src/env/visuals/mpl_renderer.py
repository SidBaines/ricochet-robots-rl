from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..ricochet_core import Board
from .renderer_base import BoardRenderer


class MatplotlibBoardRenderer(BoardRenderer):
    """Simple raster renderer using the env's internal RGB drawer for consistency.

    Relies on `RicochetRobotsEnv._render_rgb` behavior for exact visuals by porting
    the drawing logic here minimally. For now, we call a light copy of that logic.
    """

    def __init__(self, cell_size: int = 24) -> None:
        self.cell_size = int(cell_size)

    def draw_rgb(self, board: Board) -> np.ndarray:
        # Minimal duplication of env._render_rgb for out-of-env usage
        H, W = board.height, board.width
        cell = self.cell_size
        H_px, W_px = H * cell, W * cell
        img = np.ones((H_px, W_px, 3), dtype=np.uint8) * 255

        grid_color = (200, 200, 200)
        grid_th = 1
        wall_color = (0, 0, 0)
        wall_th = 3
        robot_colors: List[Tuple[int, int, int]] = [
            (220, 50, 50), (50, 120, 220), (50, 180, 80), (230, 200, 40), (180, 80, 180)
        ]
        circle_fill = True
        circle_r_frac = 0.35
        target_dark_factor = 0.6
        star_th = 2

        def draw_hline(y: int, x0: int, x1: int, color: Tuple[int, int, int], thickness: int) -> None:
            y0 = max(0, y - thickness // 2)
            y1 = min(H_px, y + (thickness - thickness // 2))
            x0c = max(0, min(x0, x1))
            x1c = min(W_px, max(x0, x1))
            img[y0:y1, x0c:x1c] = color

        def draw_vline(x: int, y0: int, y1: int, color: Tuple[int, int, int], thickness: int) -> None:
            x0 = max(0, x - thickness // 2)
            x1 = min(W_px, x + (thickness - thickness // 2))
            y0c = max(0, min(y0, y1))
            y1c = min(H_px, max(y0, y1))
            img[y0c:y1c, x0:x1] = color

        def draw_circle(cx: float, cy: float, radius: float, color: Tuple[int, int, int], fill: bool) -> None:
            x0 = int(max(0, np.floor(cx - radius)))
            x1 = int(min(W_px - 1, np.ceil(cx + radius)))
            y0 = int(max(0, np.floor(cy - radius)))
            y1 = int(min(H_px - 1, np.ceil(cy + radius)))
            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            if fill:
                img[y0:y1 + 1, x0:x1 + 1][mask] = color
            else:
                inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius - 1) ** 2
                ring = np.logical_and(mask, np.logical_not(inner))
                img[y0:y1 + 1, x0:x1 + 1][ring] = color

        def draw_star(cx: float, cy: float, arm_len: float, color: Tuple[int, int, int], thickness: int) -> None:
            draw_hline(int(cy), int(cx - arm_len), int(cx + arm_len), color, thickness)
            draw_vline(int(cx), int(cy - arm_len), int(cy + arm_len), color, thickness)
            steps = int(arm_len)
            for s in range(-steps, steps + 1):
                x = int(cx + s)
                y1 = int(cy + s)
                y2 = int(cy - s)
                y0a = max(0, y1 - thickness // 2)
                y1a = min(H_px, y1 + (thickness - thickness // 2))
                x0a = max(0, x - thickness // 2)
                x1a = min(W_px, x + (thickness - thickness // 2))
                img[y0a:y1a, x0a:x1a] = color
                y0b = max(0, y2 - thickness // 2)
                y1b = min(H_px, y2 + (thickness - thickness // 2))
                img[y0b:y1b, x0a:x1a] = color

        # Grid
        for r in range(H + 1):
            draw_hline(int(r * cell), 0, W_px, grid_color, grid_th)
        for c in range(W + 1):
            draw_vline(int(c * cell), 0, H_px, grid_color, grid_th)

        # Walls
        for r in range(H + 1):
            for c in range(W):
                if board.h_walls[r, c]:
                    y = int(r * cell)
                    x0 = int(c * cell)
                    x1 = int((c + 1) * cell)
                    draw_hline(y, x0, x1, wall_color, wall_th)
        for r in range(H):
            for c in range(W + 1):
                if board.v_walls[r, c]:
                    x = int(c * cell)
                    y0 = int(r * cell)
                    y1 = int((r + 1) * cell)
                    draw_vline(x, y0, y1, wall_color, wall_th)

        # Robots
        radius = cell * float(circle_r_frac)
        for rid, (rr, cc) in board.robot_positions.items():
            color = robot_colors[rid % len(robot_colors)]
            cy = rr * cell + cell * 0.5
            cx = cc * cell + cell * 0.5
            draw_circle(cx, cy, radius, color, circle_fill)

        # Target
        tr = board.target_robot
        tr_color = robot_colors[tr % len(robot_colors)]
        dark_color = tuple(int(max(0, min(255, target_dark_factor * v))) for v in tr_color)
        gr, gc = board.goal_position
        cy = gr * cell + cell * 0.5
        cx = gc * cell + cell * 0.5
        draw_star(cx, cy, arm_len=cell * 0.35, color=dark_color, thickness=star_th)

        return img


