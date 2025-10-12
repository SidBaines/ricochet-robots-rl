from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Provide import-time hints for linters without requiring runtime availability
    import plotly.graph_objects as _go  # type: ignore[import-not-found]
    import plotly.io as _pio  # type: ignore[import-not-found]

import numpy as np

from ..ricochet_core import Board
from .renderer_base import BoardRenderer


class PlotlyBoardRenderer(BoardRenderer):
    """Render a board using Plotly shapes; return RGB array via kaleido export when needed.

    This supports headless PNG export for notebooks and scripts.
    """

    def __init__(self, cell_size: int = 32, show_grid: bool = True) -> None:
        self.cell_size = int(cell_size)
        self.show_grid = bool(show_grid)

    def _build_figure(self, board: Board):
        import importlib
        go = importlib.import_module("plotly.graph_objects")  # type: ignore[assignment]
        H, W = board.height, board.width
        cell = self.cell_size
        width = W * cell
        height = H * cell

        fig = go.Figure()
        # Background
        fig.add_shape(type="rect", x0=0, y0=0, x1=width, y1=height, line=dict(width=0), fillcolor="white")

        # Grid
        if self.show_grid:
            for r in range(H + 1):
                y = r * cell
                fig.add_shape(type="line", x0=0, y0=y, x1=width, y1=y, line=dict(color="#CCCCCC", width=1))
            for c in range(W + 1):
                x = c * cell
                fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=height, line=dict(color="#CCCCCC", width=1))

        # Walls
        for r in range(H + 1):
            for c in range(W):
                if board.h_walls[r, c]:
                    y = r * cell
                    x0 = c * cell
                    x1 = (c + 1) * cell
                    fig.add_shape(type="line", x0=x0, y0=y, x1=x1, y1=y, line=dict(color="#000000", width=3))
        for r in range(H):
            for c in range(W + 1):
                if board.v_walls[r, c]:
                    x = c * cell
                    y0 = r * cell
                    y1 = (r + 1) * cell
                    fig.add_shape(type="line", x0=x, y0=y0, x1=x, y1=y1, line=dict(color="#000000", width=3))

        # Robots
        robot_colors = ["#DC3232", "#3278DC", "#32B450", "#E6C828", "#B450B4"]
        for rid, (rr, cc) in board.robot_positions.items():
            cx = cc * cell + cell * 0.5
            cy = rr * cell + cell * 0.5
            r_px = cell * 0.35
            fig.add_shape(type="circle", x0=cx - r_px, y0=cy - r_px, x1=cx + r_px, y1=cy + r_px,
                          line=dict(color=robot_colors[rid % len(robot_colors)], width=1),
                          fillcolor=robot_colors[rid % len(robot_colors)])

        # Target star as a plus for simplicity
        tr = board.target_robot
        tr_color = robot_colors[tr % len(robot_colors)]
        gr, gc = board.goal_position
        gx = gc * cell + cell * 0.5
        gy = gr * cell + cell * 0.5
        arm = cell * 0.35
        fig.add_shape(type="line", x0=gx - arm, y0=gy, x1=gx + arm, y1=gy, line=dict(color=tr_color, width=2))
        fig.add_shape(type="line", x0=gx, y0=gy - arm, x1=gx, y1=gy + arm, line=dict(color=tr_color, width=2))

        # Axes
        fig.update_xaxes(visible=False, range=[0, width])
        fig.update_yaxes(visible=False, range=[height, 0])  # invert y
        fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    def draw_rgb(self, board: Board) -> np.ndarray:
        fig = self._build_figure(board)
        # Export to image via kaleido
        import importlib
        pio = importlib.import_module("plotly.io")  # type: ignore[assignment]
        try:
            bytes_png = pio.to_image(fig, format="png")  # requires kaleido
        except Exception as e:
            raise RuntimeError("Plotly export requires 'kaleido' installed: pip install -U kaleido") from e
        import PIL.Image as Image
        import io
        img = Image.open(io.BytesIO(bytes_png)).convert("RGB")
        return np.array(img)


