from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from env.ricochet_core import Board


class BoardRenderer(ABC):
    """Abstract renderer that can draw a single `Board` or a sequence.

    Implementations should support at least one of:
    - returning an RGB numpy array for a single frame
    - displaying inline (e.g., in notebooks)
    - exporting animations from a sequence of boards
    """

    @abstractmethod
    def draw_rgb(self, board: Board) -> np.ndarray:
        """Render the board to an (H, W, 3) uint8 RGB array."""
        raise NotImplementedError

    def animate_rgb(self, boards: List[Board], fps: int = 4) -> Optional[np.ndarray]:
        """Optionally return a stacked array of frames (T, H, W, 3) for simple encoders.

        Implementations may override for efficiency. Default composes via draw_rgb.
        """
        _ = fps  # unused in default implementation
        frames: List[np.ndarray] = []
        for b in boards:
            frames.append(self.draw_rgb(b))
        if not frames:
            return None
        return np.stack(frames, axis=0)


