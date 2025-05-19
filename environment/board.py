import numpy as np
from .utils import NORTH, EAST, SOUTH, WEST, DIRECTIONS

class Board:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        # self.walls[r, c, direction_idx] is True if cell (r,c) has a wall on that side
        self.walls = np.zeros((height, width, 4), dtype=bool)
        self._setup_perimeter_walls()

    def _setup_perimeter_walls(self):
        # North walls for top row
        self.walls[0, :, NORTH] = True
        # South walls for bottom row
        self.walls[self.height - 1, :, SOUTH] = True
        # West walls for first column
        self.walls[:, 0, WEST] = True
        # East walls for last column
        self.walls[:, self.width - 1, EAST] = True

    def add_wall(self, r: int, c: int, direction_idx: int):
        """Adds a wall to cell (r,c) on the specified direction_idx side.
        Also adds the corresponding wall to the neighboring cell.
        """
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"Cell ({r},{c}) is out of bounds.")

        self.walls[r, c, direction_idx] = True

        # Add corresponding wall to neighbor
        dr, dc = DIRECTIONS[direction_idx]
        neighbor_r, neighbor_c = r + dr, c + dc

        if 0 <= neighbor_r < self.height and 0 <= neighbor_c < self.width:
            opposite_direction_idx = (direction_idx + 2) % 4 # N<->S, E<->W
            self.walls[neighbor_r, neighbor_c, opposite_direction_idx] = True

    def has_wall(self, r: int, c: int, direction_idx: int) -> bool:
        """Checks if cell (r,c) has a wall on the specified direction_idx side."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            # Outside bounds is effectively a wall in that direction
            return True
        return self.walls[r, c, direction_idx]

    def __repr__(self):
        return f"Board(height={self.height}, width={self.width})"

    def get_center_square_coords(self) -> list[tuple[int,int]]:
        """Returns coordinates of the 2x2 center square, common in Ricochet Robots."""
        if self.height < 2 or self.width < 2:
            return []
        center_r, center_c = self.height // 2, self.width // 2
        return [
            (center_r - 1, center_c - 1), (center_r - 1, center_c),
            (center_r, center_c - 1), (center_r, center_c)
        ]

    def add_standard_ricochet_walls(self):
        """Adds some example walls typical of Ricochet Robots boards.
        This is a simplified example. Real boards have specific configurations.
        """
        # Block off the center 2x2 square (usually it's impassable or special)
        # For simplicity, we'll just add walls around it.
        if self.height == 16 and self.width == 16: # Standard size
            cr, cc = self.height // 2, self.width // 2
            # Walls around (cr-2, cc-2) to (cr+1, cc+1) effectively
            # (7,7) (7,8)
            # (8,7) (8,8)
            # Top walls of (7,7) and (7,8)
            self.add_wall(cr - 2, cc - 2, SOUTH) # Wall below (6,6) which is N of (7,6)
            self.add_wall(cr - 2, cc - 1, SOUTH) # Wall below (6,7) which is N of (7,7)

            # Bottom walls of (8,7) and (8,8)
            self.add_wall(cr -1 , cc - 2, NORTH) # Wall above (9,6) which is S of (8,6)
            self.add_wall(cr -1 , cc - 1, NORTH) # Wall above (9,7) which is S of (8,7)

            # Left walls of (7,7) and (8,7)
            self.add_wall(cr - 2, cc - 2, EAST) # Wall right of (7,5) which is W of (7,6)
            self.add_wall(cr - 1, cc - 2, EAST) # Wall right of (8,5) which is W of (8,6)

            # Right walls of (7,8) and (8,8)
            self.add_wall(cr - 2, cc -1 , WEST) # Wall left of (7,9) which is E of (7,8)
            self.add_wall(cr - 1, cc -1 , WEST) # Wall left of (8,9) which is E of (8,8)

            # Some example outer walls (these are often specific per quadrant)
            self.add_wall(2, 2, EAST)
            self.add_wall(2, 2, SOUTH)

            self.add_wall(2, 13, WEST)
            self.add_wall(2, 13, SOUTH)

            self.add_wall(13, 2, EAST)
            self.add_wall(13, 2, NORTH)

            self.add_wall(13, 13, WEST)
            self.add_wall(13, 13, NORTH) 