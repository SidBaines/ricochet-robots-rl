import pytest
import numpy as np
from environment.board import Board
from environment.utils import NORTH, EAST, SOUTH, WEST, DIRECTIONS

class TestBoard:
    def test_board_creation(self):
        board = Board(height=8, width=10)
        assert board.height == 8
        assert board.width == 10
        assert board.walls.shape == (8, 10, 4)
        assert np.all(board.walls[0, :, NORTH]) # Top perimeter
        assert np.all(board.walls[7, :, SOUTH]) # Bottom perimeter
        assert np.all(board.walls[:, 0, WEST])  # Left perimeter
        assert np.all(board.walls[:, 9, EAST])  # Right perimeter

    def test_add_wall_and_has_wall(self):
        board = Board(5, 5)
        # Add a wall to the EAST of cell (1,1)
        board.add_wall(1, 1, EAST)
        assert board.has_wall(1, 1, EAST)
        # Check corresponding wall on neighbor (1,2) WEST
        assert board.has_wall(1, 2, WEST)

        # Add a wall to the SOUTH of cell (2,2)
        board.add_wall(2, 2, SOUTH)
        assert board.has_wall(2, 2, SOUTH)
        # Check corresponding wall on neighbor (3,2) NORTH
        assert board.has_wall(3, 2, NORTH)

    def test_add_wall_out_of_bounds(self):
        board = Board(5, 5)
        with pytest.raises(ValueError):
            board.add_wall(5, 5, NORTH) # r is out of bounds

    def test_has_wall_out_of_bounds(self):
        board = Board(5, 5)
        # Querying a wall for a cell outside bounds should effectively be true
        # as if there's an implicit boundary wall.
        assert board.has_wall(5, 2, NORTH) # r out of bounds
        assert board.has_wall(2, 5, EAST)  # c out of bounds
        assert board.has_wall(-1, 2, SOUTH) # r out of bounds
        assert board.has_wall(2, -1, WEST)  # c out of bounds

    def test_standard_walls_16x16(self):
        board = Board(16, 16)
        board.add_standard_ricochet_walls()
        # Check a few expected walls from the standard setup
        # Example: wall EAST of (2,2)
        assert board.has_wall(2, 2, EAST)
        assert board.has_wall(2, 3, WEST) # Its pair

        # Example: wall SOUTH of (2,2)
        assert board.has_wall(2, 2, SOUTH)
        assert board.has_wall(3, 2, NORTH) # Its pair

        # Center area walls (example around (7,7) which is cr-1, cc-1 for cr,cc=8,8)
        # (6,6) SOUTH wall -> (7,6) NORTH wall
        assert board.has_wall(6,6,SOUTH)
        assert board.has_wall(7,6,NORTH) 