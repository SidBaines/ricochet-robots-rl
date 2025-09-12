from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

# Directions encoded as (dr, dc)
DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
]

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


@dataclass
class Board:
    height: int
    width: int
    # Edge walls: canonical Ricochet Robots representation
    # h_walls[r, c] is a horizontal wall between (r-1, c) and (r, c), shape (H+1, W)
    # v_walls[r, c] is a vertical wall between (r, c-1) and (r, c), shape (H, W+1)
    h_walls: np.ndarray  # dtype=bool, shape (H+1, W)
    v_walls: np.ndarray  # dtype=bool, shape (H, W+1)
    # robots indexed 0..N-1
    robot_positions: Dict[int, Tuple[int, int]]
    # goal location and which robot is the target
    goal_position: Tuple[int, int]
    target_robot: int

    def __post_init__(self) -> None:
        assert self.h_walls.shape == (self.height + 1, self.width)
        assert self.v_walls.shape == (self.height, self.width + 1)
        for _, (r, c) in self.robot_positions.items():
            assert 0 <= r < self.height and 0 <= c < self.width, "Robot out of bounds"
        gr, gc = self.goal_position
        assert 0 <= gr < self.height and 0 <= gc < self.width, "Goal out of bounds"

    @property
    def num_robots(self) -> int:
        return len(self.robot_positions)

    def is_occupied(self, row: int, col: int, ignore_robot: Optional[int] = None) -> bool:
        for rid, (rr, cc) in self.robot_positions.items():
            if ignore_robot is not None and rid == ignore_robot:
                continue
            if rr == row and cc == col:
                return True
        return False

    def has_wall_up(self, row: int, col: int) -> bool:
        return self.h_walls[row, col]

    def has_wall_down(self, row: int, col: int) -> bool:
        return self.h_walls[row + 1, col]

    def has_wall_left(self, row: int, col: int) -> bool:
        return self.v_walls[row, col]

    def has_wall_right(self, row: int, col: int) -> bool:
        return self.v_walls[row, col + 1]

    def clone(self) -> "Board":
        return Board(
            height=self.height,
            width=self.width,
            h_walls=self.h_walls.copy(),
            v_walls=self.v_walls.copy(),
            robot_positions=dict(self.robot_positions),
            goal_position=tuple(self.goal_position),
            target_robot=self.target_robot,
        )


def slide_until_blocked(board: Board, robot_id: int, direction: int) -> Tuple[int, int]:
    """
    Slide the specified robot in the given direction until hitting a wall edge or another robot.
    Returns the final (row, col) position after sliding.
    """
    assert 0 <= direction < 4
    r, c = board.robot_positions[robot_id]

    # Precompute occupied positions excluding the moving robot
    occupied = np.zeros((board.height, board.width), dtype=bool)
    for rid, (rr, cc) in board.robot_positions.items():
        if rid == robot_id:
            continue
        occupied[rr, cc] = True

    if direction == UP:
        # Move up until wall above or occupied cell above
        while r > 0 and not board.has_wall_up(r, c) and not occupied[r - 1, c]:
            r -= 1
    elif direction == DOWN:
        while r < board.height - 1 and not board.has_wall_down(r, c) and not occupied[r + 1, c]:
            r += 1
    elif direction == LEFT:
        while c > 0 and not board.has_wall_left(r, c) and not occupied[r, c - 1]:
            c -= 1
    elif direction == RIGHT:
        while c < board.width - 1 and not board.has_wall_right(r, c) and not occupied[r, c + 1]:
            c += 1

    return r, c


def apply_action(board: Board, robot_id: int, direction: int) -> Board:
    """
    Return a new Board after applying the action. If slide results in no movement,
    the robot's position remains the same (board state equal to input aside from copy).
    """
    new_board = board.clone()
    end_r, end_c = slide_until_blocked(new_board, robot_id, direction)
    new_board.robot_positions[robot_id] = (end_r, end_c)
    return new_board


def reached_goal(board: Board) -> bool:
    r, c = board.robot_positions[board.target_robot]
    return (r, c) == board.goal_position


def render_ascii(board: Board) -> str:
    # Simple ASCII with edge walls: draw a grid with +-| for walls
    h, w = board.height, board.width
    lines: List[str] = []

    # Top border
    top = "+"
    for c in range(w):
        top += "---" if board.h_walls[0, c] else "   "
        top += "+"
    lines.append(top)

    for r in range(h):
        # Row with vertical walls and cells
        row_str = ""
        for c in range(w):
            left_wall = "|" if board.v_walls[r, c] else " "
            cell_char = "."
            if (r, c) == board.goal_position:
                cell_char = "G"
            for rid, (rr, cc) in board.robot_positions.items():
                if rr == r and cc == c:
                    cell_char = chr(ord('A') + rid)
            row_str += f"{left_wall} {cell_char} "
        # Rightmost wall
        row_str += "|" if board.v_walls[r, w] else " "
        lines.append(row_str)
        # Horizontal walls below cells
        hline = "+"
        for c in range(w):
            hline += "---" if board.h_walls[r + 1, c] else "   "
            hline += "+"
        lines.append(hline)

    return "\n".join(lines)
