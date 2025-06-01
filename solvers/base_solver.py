from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class BaseSolver(ABC):
    """Base class for all Ricochet Robots solvers."""
    
    def __init__(self, board_size: Tuple[int, int], num_robots: int):
        """
        Initialize the solver.
        
        Args:
            board_size: Tuple of (height, width) for the board
            num_robots: Number of robots in the puzzle
        """
        self.height, self.width = board_size
        self.num_robots = num_robots
        
    @abstractmethod
    def solve(self, 
              robot_positions: List[Tuple[int, int]], 
              target_robot_idx: int,
              target_pos: Tuple[int, int],
              walls: np.ndarray) -> Optional[List[int]]:
        """
        Solve the Ricochet Robots puzzle.
        
        Args:
            robot_positions: List of (row, col) positions for each robot
            target_robot_idx: Index of the robot that needs to reach the target
            target_pos: (row, col) position of the target
            walls: Boolean array of shape (height, width, 4) indicating walls
                  walls[r,c,d] is True if there's a wall in direction d at (r,c)
                  Directions are: 0=North, 1=East, 2=South, 3=West
                  
        Returns:
            List of actions that solve the puzzle, or None if no solution exists.
            Each action is an integer: robot_idx * 4 + direction_idx
            where direction_idx is 0=North, 1=East, 2=South, 3=West
        """
        pass
    
    @abstractmethod
    def get_solution_length(self) -> Optional[int]:
        """
        Get the length of the solution found by the solver.
        
        Returns:
            Number of moves in the solution, or None if no solution was found
        """
        pass 