# environment/custom_difficulty_envs.py
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from .ricochet_env import RicochetRobotsEnv
from .board import Board
from .utils import DIRECTIONS, NORTH, EAST, SOUTH, WEST

class RicochetRobotsEnvCustomDifficulty(RicochetRobotsEnv):
    """Base class for custom difficulty Ricochet Robots environments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def reset_with_custom_config(self, 
                                seed: Optional[int] = None, 
                                options: Optional[dict] = None,
                                robot_positions: Optional[List[Tuple[int, int]]] = None,
                                target_robot_idx: Optional[int] = None,
                                target_pos: Optional[Tuple[int, int]] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment with custom robot and target positions.
        
        Args:
            seed: Random seed
            options: Additional options
            robot_positions: List of (row, col) positions for robots
            target_robot_idx: Index of the target robot
            target_pos: (row, col) position of the target
            
        Returns:
            Observation and info dictionaries
        """
        # Call parent reset to initialize basic state
        super().reset(seed=seed)
        self.current_step = 0
        
        # Set custom robot positions if provided
        if robot_positions is not None:
            assert len(robot_positions) == self.num_robots, "Number of positions must match number of robots"
            for i, pos in enumerate(robot_positions):
                self.robots[i].set_position(pos[0], pos[1])
        
        # Set custom target robot if provided
        if target_robot_idx is not None:
            assert 0 <= target_robot_idx < self.num_robots, "Target robot index out of range"
            self.target_robot_idx = target_robot_idx
            
        # Set custom target position if provided
        if target_pos is not None:
            assert 0 <= target_pos[0] < self.height and 0 <= target_pos[1] < self.width, "Target position out of bounds"
            self.target_pos = target_pos
            
        return self._get_obs(), self._get_info()
