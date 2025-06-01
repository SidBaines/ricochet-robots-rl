"""
Curriculum environment for self-paced Ricochet Robots RL training.

Environment subclass that receives boards from the curriculum board pool
instead of generating them randomly, enabling progressive difficulty scaling.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import random
import time

from environment.utils import ROBOT_COLORS
from environment.ricochet_env import RicochetRobotsEnv
from environment.robot import Robot
from .board_pool import BoardPoolManager


class CurriculumRicochetEnv(RicochetRobotsEnv):
    """
    Ricochet Robots environment that uses curriculum learning with board pool.
    
    Instead of generating random boards, this environment gets pre-solved boards
    from a board pool that maintains boards with optimal solution length ≤ k,
    where k is dynamically adjusted based on agent performance.
    """
    
    def __init__(self,
                 board_pool_manager: BoardPoolManager,
                 max_steps: int = 20,
                 render_mode: Optional[str] = None,
                 epsilon_random: float = 0.05,
                 fallback_env_class=None,
                 **kwargs):
        """
        Initialize curriculum environment.
        
        Args:
            board_pool_manager: Manager that provides curriculum boards
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ("human", "rgb_array", or None)
            epsilon_random: Probability of sampling any board instead of curriculum board
            fallback_env_class: Environment class to use when board pool is empty
            **kwargs: Additional arguments passed to base environment
        """
        # Initialize with dummy parameters (will be overridden by curriculum boards)
        super().__init__(
            board_size=5,  # Will be overridden
            num_robots=3,  # Will be overridden  
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs
        )
        
        self.board_pool_manager = board_pool_manager
        self.epsilon_random = epsilon_random
        self.fallback_env_class = fallback_env_class
        
        # Statistics
        self.curriculum_boards_used = 0
        self.fallback_boards_used = 0
        self.pool_empty_count = 0
        self.current_curriculum_k = None
        
        # Store the last board data for debugging
        self.last_board_data = None
    
    def _load_board_from_data(self, board_data: Dict[str, Any]):
        """
        Load board configuration from curriculum data.
        
        Args:
            board_data: Dictionary containing board, robot positions, target info
        """
        # Extract board data
        board = board_data['board']
        robot_positions = board_data['robot_positions']
        target_robot_idx = board_data['target_robot_idx']
        target_pos = board_data['target_pos']
        
        # Update environment dimensions
        self.height = board.height
        self.width = board.width
        self.board_size = board.height  # Assuming square boards
        
        # Update board
        self.board = board
        
        # Update robots
        self.num_robots = len(robot_positions)
        self.robots = []
        for i, (r, c) in enumerate(robot_positions):
            robot = Robot(robot_id=i, color=ROBOT_COLORS[i], initial_pos=(r, c))
            self.robots.append(robot)
        
        # Update target
        self.target_robot_idx = target_robot_idx
        self.target_pos = target_pos
        
        # Update observation and action spaces if dimensions changed
        self._update_spaces()
        
        # Store for debugging
        self.last_board_data = board_data
        
        # Track current curriculum level
        if 'optimal_length' in board_data:
            self.current_curriculum_k = board_data['optimal_length']
    
    def _update_spaces(self):
        """Update observation and action spaces when board dimensions change."""
        import gymnasium as gym
        
        # Update observation space
        self.observation_space = gym.spaces.Dict({
            'board': gym.spaces.Box(
                low=0, high=1,
                shape=(self.height, self.width, 4),  # 4 directions for walls
                dtype=np.float32
            ),
            'robots': gym.spaces.Box(
                low=0, high=max(self.height, self.width) - 1,
                shape=(self.num_robots, 2),
                dtype=np.int32
            ),
            'target_robot_idx': gym.spaces.Discrete(self.num_robots),
            'target_pos': gym.spaces.Box(
                low=0, high=max(self.height, self.width) - 1,
                shape=(2,),
                dtype=np.int32
            )
        })
        
        # Update action space (robot_idx * 4 + direction)
        self.action_space = gym.spaces.Discrete(self.num_robots * 4)
    
    def _get_fallback_board(self) -> bool:
        """
        Generate a fallback board when curriculum pool is empty.
        
        Returns:
            True if fallback board created successfully, False otherwise
        """
        if self.fallback_env_class is None:
            return False
        
        try:
            # Create temporary fallback environment
            fallback_env = self.fallback_env_class(
                board_size=self.board_size,
                num_robots=self.num_robots,
                max_steps=self.max_steps
            )
            
            # Get a board from fallback
            fallback_env.reset(seed=int(time.time() * 1000) % (2**31))
            
            # Copy configuration to this environment
            self.board = fallback_env.board
            self.robots = fallback_env.robots
            self.target_robot_idx = fallback_env.target_robot_idx
            self.target_pos = fallback_env.target_pos
            self.height = fallback_env.height
            self.width = fallback_env.width
            
            self.fallback_boards_used += 1
            return True
            
        except Exception as e:
            print(f"Fallback board generation failed: {e}")
            return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment with a new curriculum board.
        
        Args:
            seed: Random seed (used for epsilon-random sampling)
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self._set_seed(seed)
        
        # Decide whether to use curriculum board or sample randomly
        use_curriculum = (self.np_random.random() > self.epsilon_random)
        
        if use_curriculum:
            # Try to get board from curriculum pool
            board_data = self.board_pool_manager.get_board(timeout=0.1)
            
            if board_data is not None:
                # Load curriculum board
                self._load_board_from_data(board_data)
                self.curriculum_boards_used += 1
            else:
                # Pool is empty, use fallback
                self.pool_empty_count += 1
                if not self._get_fallback_board():
                    # Fallback failed, use default reset
                    super().reset(seed=seed, options=options)
        else:
            # Epsilon-random: use fallback for exploration
            if not self._get_fallback_board():
                super().reset(seed=seed, options=options)
        
        # Reset episode state
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """
        Get curriculum-specific statistics.
        
        Returns:
            Dictionary with curriculum metrics
        """
        total_boards = self.curriculum_boards_used + self.fallback_boards_used
        
        stats = {
            'curriculum_boards_used': self.curriculum_boards_used,
            'fallback_boards_used': self.fallback_boards_used,
            'pool_empty_count': self.pool_empty_count,
            'total_boards_used': total_boards,
            'curriculum_usage_rate': (
                self.curriculum_boards_used / total_boards if total_boards > 0 else 0.0
            ),
            'current_curriculum_k': self.current_curriculum_k,
            'epsilon_random': self.epsilon_random
        }
        
        # Add board pool stats if available
        if self.board_pool_manager:
            pool_stats = self.board_pool_manager.get_stats()
            stats.update({
                'pool_size': pool_stats.get('current_size', 0),
                'pool_utilization': pool_stats.get('utilization', 0.0),
                'target_k': pool_stats.get('current_k', None)
            })
        
        return stats
    
    def set_curriculum_level(self, k: int):
        """
        Update curriculum difficulty level.
        
        Args:
            k: New curriculum difficulty level
        """
        if self.board_pool_manager:
            self.board_pool_manager.set_curriculum_level(k)
    
    def close(self):
        """Clean up resources."""
        if self.board_pool_manager:
            self.board_pool_manager.stop()
        super().close()


def create_curriculum_env(curriculum_config: Dict[str, Any], 
                         max_steps: int = 20,
                         render_mode: Optional[str] = None) -> CurriculumRicochetEnv:
    """
    Factory function to create a curriculum environment with board pool.
    
    Args:
        curriculum_config: Configuration for curriculum learning
        max_steps: Maximum steps per episode
        render_mode: Rendering mode
        
    Returns:
        Initialized CurriculumRicochetEnv
    """
    from .board_pool import create_board_pool_manager
    from environment.simpler_ricochet_env import RicochetRobotsEnvOneStepAway
    
    # Create board pool manager
    board_pool_manager = create_board_pool_manager(curriculum_config)
    
    # Start the board pool (this will begin generating boards)
    board_pool_manager.start()
    
    # Create curriculum environment
    env = CurriculumRicochetEnv(
        board_pool_manager=board_pool_manager,
        max_steps=max_steps,
        render_mode=render_mode,
        epsilon_random=curriculum_config.get('epsilon_random', 0.05),
        fallback_env_class=RicochetRobotsEnvOneStepAway
    )
    
    return env 