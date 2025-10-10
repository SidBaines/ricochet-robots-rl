"""
Curriculum Learning Wrapper for Ricochet Robots Environment

This module provides a curriculum wrapper that progressively increases difficulty
during training by adjusting environment parameters based on agent performance.
Now supports both online solving and bank-based curriculum learning.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from collections import deque

from .ricochet_env import RicochetRobotsEnv, GymEnvBase
from .puzzle_bank import PuzzleBank, SpecKey
from .criteria_env import CriteriaFilteredEnv, PuzzleCriteria, BankCurriculumManager


class OptimalLengthFilteredEnv(RicochetRobotsEnv):
    """
    Environment wrapper that filters puzzles by optimal solution length.
    
    This ensures that only puzzles with optimal_length <= max_optimal_length
    are generated for curriculum learning.
    """
    
    def __init__(self, max_optimal_length: int, *args, **kwargs):
        """
        Initialize with optimal length filtering.
        
        Args:
            max_optimal_length: Maximum allowed optimal solution length
            *args, **kwargs: Arguments passed to RicochetRobotsEnv
        """
        self.max_optimal_length = max_optimal_length
        super().__init__(*args, **kwargs)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset and generate a puzzle with optimal length <= max_optimal_length."""
        # Generate puzzles until we find one with acceptable optimal length
        attempts = 0
        max_attempts = 100  # Prevent infinite loops
        
        while attempts < max_attempts:
            attempts += 1
            obs, info = super().reset(seed=seed, options=options)
            
            # Check if the puzzle has acceptable optimal length
            if not self.ensure_solvable:
                # If solvability not enforced, accept any puzzle
                return obs, info
            
            optimal_length = info.get('optimal_length')
            if optimal_length is not None and optimal_length <= self.max_optimal_length:
                return obs, info
            
            # If we're using a fixed layout, we can't regenerate
            if self.fixed_layout is not None:
                return obs, info
        
        # If we couldn't find a suitable puzzle, return the last one generated
        # This ensures we don't get stuck in infinite loops
        return obs, info


@dataclass
class CurriculumLevel:
    """Configuration for a single curriculum level."""
    level: int
    name: str
    height: int
    width: int
    num_robots: int
    edge_t_per_quadrant: int
    central_l_per_quadrant: int
    max_optimal_length: int
    solver_max_depth: int
    solver_max_nodes: int
    description: str


@dataclass
class CurriculumConfig:
    """Configuration for curriculum progression."""
    levels: List[CurriculumLevel]
    success_rate_threshold: float = 0.8
    min_episodes_per_level: int = 100
    success_rate_window_size: int = 200
    advancement_check_frequency: int = 50  # Check every N episodes
    max_level: Optional[int] = None  # None means use all levels


class CurriculumManager:
    """
    Manages curriculum progression across multiple environments.
    
    This class coordinates curriculum advancement based on aggregate performance
    across all vectorized environments.
    """
    
    def __init__(self, curriculum_config: CurriculumConfig, initial_level: int = 0, verbose: bool = True):
        """Initialize curriculum manager."""
        self.config = curriculum_config
        self.current_level = initial_level
        self.verbose = verbose
        
        # Validate curriculum configuration
        self._validate_curriculum_config()
        
        # Success rate tracking across all environments
        self.success_rate_window = deque(maxlen=curriculum_config.success_rate_window_size)
        self.episode_count = 0
        self.level_start_episode = 0
        
        if self.verbose:
            print(f"Curriculum manager initialized at level {self.current_level}: {self._get_current_level().name}")
    
    def _validate_curriculum_config(self) -> None:
        """Validate that the curriculum configuration is sensible."""
        if not self.config.levels:
            raise ValueError("CurriculumConfig must have at least one level")
        
        if self.current_level >= len(self.config.levels):
            raise ValueError(f"Initial level {self.current_level} exceeds available levels {len(self.config.levels)}")
        
        # Validate level progression
        for i, level in enumerate(self.config.levels):
            if level.level != i:
                raise ValueError(f"Level {i} has incorrect level number {level.level}")
            
            if level.height <= 0 or level.width <= 0:
                raise ValueError(f"Level {i} has invalid dimensions: {level.height}x{level.width}")
            
            if level.num_robots <= 0:
                raise ValueError(f"Level {i} has invalid number of robots: {level.num_robots}")
            
            if level.max_optimal_length <= 0:
                raise ValueError(f"Level {i} has invalid max_optimal_length: {level.max_optimal_length}")
            
            # Check that levels progress in difficulty
            if i > 0:
                prev_level = self.config.levels[i-1]
                
                # Each level should be at least as difficult as the previous
                if (level.height < prev_level.height or 
                    level.width < prev_level.width or 
                    level.num_robots < prev_level.num_robots):
                    raise ValueError(f"Level {i} is easier than level {i-1} in some dimensions")
                
                # Optimal length should generally increase (allowing some flexibility)
                if level.max_optimal_length < prev_level.max_optimal_length * 0.8:
                    raise ValueError(f"Level {i} max_optimal_length ({level.max_optimal_length}) is too small compared to level {i-1} ({prev_level.max_optimal_length})")
        
        # Validate configuration parameters
        if not 0 < self.config.success_rate_threshold <= 1:
            raise ValueError(f"success_rate_threshold must be in (0, 1], got {self.config.success_rate_threshold}")
        
        if self.config.min_episodes_per_level <= 0:
            raise ValueError(f"min_episodes_per_level must be positive, got {self.config.min_episodes_per_level}")
        
        if self.config.success_rate_window_size <= 0:
            raise ValueError(f"success_rate_window_size must be positive, got {self.config.success_rate_window_size}")
        
        if self.config.advancement_check_frequency <= 0:
            raise ValueError(f"advancement_check_frequency must be positive, got {self.config.advancement_check_frequency}")
    
    def _get_current_level(self) -> CurriculumLevel:
        """Get the current curriculum level configuration."""
        return self.config.levels[self.current_level]
    
    def record_episode_result(self, success: bool) -> None:
        """Record the result of an episode from any environment."""
        self.episode_count += 1
        self.success_rate_window.append(success)
        
        # Check for curriculum advancement
        if self._should_check_advancement():
            self._check_advancement()
    
    def _should_check_advancement(self) -> bool:
        """Check if it's time to evaluate curriculum advancement."""
        if self.current_level >= len(self.config.levels) - 1:
            return False  # Already at max level
        
        if self.config.max_level is not None and self.current_level >= self.config.max_level:
            return False  # Reached configured max level
        
        # Check if we have enough episodes at current level
        episodes_at_level = self.episode_count - self.level_start_episode
        if episodes_at_level < self.config.min_episodes_per_level:
            return False
        
        # Check if it's time for advancement check
        return episodes_at_level % self.config.advancement_check_frequency == 0
    
    def _check_advancement(self) -> None:
        """Check if agent performance warrants advancing to next level."""
        if len(self.success_rate_window) < self.config.success_rate_window_size:
            return  # Not enough data yet
        
        success_rate = np.mean(self.success_rate_window)
        
        if success_rate >= self.config.success_rate_threshold:
            self._advance_level()
    
    def _advance_level(self) -> None:
        """Advance to the next curriculum level."""
        old_level = self.current_level
        self.current_level += 1
        self.level_start_episode = self.episode_count
        
        # Clear success rate window for new level
        self.success_rate_window.clear()
        
        if self.verbose:
            new_level = self._get_current_level()
            print(f"Curriculum advanced from level {old_level} to {self.current_level}: {new_level.name}")
            print(f"  New parameters: {new_level.height}x{new_level.width}, {new_level.num_robots} robots")
            print(f"  Wall complexity: {new_level.edge_t_per_quadrant} edge-T, {new_level.central_l_per_quadrant} central-L per quadrant")
            print(f"  Max optimal length: {new_level.max_optimal_length}")
    
    def get_current_level(self) -> int:
        """Get current curriculum level."""
        return self.current_level
    
    def get_success_rate(self) -> float:
        """Get current success rate over the sliding window."""
        if not self.success_rate_window:
            return 0.0
        return float(np.mean(self.success_rate_window))
    
    def get_episodes_at_level(self) -> int:
        """Get number of episodes completed at current level."""
        return self.episode_count - self.level_start_episode
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        return {
            'current_level': self.current_level,
            'level_name': self._get_current_level().name,
            'success_rate': self.get_success_rate(),
            'episodes_at_level': self.get_episodes_at_level(),
            'total_episodes': self.episode_count,
            'window_size': len(self.success_rate_window),
        }

    # Backwards-compatibility shim for call sites expecting BankCurriculumManager API
    def get_stats(self) -> Dict[str, Any]:
        """Alias for :meth:`get_curriculum_stats` used by older logging hooks."""
        return self.get_curriculum_stats()


class CurriculumWrapper(GymEnvBase):
    """
    Wrapper that implements curriculum learning by progressively increasing
    environment difficulty based on agent performance.
    
    This wrapper works with a shared CurriculumManager to coordinate
    curriculum progression across vectorized environments.
    """
    
    def __init__(
        self,
        base_env_factory: Callable[[], RicochetRobotsEnv],
        curriculum_manager: CurriculumManager,
        verbose: bool = True
    ):
        """
        Initialize curriculum wrapper.
        
        Args:
            base_env_factory: Function that creates base environment instances
            curriculum_manager: Shared curriculum manager for coordination
            verbose: Whether to print curriculum progression messages
        """
        self.base_env_factory = base_env_factory
        self.curriculum_manager = curriculum_manager
        self.verbose = verbose
        
        # Create current environment
        self._current_env = None
        self._last_known_level = self.curriculum_manager.current_level
        self._create_current_env()
        
        if self.verbose:
            level = self.curriculum_manager.config.levels[self.curriculum_manager.current_level]
            print(f"Curriculum wrapper initialized at level {self.curriculum_manager.current_level}: {level.name}")
    
    def _get_current_level(self) -> CurriculumLevel:
        """Get the current curriculum level configuration."""
        return self.curriculum_manager.config.levels[self.curriculum_manager.current_level]
    
    def _create_current_env(self) -> None:
        """Create a new environment instance with current level parameters."""
        level = self._get_current_level()
        
        # Get observation mode and other settings from the base environment factory
        # by creating a temporary environment to inspect its settings
        temp_env = self.base_env_factory()
        obs_mode = temp_env.obs_mode
        channels_first = temp_env.channels_first
        temp_env.close()
        
        # Create environment with level-specific parameters and optimal length filtering
        self._current_env = OptimalLengthFilteredEnv(
            max_optimal_length=level.max_optimal_length,
            height=level.height,
            width=level.width,
            num_robots=level.num_robots,
            edge_t_per_quadrant=level.edge_t_per_quadrant,
            central_l_per_quadrant=level.central_l_per_quadrant,
            solver_max_depth=level.solver_max_depth,
            solver_max_nodes=level.solver_max_nodes,
            ensure_solvable=True,  # Always ensure solvable for curriculum
            obs_mode=obs_mode,  # Use the same observation mode as base environment
            channels_first=channels_first,  # Use the same channels_first setting
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and return initial observation."""
        return self._current_env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Step the environment and track success for curriculum progression."""
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        
        # Track episode completion and success
        if terminated or truncated:
            success = info.get('is_success', False) and not truncated
            self.curriculum_manager.record_episode_result(success)
            
            # Check if curriculum level has changed and recreate environment if needed
            if self.curriculum_manager.current_level != self._last_known_level:
                self._create_current_env()
                self._last_known_level = self.curriculum_manager.current_level
        
        return obs, reward, terminated, truncated, info
    
    def get_current_level(self) -> int:
        """Get current curriculum level."""
        return self.curriculum_manager.get_current_level()
    
    def get_success_rate(self) -> float:
        """Get current success rate over the sliding window."""
        return self.curriculum_manager.get_success_rate()
    
    def get_episodes_at_level(self) -> int:
        """Get number of episodes completed at current level."""
        return self.curriculum_manager.get_episodes_at_level()
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        return self.curriculum_manager.get_curriculum_stats()
    
    # Delegate other methods to current environment
    def render(self, *args, **kwargs):
        return self._current_env.render(*args, **kwargs)
    
    def close(self, *args, **kwargs):
        return self._current_env.close(*args, **kwargs)
    
    def get_board(self):
        """Forward underlying board for visualization utilities."""
        return self._current_env.get_board()
    
    @property
    def action_space(self):
        return self._current_env.action_space
    
    @property
    def observation_space(self):
        return self._current_env.observation_space
    
    @property
    def metadata(self):
        return self._current_env.metadata


def create_default_curriculum() -> CurriculumConfig:
    """
    Create a default curriculum configuration based on the research plan.
    
    Levels progress gradually from simple to complex:
    - Level 0: Single robot, no walls, trivial puzzles (1-2 moves) - sanity check
    - Level 1: Single robot, some walls, simple puzzles (2-4 moves) - like a maze
    - Level 2: Multiple robots, simple configurations, medium puzzles (4-6 moves)
    - Level 3: Multiple robots, more walls, harder puzzles (6-10 moves)
    - Level 4: Full complexity, many walls, hard puzzles (10+ moves)
    """
    levels = [
        CurriculumLevel(
            level=0,
            name="Single Robot, No Walls",
            height=16,
            width=16,
            num_robots=1,
            edge_t_per_quadrant=0,
            central_l_per_quadrant=0,
            max_optimal_length=2,
            solver_max_depth=10,
            solver_max_nodes=1000,
            description="Single robot with no internal walls, trivial puzzles - sanity check"
        ),
        CurriculumLevel(
            level=1,
            name="Single Robot, Some Walls",
            height=16,
            width=16,
            num_robots=1,
            edge_t_per_quadrant=1,
            central_l_per_quadrant=1,
            max_optimal_length=4,
            solver_max_depth=15,
            solver_max_nodes=5000,
            description="Single robot with some walls, simple puzzles - like a maze"
        ),
        CurriculumLevel(
            level=2,
            name="Multiple Robots, Simple Configurations",
            height=16,
            width=16,
            num_robots=2,
            edge_t_per_quadrant=1,
            central_l_per_quadrant=1,
            max_optimal_length=6,
            solver_max_depth=20,
            solver_max_nodes=10000,
            description="Multiple robots with simple wall configurations"
        ),
        CurriculumLevel(
            level=3,
            name="Multiple Robots, More Walls",
            height=16,
            width=16,
            num_robots=2,
            edge_t_per_quadrant=2,
            central_l_per_quadrant=2,
            max_optimal_length=5,
            solver_max_depth=30,
            solver_max_nodes=20000,
            description="Multiple robots with moderate wall complexity"
        ),
        CurriculumLevel(
            level=4,
            name="Full Complexity",
            height=16,
            width=16,
            num_robots=3,
            edge_t_per_quadrant=2,
            central_l_per_quadrant=2,
            max_optimal_length=10,
            solver_max_depth=40,
            solver_max_nodes=30000,
            description="Full complexity with many robots and walls"
        ),
    ]
    
    return CurriculumConfig(
        levels=levels,
        success_rate_threshold=0.8,
        min_episodes_per_level=100,
        success_rate_window_size=200,
        advancement_check_frequency=50,
    )


def create_curriculum_manager(
    curriculum_config: Optional[CurriculumConfig] = None,
    initial_level: int = 0,
    verbose: bool = True
) -> CurriculumManager:
    """
    Create a curriculum manager with default configuration.
    
    Args:
        curriculum_config: Optional custom curriculum configuration
        initial_level: Starting difficulty level
        verbose: Whether to print curriculum progression messages
    
    Returns:
        Configured CurriculumManager instance
    """
    if curriculum_config is None:
        curriculum_config = create_default_curriculum()
    
    return CurriculumManager(
        curriculum_config=curriculum_config,
        initial_level=initial_level,
        verbose=verbose
    )


def create_curriculum_wrapper(
    base_env_factory: Callable[[], RicochetRobotsEnv],
    curriculum_manager: CurriculumManager,
    verbose: bool = True
) -> CurriculumWrapper:
    """
    Create a curriculum wrapper with a shared curriculum manager.
    
    Args:
        base_env_factory: Function that creates base environment instances
        curriculum_manager: Shared curriculum manager for coordination
        verbose: Whether to print curriculum progression messages
    
    Returns:
        Configured CurriculumWrapper instance
    """
    return CurriculumWrapper(
        base_env_factory=base_env_factory,
        curriculum_manager=curriculum_manager,
        verbose=verbose
    )


# Bank-based curriculum functions

def create_bank_curriculum_levels() -> List[Dict[str, Any]]:
    """Create curriculum levels for bank-based learning.
    
    Returns:
        List of curriculum level specifications
    """
    from .curriculum_config import get_bank_curriculum_levels
    return get_bank_curriculum_levels()


def create_bank_curriculum_manager(
    bank: PuzzleBank,
    curriculum_levels: Optional[List[Dict[str, Any]]] = None,
    success_rate_threshold: float = 0.8,
    min_episodes_per_level: int = 100,
    success_rate_window_size: int = 200,
    advancement_check_frequency: int = 50,
    verbose: bool = True
) -> BankCurriculumManager:
    """Create a bank-based curriculum manager.
    
    Args:
        bank: Puzzle bank to sample from
        curriculum_levels: Optional custom curriculum levels
        success_rate_threshold: Success rate required to advance
        min_episodes_per_level: Minimum episodes before advancement check
        success_rate_window_size: Window size for success rate calculation
        advancement_check_frequency: How often to check for advancement
        verbose: Whether to print progress
    
    Returns:
        Configured BankCurriculumManager instance
    """
    if curriculum_levels is None:
        curriculum_levels = create_bank_curriculum_levels()
    
    return BankCurriculumManager(
        bank=bank,
        curriculum_levels=curriculum_levels,
        success_rate_threshold=success_rate_threshold,
        min_episodes_per_level=min_episodes_per_level,
        success_rate_window_size=success_rate_window_size,
        advancement_check_frequency=advancement_check_frequency,
        verbose=verbose
    )


def create_bank_curriculum_wrapper(
    bank: PuzzleBank,
    curriculum_manager: BankCurriculumManager,
    obs_mode: str = "rgb_image",
    channels_first: bool = True,
    render_mode: Optional[str] = None,
    verbose: bool = True
) -> CriteriaFilteredEnv:
    """Create a bank-based curriculum wrapper.
    
    Args:
        bank: Puzzle bank to sample from
        curriculum_manager: Bank curriculum manager
        obs_mode: Observation mode
        channels_first: Whether to use channels-first format
        render_mode: Rendering mode
        verbose: Whether to print debug information
    
    Returns:
        Configured CriteriaFilteredEnv instance
    """
    # Get current criteria from curriculum manager
    criteria = curriculum_manager.get_current_criteria()
    
    return CriteriaFilteredEnv(
        bank=bank,
        criteria=criteria,
        obs_mode=obs_mode,
        channels_first=channels_first,
        render_mode=render_mode,
        verbose=verbose
    )


class BankCurriculumWrapper(GymEnvBase):
    """
    Wrapper that implements bank-based curriculum learning by progressively
    increasing difficulty based on agent performance.
    
    This wrapper uses the puzzle bank to provide puzzles matching specific
    criteria without requiring online solving.
    """
    
    def __init__(
        self,
        bank: PuzzleBank,
        curriculum_manager: BankCurriculumManager,
        obs_mode: str = "rgb_image",
        channels_first: bool = True,
        include_noop: bool = False,
        render_mode: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize bank curriculum wrapper.
        
        Args:
            bank: Puzzle bank to sample from
            curriculum_manager: Bank curriculum manager
            obs_mode: Observation mode
            channels_first: Whether to use channels-first format
            render_mode: Rendering mode
            verbose: Whether to print debug information
        """
        self.bank = bank
        self.curriculum_manager = curriculum_manager
        self.obs_mode = obs_mode
        self.channels_first = channels_first
        self.include_noop = include_noop
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Create current environment
        self._current_env = None
        self._last_known_level = self.curriculum_manager.current_level
        self._create_current_env()
        
        if self.verbose:
            level = self.curriculum_manager.curriculum_levels[self.curriculum_manager.current_level]
            print(f"Bank curriculum wrapper initialized at level {self.curriculum_manager.current_level}: {level['name']}")
    
    def _create_current_env(self) -> None:
        """Create a new environment instance with current level criteria."""
        criteria = self.curriculum_manager.get_current_criteria()
        
        self._current_env = CriteriaFilteredEnv(
            bank=self.bank,
            criteria=criteria,
            obs_mode=self.obs_mode,
            channels_first=self.channels_first,
            include_noop=self.include_noop,
            render_mode=self.render_mode,
            verbose=self.verbose
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and return initial observation."""
        return self._current_env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Step the environment and track success for curriculum progression."""
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        
        # Track episode completion and success
        if terminated or truncated:
            success = info.get('is_success', False) and not truncated
            self.curriculum_manager.record_episode_result(success)
            
            # Check if curriculum level has changed and recreate environment if needed
            if self.curriculum_manager.current_level != self._last_known_level:
                self._create_current_env()
                self._last_known_level = self.curriculum_manager.current_level
        
        return obs, reward, terminated, truncated, info
    
    def get_current_level(self) -> int:
        """Get current curriculum level."""
        return self.curriculum_manager.get_current_level()
    
    def get_success_rate(self) -> float:
        """Get current success rate over the sliding window."""
        return self.curriculum_manager.get_success_rate()
    
    def get_episodes_at_level(self) -> int:
        """Get number of episodes completed at current level."""
        return self.curriculum_manager.get_episodes_at_level()
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        return self.curriculum_manager.get_stats()
    
    # Delegate other methods to current environment
    def render(self, *args, **kwargs):
        return self._current_env.render(*args, **kwargs)
    
    def close(self, *args, **kwargs):
        return self._current_env.close(*args, **kwargs)
    
    def get_board(self):
        """Forward underlying board for visualization utilities."""
        return self._current_env.get_board()
    
    @property
    def action_space(self):
        return self._current_env.action_space
    
    @property
    def observation_space(self):
        return self._current_env.observation_space
    
    @property
    def metadata(self):
        return self._current_env.metadata
