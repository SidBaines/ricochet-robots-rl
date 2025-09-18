"""
Criteria-Filtered Environment for Bank-Based Curriculum Learning

This module provides an environment wrapper that uses the puzzle bank to provide
puzzles matching specific criteria without requiring online solving.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .puzzle_bank import (
    PuzzleBank, BankSampler, PuzzleMetadata, SpecKey, 
    create_fixed_layout_from_seed
)
from .ricochet_env import RicochetRobotsEnv, FixedLayout, GymEnvBase


@dataclass
class PuzzleCriteria:
    """Criteria for filtering puzzles from the bank."""
    spec_key: SpecKey
    min_optimal_length: Optional[int] = None
    max_optimal_length: Optional[int] = None
    min_robots_moved: Optional[int] = None
    max_robots_moved: Optional[int] = None
    # Optional stratified bands: list of dicts with ol/rm ranges and optional weight
    bands: Optional[List[Dict[str, Any]]] = None
    
    def matches(self, metadata: PuzzleMetadata) -> bool:
        """Check if puzzle metadata matches criteria.
        
        Args:
            metadata: Puzzle metadata to check
            
        Returns:
            True if matches criteria
        """
        # Check spec key
        if metadata.spec_key != self.spec_key:
            return False
        
        # Check optimal length
        if self.min_optimal_length is not None:
            if metadata.optimal_length < self.min_optimal_length:
                return False
        if self.max_optimal_length is not None:
            if metadata.optimal_length > self.max_optimal_length:
                return False
        
        # Check robots moved
        if self.min_robots_moved is not None:
            if metadata.robots_moved < self.min_robots_moved:
                return False
        if self.max_robots_moved is not None:
            if metadata.robots_moved > self.max_robots_moved:
                return False
        
        return True


class CriteriaFilteredEnv(GymEnvBase):
    """
    Environment wrapper that provides puzzles from the bank matching specific criteria.
    
    This environment uses the puzzle bank to provide puzzles without requiring
    online solving, enabling tight criteria and stable performance.
    """
    
    def __init__(
        self,
        bank: PuzzleBank,
        criteria: PuzzleCriteria,
        obs_mode: str = "rgb_image",
        channels_first: bool = True,
        include_noop: bool = False,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        allow_fallback: bool = False,
    ):
        """Initialize criteria-filtered environment.
        
        Args:
            bank: Puzzle bank to sample from
            criteria: Criteria for puzzle selection
            obs_mode: Observation mode ("image", "rgb_image", "symbolic")
            channels_first: Whether to use channels-first format
            render_mode: Rendering mode
            verbose: Whether to print debug information
        """
        self.bank = bank
        self.criteria = criteria
        self.obs_mode = obs_mode
        self.channels_first = channels_first
        self.include_noop = include_noop
        self.verbose = verbose
        self.allow_fallback = allow_fallback
        
        # Initialize sampler
        self.sampler = BankSampler(bank, sampling_strategy="uniform")
        
        # Current puzzle metadata
        self._current_metadata: Optional[PuzzleMetadata] = None
        self._current_env: Optional[RicochetRobotsEnv] = None
        
        # Statistics
        self._total_episodes = 0
        self._successful_episodes = 0
        self._failed_samples = 0
        # Episode step counter (for optimality gap without online solving)
        self._episode_steps: int = 0
        
        # Create initial environment
        self._create_current_env()
        
        # Set up render mode
        self.render_mode = render_mode
        if self.render_mode is not None and self.render_mode not in ["ascii", "rgb"]:
            raise ValueError(f"Unsupported render_mode {self.render_mode}")
    
    def _create_current_env(self) -> None:
        """Create current environment from bank sample."""
        # Sample puzzle from bank (optionally stratified)
        if self.criteria.bands:
            samples = self.sampler.sample_puzzles_stratified(
                spec_key=self.criteria.spec_key,
                total_count=1,
                bands=self.criteria.bands,
                random_seed=None,
            )
            metadata = samples[0] if samples else None
        else:
            metadata = self.sampler.sample_puzzle(
                spec_key=self.criteria.spec_key,
                min_optimal_length=self.criteria.min_optimal_length,
                max_optimal_length=self.criteria.max_optimal_length,
                min_robots_moved=self.criteria.min_robots_moved,
                max_robots_moved=self.criteria.max_robots_moved
            )
        
        if metadata is None:
            if self.verbose:
                print(f"Warning: No puzzle found matching criteria {self.criteria}")
            self._failed_samples += 1
            # Fallback only if explicitly allowed
            if self.allow_fallback:
                self._create_fallback_env()
            else:
                raise RuntimeError("No puzzle found matching criteria and fallback disabled. Please increase bank coverage or relax criteria.")
            return
        
        self._current_metadata = metadata
        
        try:
            # Create FixedLayout from metadata
            fixed_layout = create_fixed_layout_from_seed(metadata)
            
            # Create environment with fixed layout
            self._current_env = RicochetRobotsEnv(
                fixed_layout=fixed_layout,
                obs_mode=self.obs_mode,
                channels_first=self.channels_first,
                include_noop=self.include_noop,
                render_mode=self.render_mode
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error creating environment from metadata: {e}")
            self._failed_samples += 1
            if self.allow_fallback:
                self._create_fallback_env()
            else:
                raise
    
    def _create_fallback_env(self) -> None:
        """Create fallback environment when bank sampling fails."""
        if self.verbose:
            print("Creating fallback environment")
        
        # Create a simple environment with the spec
        self._current_env = RicochetRobotsEnv(
            height=self.criteria.spec_key.height,
            width=self.criteria.spec_key.width,
            num_robots=self.criteria.spec_key.num_robots,
            edge_t_per_quadrant=self.criteria.spec_key.edge_t_per_quadrant,
            central_l_per_quadrant=self.criteria.spec_key.central_l_per_quadrant,
            ensure_solvable=True,  # Fallback to solvable generation
            obs_mode=self.obs_mode,
            channels_first=self.channels_first,
            include_noop=self.include_noop,
            render_mode=self.render_mode
        )
        
        self._current_metadata = None
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and return initial observation.
        
        Args:
            seed: Random seed (ignored for bank-based sampling)
            options: Additional options (ignored)
            
        Returns:
            (observation, info) tuple
        """
        # Create new environment for this episode
        self._create_current_env()
        
        # Reset environment
        obs, info = self._current_env.reset()
        # Reset step counter at start of episode
        self._episode_steps = 0
        
        # Add bank-specific info
        info.update({
            "bank_metadata": self._current_metadata,
            "criteria": {
                "spec_key": {
                    "height": self.criteria.spec_key.height,
                    "width": self.criteria.spec_key.width,
                    "num_robots": self.criteria.spec_key.num_robots,
                    "edge_t_per_quadrant": self.criteria.spec_key.edge_t_per_quadrant,
                    "central_l_per_quadrant": self.criteria.spec_key.central_l_per_quadrant
                },
                "min_optimal_length": self.criteria.min_optimal_length,
                "max_optimal_length": self.criteria.max_optimal_length,
                "min_robots_moved": self.criteria.min_robots_moved,
                "max_robots_moved": self.criteria.max_robots_moved,
                "bands": self.criteria.bands
            },
            "bank_stats": {
                "total_episodes": self._total_episodes,
                "successful_episodes": self._successful_episodes,
                "failed_samples": self._failed_samples
            }
        })

        # Attach stratified sampling plan if used
        if self.criteria.bands:
            plan = self.sampler.get_last_stratified_plan()
            if plan is not None:
                info["stratified_plan"] = plan
        
        if self._current_metadata is not None:
            info.update({
                "optimal_length": self._current_metadata.optimal_length,
                "robots_moved": self._current_metadata.robots_moved,
                "puzzle_seed": self._current_metadata.seed
            })
        
        return obs, info
    
    def step(self, action):
        """Step environment and return (obs, reward, terminated, truncated, info).
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        # Count steps taken in the current episode
        self._episode_steps += 1
        
        # Track episode completion
        if terminated or truncated:
            self._total_episodes += 1
            if info.get("is_success", False) and not truncated:
                self._successful_episodes += 1
            # If we have optimal solution length from bank metadata, compute gap
            if self._current_metadata is not None:
                optimal_len = int(self._current_metadata.optimal_length)
                # Report raw episode steps and optimality gap only on episode end
                info.setdefault("episode_steps", int(self._episode_steps))
                # Only meaningful if success and not truncated; otherwise it's just steps until termination
                if info.get("is_success", False) and not truncated:
                    info["optimality_gap"] = int(self._episode_steps - optimal_len)
                else:
                    # For failures, still attach gap as None to indicate unavailable
                    info.setdefault("optimality_gap", None)
            # Reset step counter for safety (env will usually be reset by caller)
            self._episode_steps = 0
        
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        """Render environment."""
        if self._current_env is None:
            return None
        return self._current_env.render(*args, **kwargs)

    def get_board(self):
        """Forward underlying board for visualization utilities."""
        if self._current_env is None:
            raise RuntimeError("Environment not initialized")
        return self._current_env.get_board()
    
    def close(self, *args, **kwargs):
        """Close environment."""
        if self._current_env is not None:
            self._current_env.close(*args, **kwargs)
    
    @property
    def action_space(self):
        """Action space."""
        if self._current_env is None:
            raise RuntimeError("Environment not initialized")
        return self._current_env.action_space
    
    @property
    def observation_space(self):
        """Observation space."""
        if self._current_env is None:
            raise RuntimeError("Environment not initialized")
        return self._current_env.observation_space
    
    @property
    def metadata(self):
        """Environment metadata."""
        if self._current_env is None:
            return {"render_modes": ["ascii", "rgb"], "render_fps": 4}
        return self._current_env.metadata
    
    def get_criteria(self) -> PuzzleCriteria:
        """Get current criteria."""
        return self.criteria
    
    def get_bank_stats(self) -> Dict[str, Any]:
        """Get bank statistics."""
        return {
            "total_episodes": self._total_episodes,
            "successful_episodes": self._successful_episodes,
            "failed_samples": self._failed_samples,
            "success_rate": (
                self._successful_episodes / self._total_episodes
                if self._total_episodes > 0 else 0
            )
        }
    
    def update_criteria(self, new_criteria: PuzzleCriteria) -> None:
        """Update puzzle criteria.
        
        Args:
            new_criteria: New criteria to use
        """
        self.criteria = new_criteria
        # Environment will be recreated on next reset


class BankCurriculumManager:
    """Curriculum manager that uses the puzzle bank for level progression."""
    
    def __init__(
        self,
        bank: PuzzleBank,
        curriculum_levels: List[Dict[str, Any]],
        success_rate_threshold: float = 0.8,
        min_episodes_per_level: int = 100,
        success_rate_window_size: int = 200,
        advancement_check_frequency: int = 50,
        verbose: bool = True
    ):
        """Initialize bank-based curriculum manager.
        
        Args:
            bank: Puzzle bank to sample from
            curriculum_levels: List of curriculum level specifications
            success_rate_threshold: Success rate required to advance
            min_episodes_per_level: Minimum episodes before advancement check
            success_rate_window_size: Window size for success rate calculation
            advancement_check_frequency: How often to check for advancement
            verbose: Whether to print progress
        """
        self.bank = bank
        self.curriculum_levels = curriculum_levels
        self.success_rate_threshold = success_rate_threshold
        self.min_episodes_per_level = min_episodes_per_level
        self.success_rate_window_size = success_rate_window_size
        self.advancement_check_frequency = advancement_check_frequency
        self.verbose = verbose
        
        # Current state
        self.current_level = 0
        self.success_rate_window = []
        self.episode_count = 0
        self.level_start_episode = 0
        
        if self.verbose:
            print(f"Bank curriculum manager initialized at level {self.current_level}")
    
    def get_current_criteria(self) -> PuzzleCriteria:
        """Get criteria for current level.
        
        Returns:
            PuzzleCriteria for current level
        """
        level_spec = self.curriculum_levels[self.current_level]
        # Build simple bands across discrete optimal lengths within range
        bands: Optional[List[Dict[str, Any]]] = None
        if (
            level_spec.get("min_optimal_length") is not None and
            level_spec.get("max_optimal_length") is not None
        ):
            try:
                min_ol = int(level_spec.get("min_optimal_length"))
                max_ol = int(level_spec.get("max_optimal_length"))
                min_rm = level_spec.get("min_robots_moved")
                max_rm = level_spec.get("max_robots_moved")
                bands = [
                    {
                        "min_optimal_length": ol,
                        "max_optimal_length": ol,
                        "min_robots_moved": min_rm,
                        "max_robots_moved": max_rm,
                        "weight": 1.0,
                    }
                    for ol in range(min_ol, max_ol + 1)
                ]
            except Exception:
                bands = None

        return PuzzleCriteria(
            spec_key=level_spec["spec_key"],
            min_optimal_length=level_spec.get("min_optimal_length"),
            max_optimal_length=level_spec.get("max_optimal_length"),
            min_robots_moved=level_spec.get("min_robots_moved"),
            max_robots_moved=level_spec.get("max_robots_moved"),
            bands=bands
        )
    
    def record_episode_result(self, success: bool) -> None:
        """Record episode result.
        
        Args:
            success: Whether episode was successful
        """
        self.episode_count += 1
        self.success_rate_window.append(success)
        
        # Trim window if too large
        if len(self.success_rate_window) > self.success_rate_window_size:
            self.success_rate_window = self.success_rate_window[-self.success_rate_window_size:]
        
        # Check for advancement
        if self._should_check_advancement():
            self._check_advancement()
    
    def _should_check_advancement(self) -> bool:
        """Check if it's time to evaluate advancement."""
        if self.current_level >= len(self.curriculum_levels) - 1:
            return False  # Already at max level
        
        # Check if we have enough episodes at current level
        episodes_at_level = self.episode_count - self.level_start_episode
        if episodes_at_level < self.min_episodes_per_level:
            return False
        
        # Check if it's time for advancement check
        return episodes_at_level % self.advancement_check_frequency == 0
    
    def _check_advancement(self) -> None:
        """Check if agent performance warrants advancing to next level."""
        if len(self.success_rate_window) < self.success_rate_window_size:
            return  # Not enough data yet
        
        success_rate = np.mean(self.success_rate_window)
        
        if success_rate >= self.success_rate_threshold:
            self._advance_level()
    
    def _advance_level(self) -> None:
        """Advance to the next curriculum level."""
        old_level = self.current_level
        self.current_level += 1
        self.level_start_episode = self.episode_count
        
        # Clear success rate window for new level
        self.success_rate_window.clear()
        
        if self.verbose:
            level_spec = self.curriculum_levels[self.current_level]
            print(f"Curriculum advanced from level {old_level} to {self.current_level}: {level_spec['name']}")
    
    def get_current_level(self) -> int:
        """Get current curriculum level."""
        return self.current_level
    
    def get_success_rate(self) -> float:
        """Get current success rate."""
        if not self.success_rate_window:
            return 0.0
        return float(np.mean(self.success_rate_window))
    
    def get_episodes_at_level(self) -> int:
        """Get number of episodes at current level."""
        return self.episode_count - self.level_start_episode
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        level_spec = self.curriculum_levels[self.current_level]
        return {
            "current_level": self.current_level,
            "level_name": level_spec["name"],
            "success_rate": self.get_success_rate(),
            "episodes_at_level": self.get_episodes_at_level(),
            "total_episodes": self.episode_count,
            "window_size": len(self.success_rate_window),
            "criteria": self.get_current_criteria()
        }
