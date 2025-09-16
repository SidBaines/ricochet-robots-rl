"""
Centralized curriculum configuration for Ricochet Robots.

This module provides a single source of truth for curriculum level definitions,
ensuring consistency between precomputation, training, and evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Any
from dataclasses import dataclass
from .puzzle_bank import SpecKey


@dataclass
class CurriculumLevel:
    """A single curriculum level definition."""
    level: int
    name: str
    spec_key: SpecKey
    min_optimal_length: int
    max_optimal_length: int
    min_robots_moved: int
    max_robots_moved: int
    description: str = ""
    num_puzzles: int = 1000  # Default number of puzzles to generate


def get_default_curriculum_levels() -> List[CurriculumLevel]:
    """Get the default curriculum levels.
    
    Returns:
        List of curriculum level definitions
    """
    return [
        # Level 0: Single Robot present, Single Move
        CurriculumLevel(
            level=0,
            name="Single Robot present, Single Move",
            spec_key=SpecKey(
                height=16, width=16, num_robots=1,
                edge_t_per_quadrant=2, central_l_per_quadrant=2
            ),
            min_optimal_length=1,
            max_optimal_length=1,
            min_robots_moved=1,
            max_robots_moved=1,
            description="Single robot with some internal walls, single move solution puzzles",
            num_puzzles=1000
        ),
        
        # Level 1: Single robot, some walls, more permissive
        CurriculumLevel(
            level=1,
            name="Single Robot present, single robot moved, Up to two Moves",
            spec_key=SpecKey(
                height=16, width=16, num_robots=1,
                edge_t_per_quadrant=2, central_l_per_quadrant=2
            ),
            min_optimal_length=1,
            max_optimal_length=2,
            min_robots_moved=1,
            max_robots_moved=1,
            description="Single robot with some internal walls, easy puzzles, <= 2 moves",
            num_puzzles=1000
        ),
        
        # Level 2: Single robot, more walls, more permissive
        CurriculumLevel(
            level=2,
            name="Multiple robots present, single robot moved, Up to two Moves",
            spec_key=SpecKey(
                height=16, width=16, num_robots=4,
                edge_t_per_quadrant=1, central_l_per_quadrant=1
            ),
            min_optimal_length=1,
            max_optimal_length=2,
            min_robots_moved=1,
            max_robots_moved=1,
            description="Multiple robots with simple wall configurations, single robot solution puzzles, <= 2 moves",
            num_puzzles=1000
        ),
        
        # Level 3: Single robot, <= four moves
        CurriculumLevel(
            level=3,
            name="Multiple robots present, single robot moved, Up to four Moves",
            spec_key=SpecKey(
                height=16, width=16, num_robots=4,
                edge_t_per_quadrant=1, central_l_per_quadrant=1
            ),
            min_optimal_length=1,
            max_optimal_length=4,
            min_robots_moved=1,
            max_robots_moved=1,
            description="Multiple robots with simple wall configurations, single robot solution puzzles, <= 4 moves",
            num_puzzles=1000
        ),
        
        # Level 4: Single robot, <= eight moves
        CurriculumLevel(
            level=4,
            name="Multiple robots present, single robot moved, Up to eight Moves",
            spec_key=SpecKey(
                height=16, width=16, num_robots=4,
                edge_t_per_quadrant=2, central_l_per_quadrant=2
            ),
            min_optimal_length=1,
            max_optimal_length=8,
            min_robots_moved=1,
            max_robots_moved=2,
            description="Multiple robots with simple wall configurations, single robot solution puzzles, <= 8 moves",
            num_puzzles=1000
        ),
        
        # Level 5: Three robots, expert level, more permissive
        CurriculumLevel(
            level=5,
            name="Multiple robots present, multiple robots moved, Up to four Moves",
            spec_key=SpecKey(
                height=16, width=16, num_robots=4,
                edge_t_per_quadrant=2, central_l_per_quadrant=2
            ),
            min_optimal_length=1,
            max_optimal_length=4,  # More permissive
            min_robots_moved=1,
            max_robots_moved=2,
            description="Multiple robots with simple wall configurations, multiple robot solution puzzles, <= 4 moves",
            num_puzzles=1000
        ),

        # Level 6: Multiple robots, <= 5 moves
        CurriculumLevel(
            level=6,
            name="Multiple robots present, multiple robots moved, Up to five Moves",
            spec_key=SpecKey(
                height=16, width=16, num_robots=4,
                edge_t_per_quadrant=2, central_l_per_quadrant=2
            ),
            min_optimal_length=1,
            max_optimal_length=5,
            min_robots_moved=1,
            max_robots_moved=2,
            description="Multiple robots with simple wall configurations, multiple robot solution puzzles, <= 5 moves",
            num_puzzles=1000
        )
    ]


def get_curriculum_levels_as_dicts() -> List[Dict[str, Any]]:
    """Get curriculum levels as dictionaries for JSON serialization.
    
    Returns:
        List of curriculum level dictionaries
    """
    levels = get_default_curriculum_levels()
    return [
        {
            "level": level.level,
            "name": level.name,
            "spec_key": {
                "height": level.spec_key.height,
                "width": level.spec_key.width,
                "num_robots": level.spec_key.num_robots,
                "edge_t_per_quadrant": level.spec_key.edge_t_per_quadrant,
                "central_l_per_quadrant": level.spec_key.central_l_per_quadrant,
                "generator_rules_version": level.spec_key.generator_rules_version
            },
            "min_optimal_length": level.min_optimal_length,
            "max_optimal_length": level.max_optimal_length,
            "min_robots_moved": level.min_robots_moved,
            "max_robots_moved": level.max_robots_moved,
            "description": level.description,
            "num_puzzles": level.num_puzzles
        }
        for level in levels
    ]


def get_curriculum_specs_for_precomputation() -> List[Dict[str, Any]]:
    """Get curriculum specs for precomputation pipeline.
    
    Returns:
        List of curriculum specifications for puzzle generation
    """
    levels = get_default_curriculum_levels()
    return [
        {
            "level": level.level,
            "name": level.name,
            "spec_key": level.spec_key,
            "min_optimal_length": level.min_optimal_length,
            "max_optimal_length": level.max_optimal_length,
            "min_robots_moved": level.min_robots_moved,
            "max_robots_moved": level.max_robots_moved,
            "num_puzzles": level.num_puzzles
        }
        for level in levels
    ]


def get_bank_curriculum_levels() -> List[Dict[str, Any]]:
    """Get curriculum levels for bank-based learning.
    
    Returns:
        List of curriculum level specifications for bank curriculum
    """
    levels = get_default_curriculum_levels()
    return [
        {
            "level": level.level,
            "name": level.name,
            "spec_key": level.spec_key,
            "min_optimal_length": level.min_optimal_length,
            "max_optimal_length": level.max_optimal_length,
            "min_robots_moved": level.min_robots_moved,
            "max_robots_moved": level.max_robots_moved,
            "description": level.description
        }
        for level in levels
    ]


def save_curriculum_config_to_json(filepath: str) -> None:
    """Save curriculum configuration to JSON file.
    
    Args:
        filepath: Path to save the JSON file
    """
    import json
    
    config = {
        "levels": get_curriculum_levels_as_dicts(),
        "success_rate_threshold": 0.75,
        "min_episodes_per_level": 200,
        "success_rate_window_size": 100,
        "advancement_check_frequency": 50,
        "max_level": 5
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def load_curriculum_config_from_json(filepath: str) -> List[CurriculumLevel]:
    """Load curriculum configuration from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of curriculum level definitions
    """
    import json
    from .puzzle_bank import SpecKey
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    levels = []
    for level_dict in config["levels"]:
        spec_key_dict = level_dict["spec_key"]
        spec_key = SpecKey(
            height=spec_key_dict["height"],
            width=spec_key_dict["width"],
            num_robots=spec_key_dict["num_robots"],
            edge_t_per_quadrant=spec_key_dict["edge_t_per_quadrant"],
            central_l_per_quadrant=spec_key_dict["central_l_per_quadrant"],
            generator_rules_version=spec_key_dict["generator_rules_version"]
        )
        
        level = CurriculumLevel(
            level=level_dict["level"],
            name=level_dict["name"],
            spec_key=spec_key,
            min_optimal_length=level_dict["min_optimal_length"],
            max_optimal_length=level_dict["max_optimal_length"],
            min_robots_moved=level_dict["min_robots_moved"],
            max_robots_moved=level_dict["max_robots_moved"],
            description=level_dict.get("description", ""),
            num_puzzles=level_dict.get("num_puzzles", 1000)
        )
        levels.append(level)
    
    return levels
