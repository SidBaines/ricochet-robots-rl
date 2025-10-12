#!/usr/bin/env python3
"""
Generate puzzle bank for curriculum levels.

This script generates puzzles for all curriculum levels defined in the shared
curriculum configuration. It can use either the default curriculum or load
from a custom JSON configuration file.

Usage:
    # Use default curriculum
    python generate_curriculum_bank.py --bank_dir artifacts/puzzle_bank --puzzles_per_level 1000
    
    # Use custom curriculum config
    python generate_curriculum_bank.py --config my_curriculum.json --bank_dir artifacts/puzzle_bank --puzzles_per_level 1000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from src.env.puzzle_bank import PuzzleBank, SpecKey
from src.env.precompute_pipeline import PuzzleGenerator, run_band_first_precompute_controller
from src.env.curriculum_config import get_curriculum_specs_for_precomputation


def load_curriculum_config(config_path: str) -> Dict[str, Any]:
    """Load curriculum configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_curriculum_bank(
    config_path: str = None,
    bank_dir: str = str(ARTIFACTS_ROOT / "puzzle_bank"),
    puzzles_per_level: int = 1000,
    solver_config: Dict[str, Any] = None,
    use_controller: bool = False,
    controller_target_per_level: int = 1000,
    controller_max_puzzles_global: int = 200000,
    controller_chunk_per_spec: int = 200,
    verbose: bool = False
) -> None:
    """Generate puzzle bank for all curriculum levels.
    
    Args:
        config_path: Path to curriculum config JSON file (optional, uses default if None)
        bank_dir: Directory to store puzzle bank
        puzzles_per_level: Number of puzzles to generate per level
        solver_config: Solver configuration
    """
    if solver_config is None:
        solver_config = {"solver_type": "bfs", "max_depth": 10, "max_nodes": 100000}
    
    # Use shared curriculum config or load from file
    if config_path and Path(config_path).exists():
        config = load_curriculum_config(config_path)
        levels = config["levels"]
        print(f"Loaded curriculum from {config_path} with {len(levels)} levels")
        
        # Convert spec_key dictionaries to SpecKey objects
        for level in levels:
            if isinstance(level["spec_key"], dict):
                spec_dict = level["spec_key"]
                level["spec_key"] = SpecKey(
                    height=spec_dict["height"],
                    width=spec_dict["width"],
                    num_robots=spec_dict["num_robots"],
                    edge_t_per_quadrant=spec_dict["edge_t_per_quadrant"],
                    central_l_per_quadrant=spec_dict["central_l_per_quadrant"],
                    generator_rules_version=spec_dict["generator_rules_version"]
                )
    else:
        # Use default curriculum from shared config
        levels = get_curriculum_specs_for_precomputation()
        print(f"Using default curriculum with {len(levels)} levels")
    
    print(f"Bank directory: {bank_dir}")
    if use_controller:
        print("Using band-first controller for generation")
        print(f"  Target per level: {controller_target_per_level}")
        print(f"  Global max puzzles: {controller_max_puzzles_global}")
        print(f"  Chunk per spec: {controller_chunk_per_spec}")
    else:
        print(f"Generating {puzzles_per_level} puzzles per level")
    print("=" * 60)
    
    # Create bank and generator
    bank = PuzzleBank(bank_dir)
    generator = PuzzleGenerator(bank, solver_config, verbose=verbose)

    if use_controller:
        # Let the controller drive generation using manifest histograms
        # If a config was provided, pass the parsed curriculum levels through
        controller_levels = None
        if config_path and Path(config_path).exists():
            controller_levels = levels
        results = run_band_first_precompute_controller(
            bank_dir=bank_dir,
            target_per_level=controller_target_per_level,
            max_puzzles_global=controller_max_puzzles_global,
            chunk_puzzles_per_spec=controller_chunk_per_spec,
            solver_config=solver_config,
            verbose=verbose,
            curriculum_levels=controller_levels,
        )
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE (controller)")
        print("=" * 60)
        print(f"Total puzzles generated (approx): {results['total_generated']}")
        print(f"Final coverage: {results['final_coverage']}")
        bank_stats = bank.get_stats()
        print("Bank statistics:")
        print(f"  Total puzzles: {bank_stats['total_puzzles']}")
        print(f"  Partitions: {bank_stats['partition_count']}")
        if bank_stats['optimal_length_stats']:
            ol_stats = bank_stats['optimal_length_stats']
            print(f"  Optimal length range: {ol_stats['min']}-{ol_stats['max']}")
            print(f"  Optimal length mean: {ol_stats['mean']:.1f}")
        if bank_stats['robots_moved_stats']:
            rm_stats = bank_stats['robots_moved_stats']
            print(f"  Robots moved range: {rm_stats['min']}-{rm_stats['max']}")
            print(f"  Robots moved mean: {rm_stats['mean']:.1f}")
        return
    
    # Generate puzzles for each level
    total_puzzles = 0
    for level in levels:
        level_name = level["name"]
        spec_key = level["spec_key"]  # Already a SpecKey object from shared config
        
        print(f"\nGenerating puzzles for Level {level['level']}: {level_name}")
        print(f"  Spec: {spec_key.height}x{spec_key.width}, {spec_key.num_robots} robots")
        print(f"  Optimal length: {level['min_optimal_length']}-{level['max_optimal_length']}")
        print(f"  Robots moved: {level['min_robots_moved']}-{level['max_robots_moved']}")
        
        # Generate puzzles with criteria
        criteria = {
            'min_optimal_length': level['min_optimal_length'],
            'max_depth': level['max_optimal_length'],
            'max_optimal_length': level['max_optimal_length'],
            'min_robots_moved': level['min_robots_moved'],
            'max_robots_moved': level['max_robots_moved']
        }
        stats = generator.generate_puzzles_for_spec(spec_key, num_puzzles=puzzles_per_level, criteria=criteria)
        
        print(f"  Generated: {stats['generated']}")
        print(f"  Solved: {stats['solved']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Time: {stats['elapsed_time']:.1f}s")
        
        total_puzzles += stats['solved']
    
    # Final bank statistics
    bank_stats = bank.get_stats()
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total puzzles generated: {total_puzzles}")
    print("Bank statistics:")
    print(f"  Total puzzles: {bank_stats['total_puzzles']}")
    print(f"  Partitions: {bank_stats['partition_count']}")
    
    if bank_stats['optimal_length_stats']:
        ol_stats = bank_stats['optimal_length_stats']
        print(f"  Optimal length range: {ol_stats['min']}-{ol_stats['max']}")
        print(f"  Optimal length mean: {ol_stats['mean']:.1f}")
    
    if bank_stats['robots_moved_stats']:
        rm_stats = bank_stats['robots_moved_stats']
        print(f"  Robots moved range: {rm_stats['min']}-{rm_stats['max']}")
        print(f"  Robots moved mean: {rm_stats['mean']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Generate puzzle bank for curriculum levels")
    parser.add_argument("--config", help="Path to curriculum config JSON file (optional, uses default if not provided)")
    parser.add_argument("--bank_dir", default=str(ARTIFACTS_ROOT / "puzzle_bank"), help="Directory to store puzzle bank")
    parser.add_argument("--puzzles_per_level", type=int, default=1000, help="Number of puzzles per level")
    # parser.add_argument("--use_controller", action="store_true", help="Use band-first controller to hit targets per level")
    parser.add_argument("--target_per_level", type=int, default=1000, help="Controller target puzzles per level (available)")
    parser.add_argument("--max_puzzles_global", type=int, default=200000, help="Global cap on puzzles attempted/generated")
    parser.add_argument("--chunk_per_spec", type=int, default=200, help="Chunk size per spec per iteration for controller")
    parser.add_argument("--solver", choices=["bfs", "astar_zero", "astar_one"], default="bfs", 
                       help="Solver type: bfs (BFS), astar_zero (A* with zero heuristic), astar_one (A* with one heuristic)")
    parser.add_argument("--max_depth", type=int, default=10, help="Solver max depth")
    parser.add_argument("--max_nodes", type=int, default=100000, help="Solver max nodes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    solver_config = {
        "solver_type": args.solver,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes
    }
    
    generate_curriculum_bank(
        config_path=args.config,
        bank_dir=args.bank_dir,
        puzzles_per_level=args.puzzles_per_level,
        solver_config=solver_config,
        # use_controller=args.use_controller,
        use_controller=True,
        controller_target_per_level=args.target_per_level,
        controller_max_puzzles_global=args.max_puzzles_global,
        controller_chunk_per_spec=args.chunk_per_spec,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
