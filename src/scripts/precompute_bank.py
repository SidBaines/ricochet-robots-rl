#!/usr/bin/env python3
"""
Precompute Puzzle Bank

This script generates and stores puzzles in the bank for curriculum learning.
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

from src.env.precompute_pipeline import run_precomputation, CurriculumSpecGenerator
from src.env.puzzle_bank import PuzzleBank, SolverKey


def main():
    parser = argparse.ArgumentParser(description="Precompute puzzle bank for curriculum learning")
    parser.add_argument("--bank_dir", type=str, default=str(ARTIFACTS_ROOT / "puzzle_bank"), 
                       help="Directory to store the puzzle bank")
    parser.add_argument("--num_puzzles", type=int, default=None,
                       help="Override number of puzzles per level (default: use curriculum specs)")
    parser.add_argument("--solver_depth", type=int, default=50,
                       help="Maximum solver depth")
    parser.add_argument("--solver_nodes", type=int, default=100000,
                       help="Maximum solver nodes")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be generated without actually doing it")
    
    args = parser.parse_args()
    
    # Create solver key
    solver_key = SolverKey(
        max_depth=args.solver_depth,
        max_nodes=args.solver_nodes
    )
    
    # Get curriculum specs
    curriculum_specs = CurriculumSpecGenerator.create_curriculum_specs()
    
    # Override num_puzzles if specified
    if args.num_puzzles is not None:
        for spec in curriculum_specs:
            spec["num_puzzles"] = args.num_puzzles
    
    if args.dry_run:
        print("Dry run - would generate:")
        total_puzzles = 0
        for spec in curriculum_specs:
            print(f"  Level {spec['level']}: {spec['name']}")
            print(f"    Spec: {spec['spec_key']}")
            print(f"    Criteria: {spec['min_optimal_length']}-{spec['max_optimal_length']} moves, "
                  f"{spec['min_robots_moved']}-{spec['max_robots_moved']} robots")
            print(f"    Puzzles: {spec['num_puzzles']}")
            total_puzzles += spec['num_puzzles']
        print(f"Total puzzles: {total_puzzles}")
        return
    
    # Run precomputation
    print(f"Starting precomputation...")
    print(f"Bank directory: {args.bank_dir}")
    print(f"Solver config: depth={solver_key.max_depth}, nodes={solver_key.max_nodes}")
    
    stats = run_precomputation(
        bank_dir=args.bank_dir,
        curriculum_specs=curriculum_specs,
        solver_key=solver_key,
        verbose=args.verbose
    )
    
    print(f"\nPrecomputation completed!")
    print(f"Generated {stats['total_solved']} puzzles across {len(curriculum_specs)} levels")
    print(f"Success rate: {stats['overall_success_rate']:.2%}")
    
    # Show bank stats
    bank = PuzzleBank(args.bank_dir)
    bank_stats = bank.get_stats()
    print(f"\nBank statistics:")
    print(f"  Total puzzles: {bank_stats['total_puzzles']}")
    print(f"  Partitions: {bank_stats['partition_count']}")
    
    if bank_stats['optimal_length_stats']:
        ol_stats = bank_stats['optimal_length_stats']
        print(f"  Optimal length: {ol_stats['min']}-{ol_stats['max']} "
              f"(mean: {ol_stats['mean']:.1f}, std: {ol_stats['std']:.1f})")
    
    if bank_stats['robots_moved_stats']:
        rm_stats = bank_stats['robots_moved_stats']
        print(f"  Robots moved: {rm_stats['min']}-{rm_stats['max']} "
              f"(mean: {rm_stats['mean']:.1f}, std: {rm_stats['std']:.1f})")


if __name__ == "__main__":
    main()
