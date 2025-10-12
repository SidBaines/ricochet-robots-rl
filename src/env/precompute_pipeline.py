"""
Precomputation Pipeline for Puzzle Bank

This module provides tools for generating and storing puzzles in the bank,
including solver integration and batch processing.
"""

from __future__ import annotations

import time
from pathlib import Path
ARTIFACTS_ROOT = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_BANK_DIR = ARTIFACTS_ROOT / "puzzle_bank"

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .puzzle_bank import (
    PuzzleBank, PuzzleMetadata, SpecKey, SolverKey, 
    compute_board_hash, compute_layout_hash, count_robots_moved, serialize_layout_to_blob
)
from .ricochet_env import RicochetRobotsEnv
from .solver import solve_bfs_with_metadata, solve_astar_with_metadata
from .curriculum_config import get_curriculum_specs_for_precomputation


class PuzzleGenerator:
    """Generates puzzles and their solutions for the bank."""
    
    def __init__(
        self,
        bank: PuzzleBank,
        solver_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """Initialize puzzle generator.
        
        Args:
            bank: Puzzle bank to store results
            solver_config: Solver configuration dictionary
            verbose: Whether to print progress
        """
        self.bank = bank
        self.solver_config = solver_config or {"solver_type": "bfs", "max_depth": 10, "max_nodes": 100000}
        self.verbose = verbose
    
    def generate_puzzles_for_spec(
        self,
        spec_key: SpecKey,
        num_puzzles: int,
        seed_range: Optional[Tuple[int, int]] = None,
        batch_size: int = 100,
        criteria: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Generate puzzles for a specific spec.
        
        Args:
            spec_key: Generation specification
            num_puzzles: Number of puzzles to generate
            seed_range: Optional (min, max) seed range
            batch_size: Number of puzzles to process in each batch
            
        Returns:
            Statistics about generation
        """
        if seed_range is None:
            seed_range = (0, 2**31 - 1)
        
        # Set criteria for filtering
        self.criteria = criteria
        
        start_time = time.time()
        generated = 0
        solved = 0
        failed = 0
        
        if self.verbose:
            print(f"Generating {num_puzzles} puzzles for spec {spec_key.to_partition_key()}")
        
        # Process in batches
        for batch_start in range(0, num_puzzles, batch_size):
            batch_end = min(batch_start + batch_size, num_puzzles)
            batch_size_actual = batch_end - batch_start
            
            # Generate batch
            batch_puzzles = []
            for i in range(batch_size_actual):
                seed = np.random.randint(seed_range[0], seed_range[1])
                
                try:
                    # criteria may be None when called by controller; guard accordingly
                    max_depth_override = None
                    if criteria is not None and isinstance(criteria, dict):
                        max_depth_override = criteria.get('solver_max_depth_override', criteria.get('max_depth', None))
                    puzzle = self._generate_single_puzzle(spec_key, seed, max_depth=max_depth_override)
                    if puzzle is not None:
                        batch_puzzles.append(puzzle)
                        solved += 1
                    else:
                        failed += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Error generating puzzle with seed {seed}: {e}")
                    failed += 1
                
                generated += 1
            
            # Store batch
            if batch_puzzles:
                self.bank.add_puzzles(batch_puzzles)
            
            if self.verbose:
                print(f"Batch {batch_start//batch_size + 1}: {len(batch_puzzles)} puzzles stored")
        
        elapsed = time.time() - start_time
        
        stats = {
            "spec_key": spec_key.to_partition_key(),
            "requested": num_puzzles,
            "generated": generated,
            "solved": solved,
            "failed": failed,
            "success_rate": solved / generated if generated > 0 else 0,
            "elapsed_time": elapsed,
            "puzzles_per_second": generated / elapsed if elapsed > 0 else 0
        }
        
        if self.verbose:
            print(f"Generation complete: {solved}/{generated} solved ({stats['success_rate']:.2%})")
            print(f"Time: {elapsed:.2f}s ({stats['puzzles_per_second']:.1f} puzzles/s)")
        
        return stats
    
    def _generate_single_puzzle(
        self,
        spec_key: SpecKey,
        seed: int,
        max_depth: Optional[int] = None,
    ) -> Optional[PuzzleMetadata]:
        """Generate a single puzzle and its solution.
        
        Args:
            spec_key: Generation specification
            seed: Random seed
            
        Returns:
            PuzzleMetadata if successful, None otherwise
        """
        # Create environment
        env = RicochetRobotsEnv(
            height=spec_key.height,
            width=spec_key.width,
            num_robots=spec_key.num_robots,
            edge_t_per_quadrant=spec_key.edge_t_per_quadrant,
            central_l_per_quadrant=spec_key.central_l_per_quadrant,
            ensure_solvable=False,  # We'll solve manually
            obs_mode="image"
        )
        
        # Generate board
        obs, info = env.reset(seed=seed)
        board = env.get_board()
        env.close()  # Clean up
        
        # Compute board hash
        board_hash = compute_board_hash(board)
        # Compute layout hash (target-agnostic) for multi-round reproducibility
        layout_hash = compute_layout_hash(board)
        # Serialize layout for robust replay
        layout_blob = serialize_layout_to_blob(board)
        # Extract the exact board_seed used by the env if exposed; fallback to used episode seed
        board_seed: Optional[int] = None
        try:
            info_dict = info if isinstance(info, dict) else {}
            bs = info_dict.get("board_seed")
            if bs is not None:
                board_seed = int(bs)
            else:
                es = info_dict.get("episode_seed")
                if es is not None:
                    board_seed = int(es)
        except Exception:
            board_seed = None
        
        # Solve puzzle
        solve_start = time.time()
        try:
            solver_type = self.solver_config.get("solver_type", "bfs")
            max_depth_val = max_depth or self.solver_config['max_depth']
            max_nodes_val = self.solver_config['max_nodes']
            
            if solver_type == "bfs":
                result = solve_bfs_with_metadata(board, max_depth=max_depth_val, max_nodes=max_nodes_val)
            elif solver_type == "astar_zero":
                result = solve_astar_with_metadata(board, max_depth=max_depth_val, max_nodes=max_nodes_val, h_mode="admissible_zero")
            elif solver_type == "astar_one":
                result = solve_astar_with_metadata(board, max_depth=max_depth_val, max_nodes=max_nodes_val, h_mode="admissible_one")
            else:
                raise ValueError(f"Unknown solver type: {solver_type}")
                
            solve_time = (time.time() - solve_start) * 1000  # Convert to ms
        except Exception as e:
            if self.verbose:
                print(f"  Solver exception: {e}")
            return None
        
        if result is None or not result.found:
            if self.verbose:
                print(f"  No solution found")
            return None
        
        # Extract solution metadata
        optimal_length = int(result.depth)
        if self.verbose:
            print(f"  Solution found: length={optimal_length}")
        
        # Normalize solution to encoded integer actions robot_id*4 + direction
        actions_encoded: List[int] = []
        for act in result.actions:
            if isinstance(act, (list, tuple)) and len(act) == 2:
                rid, direction = act
                actions_encoded.append(int(rid) * 4 + int(direction))
            elif isinstance(act, int):
                actions_encoded.append(int(act))
            else:
                # Unknown format; skip this puzzle
                if self.verbose:
                    print(f"  Unknown action format: {act}")
                return None
        robots_moved = count_robots_moved(actions_encoded)
        if self.verbose:
            print(f"  Robots moved: {robots_moved}")
        
        # Check if solution meets criteria (if provided)
        if hasattr(self, 'criteria') and self.criteria:
            if 'min_optimal_length' in self.criteria:
                if not (self.criteria['min_optimal_length'] <= optimal_length <= self.criteria['max_optimal_length']):
                    if self.verbose:
                        print(f"  Rejected: optimal_length {optimal_length} not in range [{self.criteria['min_optimal_length']}, {self.criteria['max_optimal_length']}]")
                    return None
                if not (self.criteria['min_robots_moved'] <= robots_moved <= self.criteria['max_robots_moved']):
                    if self.verbose:
                        print(f"  Rejected: robots_moved {robots_moved} not in range [{self.criteria['min_robots_moved']}, {self.criteria['max_robots_moved']}]")
                    return None
        
        # Determine flags
        solved_within_limits = True  # If we got here, it was solved
        hit_depth_limit = optimal_length >= self.solver_config['max_depth']
        hit_node_limit = result.nodes_expanded >= self.solver_config['max_nodes']
        
        # Map solver type to algorithm name
        solver_type = self.solver_config.get("solver_type", "bfs")
        if solver_type == "bfs":
            algorithm = "BFS"
        elif solver_type == "astar_zero":
            algorithm = "A*_zero"
        elif solver_type == "astar_one":
            algorithm = "A*_one"
        else:
            algorithm = "BFS"  # fallback
        
        # Create solver key with correct algorithm name
        solver_key = SolverKey(
            algorithm=algorithm,
            max_depth=self.solver_config['max_depth'],
            max_nodes=self.solver_config['max_nodes']
        )
        
        # Create metadata
        metadata = PuzzleMetadata(
            seed=seed,
            spec_key=spec_key,
            solver_key=solver_key,
            board_hash=board_hash,
            board_seed=board_seed,
            layout_hash=layout_hash,
            layout_blob=layout_blob,
            optimal_length=optimal_length,
            robots_moved=robots_moved,
            nodes_expanded=int(result.nodes_expanded),
            solve_time_ms=int(solve_time),
            solved_within_limits=solved_within_limits,
            hit_depth_limit=hit_depth_limit,
            hit_node_limit=hit_node_limit,
            actions_encoded=actions_encoded,
            created_at=time.time(),
            run_id=self.bank.run_id
        )
        
        return metadata


class CurriculumSpecGenerator:
    """Generates curriculum specifications for different difficulty levels."""
    
    @staticmethod
    def create_curriculum_specs() -> List[Dict[str, Any]]:
        """Create curriculum specifications.
        
        Returns:
            List of curriculum level specifications
        """
        from .curriculum_config import get_curriculum_specs_for_precomputation
        return get_curriculum_specs_for_precomputation()
    
    @staticmethod
    def create_band_keeping_specs() -> List[Dict[str, Any]]:
        """Create band-keeping specifications for balanced difficulty distribution.
        
        Returns:
            List of band specifications
        """
        # Define bands by optimal length and robots moved
        bands = []
        
        # Short puzzles, single robot
        for length in [1, 2, 3, 4, 5]:
            bands.append({
                "name": f"Short-{length}",
                "spec_key": SpecKey(
                    height=8, width=8, num_robots=1,
                    edge_t_per_quadrant=1, central_l_per_quadrant=1
                ),
                "min_optimal_length": length,
                "max_optimal_length": length,
                "min_robots_moved": 1,
                "max_robots_moved": 1,
                "target_fraction": 0.1  # 10% of training data
            })
        
        # Medium puzzles, single robot
        for length in [6, 7, 8, 9, 10]:
            bands.append({
                "name": f"Medium-{length}",
                "spec_key": SpecKey(
                    height=10, width=10, num_robots=1,
                    edge_t_per_quadrant=2, central_l_per_quadrant=2
                ),
                "min_optimal_length": length,
                "max_optimal_length": length,
                "min_robots_moved": 1,
                "max_robots_moved": 1,
                "target_fraction": 0.08  # 8% of training data
            })
        
        # Multi-robot puzzles
        for length in [4, 6, 8, 10, 12]:
            for robots in [2, 3]:
                bands.append({
                    "name": f"Multi-{robots}R-{length}",
                    "spec_key": SpecKey(
                        height=12, width=12, num_robots=robots,
                        edge_t_per_quadrant=2, central_l_per_quadrant=2
                    ),
                    "min_optimal_length": length,
                    "max_optimal_length": length,
                    "min_robots_moved": 1,
                    "max_robots_moved": robots,
                    "target_fraction": 0.05  # 5% of training data
                })
        
        return bands


def _speckey_dict_equals(a: dict, b: dict) -> bool:
    """Shallow equality for SpecKey dicts from manifest vs object-asdict."""
    return (
        a.get("height") == b.get("height") and
        a.get("width") == b.get("width") and
        a.get("num_robots") == b.get("num_robots") and
        a.get("edge_t_per_quadrant") == b.get("edge_t_per_quadrant") and
        a.get("central_l_per_quadrant") == b.get("central_l_per_quadrant") and
        a.get("generator_rules_version", "1.0") == b.get("generator_rules_version", "1.0")
    )


def compute_level_coverage_from_histograms(
    bank: PuzzleBank,
    curriculum_levels: "list[dict]",
) -> "dict[int, int]":
    """Compute how many puzzles are available per level using manifest histograms.

    Returns a map: level_index -> available_count satisfying that level's (optimal_length, robots_moved) bounds
    and matching the level's SpecKey.
    """
    coverage: dict[int, int] = {}
    partitions = bank.manifest.get("partitions", {})
    for level in curriculum_levels:
        level_idx = int(level["level"])
        sk_obj: SpecKey = level["spec_key"] if isinstance(level["spec_key"], SpecKey) else level["spec_key"]
        # Normalize to dict for comparison
        if isinstance(sk_obj, SpecKey):
            sk_dict = {
                "height": sk_obj.height,
                "width": sk_obj.width,
                "num_robots": sk_obj.num_robots,
                "edge_t_per_quadrant": sk_obj.edge_t_per_quadrant,
                "central_l_per_quadrant": sk_obj.central_l_per_quadrant,
                "generator_rules_version": sk_obj.generator_rules_version,
            }
        else:
            sk_dict = sk_obj

        min_ol = int(level["min_optimal_length"]) if level.get("min_optimal_length") is not None else None
        max_ol = int(level["max_optimal_length"]) if level.get("max_optimal_length") is not None else None
        min_rm = int(level["min_robots_moved"]) if level.get("min_robots_moved") is not None else None
        max_rm = int(level["max_robots_moved"]) if level.get("max_robots_moved") is not None else None

        count = 0
        for part_key, part in partitions.items():
            part_spec = part.get("spec_key", {})
            if not _speckey_dict_equals(part_spec, sk_dict):
                continue
            hist = (
                part.get("histogram", {})
                .get("by_optimal_length_and_robots_moved", {})
            )
            for k, v in hist.items():
                # k format: "ol={ol}|rm={rm}"
                try:
                    segs = dict(seg.split("=") for seg in k.split("|"))
                    ol = int(segs.get("ol", -1))
                    rm = int(segs.get("rm", -1))
                except Exception:
                    continue
                if (min_ol is not None and ol < min_ol) or (max_ol is not None and ol > max_ol):
                    continue
                if (min_rm is not None and rm < min_rm) or (max_rm is not None and rm > max_rm):
                    continue
                count += int(v)
        coverage[level_idx] = count
    return coverage


def run_band_first_precompute_controller(
    bank_dir: str,
    target_per_level: int = 1000,
    max_puzzles_global: int = 200000,
    chunk_puzzles_per_spec: int = 200,
    solver_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    curriculum_levels: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Outline controller: generate in chunks and stop when each level has enough coverage.

    This function uses manifest histograms to approximate available counts per level.
    It generates additional puzzles spec-by-spec in chunks until coverage thresholds
    are met or the global budget is exhausted.
    """
    if solver_config is None:
        solver_config = {"solver_type": "bfs", "max_depth": 10, "max_nodes": 100000}

    bank = PuzzleBank(bank_dir)
    generator = PuzzleGenerator(bank, solver_config=solver_config, verbose=False)
    levels = curriculum_levels if curriculum_levels is not None else get_curriculum_specs_for_precomputation()
    max_depth = max([level["max_optimal_length"] for level in levels])

    total_generated = 0
    loop_iter = 0
    while True:
        loop_iter += 1
        coverage = compute_level_coverage_from_histograms(bank, levels)
        if verbose:
            print(f"[controller] iteration {loop_iter} coverage: {coverage}")
        if coverage and all(c >= target_per_level for c in coverage.values()):
            if verbose:
                print("[controller] All levels meet target coverage. Stopping.")
            break
        elif coverage:
            remaining_levels = [level for level in levels if coverage[int(level["level"])] < target_per_level]
            max_depth = max([level["max_optimal_length"] for level in remaining_levels])
            if verbose:
                print(f"Tried to generate {total_generated} puzzles; total budget is {max_puzzles_global}")
                print(f"[controller] Remaining levels: {[level['level'] for level in remaining_levels]}")
                print(f"[controller] Max depth: {max_depth}")
        if total_generated >= max_puzzles_global:
            if verbose:
                print("[controller] Reached global max puzzle budget. Stopping.")
            break

        # Generate one small chunk per spec (round-robin)
        for spec in levels:
            if total_generated >= max_puzzles_global:
                break
            spec_key: SpecKey = spec["spec_key"]
            if verbose:
                print(f"[controller] Generating chunk for spec {spec_key.to_partition_key()} ({chunk_puzzles_per_spec})")
            stats = generator.generate_puzzles_for_spec(
                spec_key=spec_key,
                num_puzzles=chunk_puzzles_per_spec,
                criteria={
                    "solver_max_depth_override": max_depth,
                }
            )
            total_generated += stats.get("generated", 0)

    return {
        "total_generated": total_generated,
        "final_coverage": compute_level_coverage_from_histograms(bank, levels),
        "target_per_level": target_per_level,
        "max_puzzles_global": max_puzzles_global,
    }

def run_precomputation(
    bank_dir: str,
    curriculum_specs: Optional[List[Dict[str, Any]]] = None,
    solver_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run full precomputation pipeline.
    
    Args:
        bank_dir: Directory to store puzzle bank
        curriculum_specs: Curriculum specifications to generate
        solver_config: Solver configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Overall statistics
    """
    if curriculum_specs is None:
        curriculum_specs = CurriculumSpecGenerator.create_curriculum_specs()
    
    # Initialize bank and generator
    bank = PuzzleBank(bank_dir)
    generator = PuzzleGenerator(bank, solver_config, verbose)
    
    overall_stats = {
        "total_requested": 0,
        "total_generated": 0,
        "total_solved": 0,
        "total_failed": 0,
        "level_stats": []
    }
    
    if verbose:
        print(f"Starting precomputation for {len(curriculum_specs)} curriculum levels")
        print(f"Bank directory: {bank_dir}")
    
    # Generate puzzles for each level
    for spec in curriculum_specs:
        if verbose:
            print(f"\nGenerating Level {spec['level']}: {spec['name']}")
        
        level_stats = generator.generate_puzzles_for_spec(
            spec_key=spec["spec_key"],
            num_puzzles=spec["num_puzzles"]
        )
        
        overall_stats["total_requested"] += level_stats["requested"]
        overall_stats["total_generated"] += level_stats["generated"]
        overall_stats["total_solved"] += level_stats["solved"]
        overall_stats["total_failed"] += level_stats["failed"]
        overall_stats["level_stats"].append(level_stats)
    
    # Final statistics
    overall_stats["overall_success_rate"] = (
        overall_stats["total_solved"] / overall_stats["total_generated"]
        if overall_stats["total_generated"] > 0 else 0
    )
    
    if verbose:
        print(f"\nPrecomputation complete!")
        print(f"Total: {overall_stats['total_solved']}/{overall_stats['total_generated']} solved")
        print(f"Success rate: {overall_stats['overall_success_rate']:.2%}")
        print(f"Bank stats: {bank.get_stats()}")
    
    return overall_stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute puzzle bank")
    parser.add_argument("--bank_dir", type=str, default=str(DEFAULT_BANK_DIR), help="Bank directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    stats = run_precomputation(
        bank_dir=args.bank_dir,
        verbose=args.verbose
    )
    
    print(f"Precomputation completed with {stats['total_solved']} puzzles generated")
