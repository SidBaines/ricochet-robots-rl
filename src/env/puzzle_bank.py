"""
Puzzle Bank System for Curriculum Learning

This module provides a storage and retrieval system for precomputed Ricochet Robots
puzzles with their solution metadata. The bank stores puzzles in a lightweight format
using seeds and generation specs, with solution traces for deriving additional features.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any, Union
import numpy as np
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    # Fallback for environments without pyarrow
    import pandas as pd
    pa = None
    pq = None
from collections import defaultdict

from .ricochet_core import Board
from .ricochet_env import FixedLayout, generate_board_from_spec_with_seed


@dataclass(frozen=True)
class SpecKey:
    """Specification key that fully determines board generation given a seed."""
    height: int
    width: int
    num_robots: int
    edge_t_per_quadrant: int
    central_l_per_quadrant: int
    generator_rules_version: str = "1.0"
    
    def to_partition_key(self) -> str:
        """Generate partition key for storage organization."""
        return f"size-{self.height}x{self.width}_numR{self.num_robots}_ET{self.edge_t_per_quadrant}_CL{self.central_l_per_quadrant}_gen{self.generator_rules_version.replace('.', '')}"


@dataclass(frozen=True)
class SolverKey:
    """Solver configuration key for solution canonicalization."""
    algorithm: str = "BFS"
    neighbor_order_id: str = "canonical"  # canonical ordering for deterministic results
    max_depth: int = 50
    max_nodes: int = 100000
    solver_version: str = "1.0"
    
    def to_string(self) -> str:
        """String representation for indexing."""
        return f"{self.algorithm}_{self.neighbor_order_id}_d{self.max_depth}_n{self.max_nodes}_v{self.solver_version.replace('.', '')}"


@dataclass
class PuzzleMetadata:
    """Metadata for a single puzzle in the bank."""
    # Core identifiers
    seed: int
    spec_key: SpecKey
    solver_key: SolverKey
    board_hash: bytes  # blake3 hash of full board state (including target_robot)
    
    # Solution metrics
    optimal_length: int
    robots_moved: int  # count of distinct robot IDs in optimal plan
    nodes_expanded: int
    solve_time_ms: int
    
    # Flags
    solved_within_limits: bool
    hit_depth_limit: bool
    hit_node_limit: bool
    
    # Plan trace (for deriving additional features)
    actions_encoded: List[int]  # action = robot_id * 4 + direction

    # Metadata
    created_at: float
    run_id: str

    # Extended reproducibility
    board_seed: Optional[int] = None  # explicit seed used to generate this board
    layout_hash: Optional[bytes] = None  # hash of layout excluding target_robot
    
    # Optional layout blob (for generator drift recovery)
    layout_blob: Optional[bytes] = None
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Parquet storage."""
        data = asdict(self)
        # Convert SpecKey and SolverKey to JSON strings for proper storage
        data['spec_key'] = json.dumps(asdict(self.spec_key))
        data['solver_key'] = json.dumps(asdict(self.solver_key))
        # Convert bytes to hex for JSON serialization
        data['board_hash'] = self.board_hash.hex()
        if data.get('layout_hash') is not None:
            data['layout_hash'] = self.layout_hash.hex() if isinstance(self.layout_hash, (bytes, bytearray)) else data['layout_hash']
        if self.layout_blob is not None:
            data['layout_blob'] = self.layout_blob.hex()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PuzzleMetadata:
        """Create from dictionary loaded from Parquet."""
        # Convert back from hex - handle potential NaN/None values
        board_hash = data.get('board_hash')
        if board_hash and isinstance(board_hash, str) and board_hash.strip() and board_hash != 'nan':
            try:
                data['board_hash'] = bytes.fromhex(board_hash)
            except ValueError:
                raise ValueError(f"Invalid board_hash hex string: {board_hash}")
        else:
            raise ValueError("board_hash is required and cannot be empty")
        
        # Handle layout_blob - may be None, NaN, or empty string
        layout_blob = data.get('layout_blob')
        if (layout_blob and isinstance(layout_blob, str) and 
            layout_blob.strip() and layout_blob != 'nan'):
            try:
                data['layout_blob'] = bytes.fromhex(layout_blob)
            except ValueError:
                # Invalid hex string, treat as None
                data['layout_blob'] = None
        else:
            data['layout_blob'] = None

        # Optional fields: board_seed, layout_hash
        if 'board_seed' in data and data['board_seed'] == '':
            data['board_seed'] = None
        if 'layout_hash' in data:
            lh = data.get('layout_hash')
            if lh and isinstance(lh, str) and lh.strip() and lh != 'nan':
                try:
                    data['layout_hash'] = bytes.fromhex(lh)
                except ValueError:
                    data['layout_hash'] = None
            else:
                data['layout_hash'] = None
        
        # Reconstruct SpecKey and SolverKey - handle both dict and string formats
        import ast
        spec_key_data = data['spec_key']
        if isinstance(spec_key_data, dict):
            # Already a dictionary, use as-is
            pass
        elif isinstance(spec_key_data, str):
            try:
                # Try JSON first (for properly formatted JSON strings)
                spec_key_data = json.loads(spec_key_data)
            except json.JSONDecodeError:
                try:
                    # Fallback to ast.literal_eval for Python dict string representation
                    spec_key_data = ast.literal_eval(spec_key_data)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid spec_key format: {spec_key_data}") from e
        else:
            raise ValueError(f"spec_key must be dict or string, got {type(spec_key_data)}")
        data['spec_key'] = SpecKey(**spec_key_data)
        
        solver_key_data = data['solver_key']
        if isinstance(solver_key_data, dict):
            # Already a dictionary, use as-is
            pass
        elif isinstance(solver_key_data, str):
            try:
                # Try JSON first (for properly formatted JSON strings)
                solver_key_data = json.loads(solver_key_data)
            except json.JSONDecodeError:
                try:
                    # Fallback to ast.literal_eval for Python dict string representation
                    solver_key_data = ast.literal_eval(solver_key_data)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid solver_key format: {solver_key_data}") from e
        else:
            raise ValueError(f"solver_key must be dict or string, got {type(solver_key_data)}")
        data['solver_key'] = SolverKey(**solver_key_data)
        
        return cls(**data)


class PuzzleBank:
    """Storage and retrieval system for precomputed puzzles."""
    
    def __init__(self, bank_dir: Union[str, Path], run_id: Optional[str] = None):
        """Initialize puzzle bank.
        
        Args:
            bank_dir: Directory to store the puzzle bank
            run_id: Unique identifier for this computation run
        """
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{int(time.time())}"
        # In-memory cache of partition DataFrames keyed by partition path
        # This avoids re-reading Parquet/CSV on every query during training
        self._partition_df_cache: Dict[Path, "pd.DataFrame"] = {}
        
        # Create manifest file for tracking partitions
        self.manifest_path = self.bank_dir / "manifest.json"
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load or create manifest file."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
            # Backfill new fields for schema v1.1
            if "version" not in self.manifest or self.manifest.get("version") in {"1.0", 1.0}:
                # Upgrade to 1.1 while preserving existing data
                self.manifest["version"] = "1.1"
            # Ensure partitions dict exists
            self.manifest.setdefault("partitions", {})
            # Ensure new per-partition fields
            for part_key, part_info in self.manifest["partitions"].items():
                part_info.setdefault("histogram", {
                    "by_optimal_length_and_robots_moved": {}
                })
                part_info.setdefault("fingerprints", {})
                part_info.setdefault("dedup", {
                    "method": "none",
                    "bloom": None
                })
        else:
            self.manifest = {
                "version": "1.1",
                "partitions": {},
                "created_at": time.time()
            }
            self._save_manifest()
    
    def _save_manifest(self) -> None:
        """Save manifest file."""
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _get_partition_path(self, spec_key: SpecKey) -> Path:
        """Get path for a partition."""
        partition_key = spec_key.to_partition_key()
        return self.bank_dir / f"{partition_key}.parquet"
    
    def _load_partition_dataframe(self, partition_path: Path) -> "pd.DataFrame":
        """Load a partition into a pandas DataFrame, using an in-memory cache."""
        # Fast path: cached
        if partition_path in self._partition_df_cache:
            return self._partition_df_cache[partition_path]
        # Load from disk
        if not partition_path.exists():
            df = pd.DataFrame([])
            self._partition_df_cache[partition_path] = df
            return df
        if pa is not None and pq is not None:
            table = pq.read_table(partition_path)
            df = table.to_pandas()
        else:
            df = pd.read_csv(partition_path)
        # Clean NaNs to consistent blanks for string fields
        df = df.fillna('')
        self._partition_df_cache[partition_path] = df
        return df
    
    def get_partition_dataframe(self, spec_key: SpecKey) -> "pd.DataFrame":
        """Public helper to access a cached DataFrame for a spec partition."""
        partition_path = self._get_partition_path(spec_key)
        return self._load_partition_dataframe(partition_path)
    
    def add_puzzles(self, puzzles: List[PuzzleMetadata]) -> None:
        """Add puzzles to the bank.
        
        Args:
            puzzles: List of puzzle metadata to add
        """
        if not puzzles:
            return
        
        # Group by spec_key for partitioning
        by_spec = defaultdict(list)
        for puzzle in puzzles:
            by_spec[puzzle.spec_key].append(puzzle)
        
        for spec_key, spec_puzzles in by_spec.items():
            partition_path = self._get_partition_path(spec_key)
            
            # Convert to DataFrame
            data = [puzzle.to_dict() for puzzle in spec_puzzles]
            df = pd.DataFrame(data)
            
            if pa is not None and pq is not None:
                # Create Arrow table
                table = pa.Table.from_pandas(df)
                
                # Write to Parquet (append if exists)
                if partition_path.exists():
                    existing_table = pq.read_table(partition_path)
                    combined_table = pa.concat_tables([existing_table, table])
                    pq.write_table(combined_table, partition_path)
                else:
                    pq.write_table(table, partition_path)
            else:
                # Fallback to CSV if pyarrow not available
                if partition_path.exists():
                    existing_df = pd.read_csv(partition_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(partition_path, index=False)
                else:
                    df.to_csv(partition_path, index=False)
            
            # Update manifest
            partition_key = spec_key.to_partition_key()
            if partition_key not in self.manifest["partitions"]:
                self.manifest["partitions"][partition_key] = {
                    "spec_key": asdict(spec_key),
                    "created_at": time.time(),
                    "puzzle_count": 0,
                    "histogram": {
                        "by_optimal_length_and_robots_moved": {}
                    },
                    "fingerprints": {},
                    "dedup": {
                        "method": "none",
                        "bloom": None
                    }
                }
            self.manifest["partitions"][partition_key]["puzzle_count"] += len(spec_puzzles)
            self.manifest["partitions"][partition_key]["last_updated"] = time.time()

            # Increment histograms for (optimal_length, robots_moved)
            hist = self.manifest["partitions"][partition_key]["histogram"]["by_optimal_length_and_robots_moved"]
            for puzzle in spec_puzzles:
                ol = int(getattr(puzzle, "optimal_length", -1))
                rm = int(getattr(puzzle, "robots_moved", -1))
                key = f"ol={ol}|rm={rm}"
                hist[key] = int(hist.get(key, 0)) + 1
        
        self._save_manifest()
    
    def query_puzzles(
        self,
        spec_key: Optional[SpecKey] = None,
        min_optimal_length: Optional[int] = None,
        max_optimal_length: Optional[int] = None,
        min_robots_moved: Optional[int] = None,
        max_robots_moved: Optional[int] = None,
        limit: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> Iterator[PuzzleMetadata]:
        """Query puzzles matching criteria.
        
        Args:
            spec_key: Filter by generation spec
            min_optimal_length: Minimum optimal solution length
            max_optimal_length: Maximum optimal solution length
            min_robots_moved: Minimum number of robots moved
            max_robots_moved: Maximum number of robots moved
            limit: Maximum number of results
            random_seed: Random seed for reproducible sampling (None for random)
            
        Yields:
            PuzzleMetadata matching criteria
        """
        # Determine which partitions to search
        if spec_key is not None:
            partition_paths = [self._get_partition_path(spec_key)]
        else:
            partition_paths = [
                self.bank_dir / f"{partition_key}.parquet"
                for partition_key in self.manifest["partitions"].keys()
            ]
        
        count = 0
        for partition_path in partition_paths:
            # Load via cache (empty DataFrame if not present)
            df = self._load_partition_dataframe(partition_path)
            
            # Apply filters
            if len(df) == 0:
                continue
            mask = pd.Series([True] * len(df))
            
            if min_optimal_length is not None:
                mask &= df['optimal_length'] >= min_optimal_length
            if max_optimal_length is not None:
                mask &= df['optimal_length'] <= max_optimal_length
            if min_robots_moved is not None:
                mask &= df['robots_moved'] >= min_robots_moved
            if max_robots_moved is not None:
                mask &= df['robots_moved'] <= max_robots_moved
            
            # Convert filtered rows to PuzzleMetadata
            filtered_df = df[mask]
            
            # Shuffle the filtered data for random sampling
            if len(filtered_df) > 0:
                filtered_df = filtered_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
            
            for _, row in filtered_df.iterrows():
                if limit is not None and count >= limit:
                    return
                
                puzzle = PuzzleMetadata.from_dict(row.to_dict())
                yield puzzle
                count += 1
    
    def get_puzzle_count(self, spec_key: Optional[SpecKey] = None) -> int:
        """Get total number of puzzles in bank.
        
        Args:
            spec_key: If provided, count only for this spec
            
        Returns:
            Number of puzzles
        """
        if spec_key is not None:
            partition_key = spec_key.to_partition_key()
            return self.manifest["partitions"].get(partition_key, {}).get("puzzle_count", 0)
        else:
            return sum(
                partition.get("puzzle_count", 0)
                for partition in self.manifest["partitions"].values()
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bank statistics.
        
        Returns:
            Dictionary with bank statistics
        """
        total_puzzles = self.get_puzzle_count()
        partition_count = len(self.manifest["partitions"])
        
        # Get distribution stats
        all_puzzles = list(self.query_puzzles(limit=10000))  # Sample for stats
        if all_puzzles:
            optimal_lengths = [p.optimal_length for p in all_puzzles]
            robots_moved = [p.robots_moved for p in all_puzzles]
            
            stats = {
                "total_puzzles": total_puzzles,
                "partition_count": partition_count,
                "optimal_length_stats": {
                    "min": min(optimal_lengths),
                    "max": max(optimal_lengths),
                    "mean": np.mean(optimal_lengths),
                    "std": np.std(optimal_lengths)
                },
                "robots_moved_stats": {
                    "min": min(robots_moved),
                    "max": max(robots_moved),
                    "mean": np.mean(robots_moved),
                    "std": np.std(robots_moved)
                }
            }
        else:
            stats = {
                "total_puzzles": total_puzzles,
                "partition_count": partition_count,
                "optimal_length_stats": None,
                "robots_moved_stats": None
            }
        
        return stats


class BankSampler:
    """Sampler for retrieving puzzles from the bank during training."""
    
    def __init__(self, bank: PuzzleBank, sampling_strategy: str = "uniform"):
        """Initialize bank sampler.
        
        Args:
            bank: Puzzle bank to sample from
            sampling_strategy: Strategy for sampling ("uniform", "stratified")
        """
        self.bank = bank
        self.sampling_strategy = sampling_strategy
        # Cache of pre-filtered, shuffled views for fast per-episode sampling
        # key -> { 'df': DataFrame, 'order': np.ndarray[int], 'pos': int, 'rng': np.random.Generator }
        self._criteria_cache: Dict[Tuple[str, int, int, int, int], Dict[str, Any]] = {}
        self._cache_order: List[Tuple[str, int, int, int, int]] = []
        self._max_cache_entries: int = 128
        # Monitoring
        self._last_stratified_plan: Optional[Dict[str, Any]] = None

    def _criteria_key(
        self,
        spec_key: SpecKey,
        min_optimal_length: Optional[int],
        max_optimal_length: Optional[int],
        min_robots_moved: Optional[int],
        max_robots_moved: Optional[int],
    ) -> Tuple[str, int, int, int, int]:
        # Use partition string plus numeric bounds (None -> sentinel -1)
        part = spec_key.to_partition_key()
        def _n(v: Optional[int]) -> int:
            return -1 if v is None else int(v)
        return (part, _n(min_optimal_length), _n(max_optimal_length), _n(min_robots_moved), _n(max_robots_moved))

    def _get_or_build_iterator(
        self,
        spec_key: SpecKey,
        min_optimal_length: Optional[int],
        max_optimal_length: Optional[int],
        min_robots_moved: Optional[int],
        max_robots_moved: Optional[int],
        random_seed: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        key = self._criteria_key(spec_key, min_optimal_length, max_optimal_length, min_robots_moved, max_robots_moved)
        it = self._criteria_cache.get(key)
        if it is None:
            # Build filtered DataFrame from cached partition
            df = self.bank.get_partition_dataframe(spec_key)
            if len(df) == 0:
                return None
            mask = pd.Series([True] * len(df))
            if min_optimal_length is not None:
                mask &= df['optimal_length'] >= min_optimal_length
            if max_optimal_length is not None:
                mask &= df['optimal_length'] <= max_optimal_length
            if min_robots_moved is not None:
                mask &= df['robots_moved'] >= min_robots_moved
            if max_robots_moved is not None:
                mask &= df['robots_moved'] <= max_robots_moved
            filtered = df[mask]
            if len(filtered) == 0:
                return None
            # Initialize RNG and shuffled order
            rng = np.random.default_rng(random_seed)
            order = np.arange(len(filtered))
            rng.shuffle(order)
            it = {"df": filtered.reset_index(drop=True), "order": order, "pos": 0, "rng": rng}
            # Evict LRU if over capacity
            self._criteria_cache[key] = it
            self._cache_order.append(key)
            if len(self._criteria_cache) > self._max_cache_entries:
                old_key = self._cache_order.pop(0)
                if old_key in self._criteria_cache and old_key != key:
                    self._criteria_cache.pop(old_key, None)
        return it
    
    def _next_row(self, it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = it["df"]
        order = it["order"]
        pos = it["pos"]
        if pos >= len(order):
            # reshuffle and reset
            it["rng"].shuffle(order)
            pos = 0
        if len(order) == 0:
            return None
        idx = int(order[pos])
        it["pos"] = pos + 1
        row = df.iloc[idx]
        return row.to_dict()
    
    def sample_puzzle(
        self,
        spec_key: SpecKey,
        min_optimal_length: Optional[int] = None,
        max_optimal_length: Optional[int] = None,
        min_robots_moved: Optional[int] = None,
        max_robots_moved: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> Optional[PuzzleMetadata]:
        """Sample a single puzzle matching criteria using a cached iterator."""
        it = self._get_or_build_iterator(
            spec_key,
            min_optimal_length,
            max_optimal_length,
            min_robots_moved,
            max_robots_moved,
            random_seed,
        )
        if it is None:
            return None
        data = self._next_row(it)
        if data is None:
            return None
        return PuzzleMetadata.from_dict(data)
    
    def sample_puzzles(
        self,
        spec_key: SpecKey,
        count: int,
        min_optimal_length: Optional[int] = None,
        max_optimal_length: Optional[int] = None,
        min_robots_moved: Optional[int] = None,
        max_robots_moved: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> List[PuzzleMetadata]:
        """Sample multiple puzzles matching criteria using the cached iterator."""
        it = self._get_or_build_iterator(
            spec_key,
            min_optimal_length,
            max_optimal_length,
            min_robots_moved,
            max_robots_moved,
            random_seed,
        )
        if it is None:
            return []
        out: List[PuzzleMetadata] = []
        for _ in range(int(max(0, count))):
            data = self._next_row(it)
            if data is None:
                break
            out.append(PuzzleMetadata.from_dict(data))
        # Future: implement stratified strategy here
        return out

    def sample_puzzles_stratified(
        self,
        spec_key: SpecKey,
        total_count: int,
        bands: List[Dict[str, Any]],
        random_seed: Optional[int] = None,
        warn_on_empty_band: bool = False,
    ) -> List[PuzzleMetadata]:
        """Stratified sampling across bands defined by (ol, rm) ranges with weights.

        Args:
            spec_key: Partition to sample from
            total_count: Total number of puzzles requested
            bands: List of dicts with keys:
                - min_optimal_length, max_optimal_length
                - min_robots_moved, max_robots_moved
                - weight (float) optional; defaults to 1.0
            random_seed: Seed for deterministic allocation and iterators
            warn_on_empty_band: Emit warnings when a requested band has no samples

        Returns:
            List of PuzzleMetadata sampled approximately according to weights
        """
        if total_count <= 0 or len(bands) == 0:
            return []
        rng = np.random.default_rng(random_seed)
        weights = np.array([float(b.get("weight", 1.0)) for b in bands], dtype=float)
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            weights = np.ones(len(bands), dtype=float)
        weights = weights / weights.sum()

        # Initial integral allocation via floor, then distribute remainder by largest fractional parts
        raw_alloc = weights * int(total_count)
        base_alloc = np.floor(raw_alloc).astype(int)
        remainder = int(total_count) - int(base_alloc.sum())
        frac = raw_alloc - base_alloc
        if remainder > 0:
            frac = np.maximum(frac, 0.0)
            prob = frac.copy()
            total_frac = prob.sum()
            if total_frac <= 0:
                prob = weights.copy()
            if prob.sum() <= 0:  # fallback to uniform if weights also zero
                prob = np.ones_like(prob, dtype=float)
            prob = prob / prob.sum()
            choices = rng.choice(len(bands), size=remainder, replace=True, p=prob)
            for idx in choices:
                base_alloc[int(idx)] += 1

        band_configs: List[Dict[str, Any]] = []
        band_seeds: List[int] = []
        for band in bands:
            band_configs.append({
                "min_optimal_length": band.get("min_optimal_length"),
                "max_optimal_length": band.get("max_optimal_length"),
                "min_robots_moved": band.get("min_robots_moved"),
                "max_robots_moved": band.get("max_robots_moved"),
                "weight": float(band.get("weight", 1.0)),
            })
            # Precompute the iterator seed so fallback sampling remains deterministic
            if random_seed is None:
                band_seeds.append(int(rng.integers(0, 2**31 - 1)))
            else:
                band_seeds.append(int(random_seed))

        plan_details: List[Dict[str, Any]] = []
        out: List[PuzzleMetadata] = []
        actually_sampled = 0
        band_iters: List[Optional[Dict[str, Any]]] = [None] * len(band_configs)
        allocations: List[int] = []
        attempted: List[bool] = [False] * len(band_configs)

        for i, band_cfg in enumerate(band_configs):
            req = int(base_alloc[i])
            allocations.append(req)
            plan_entry: Dict[str, Any] = {
                "band": {
                    "min_optimal_length": band_cfg["min_optimal_length"],
                    "max_optimal_length": band_cfg["max_optimal_length"],
                    "min_robots_moved": band_cfg["min_robots_moved"],
                    "max_robots_moved": band_cfg["max_robots_moved"],
                    "weight": band_cfg["weight"],
                },
                "requested": req,
                "sampled": 0,
            }
            plan_details.append(plan_entry)
            if req <= 0:
                continue

            attempted[i] = True
            it = self._get_or_build_iterator(
                spec_key,
                band_cfg["min_optimal_length"],
                band_cfg["max_optimal_length"],
                band_cfg["min_robots_moved"],
                band_cfg["max_robots_moved"],
                band_seeds[i],
            )
            band_iters[i] = it
            band_samples: List[PuzzleMetadata] = []
            if it is not None:
                for _ in range(req):
                    data = self._next_row(it)
                    if data is None:
                        break
                    band_samples.append(PuzzleMetadata.from_dict(data))
            plan_entry["sampled"] = len(band_samples)
            out.extend(band_samples)
            actually_sampled += len(band_samples)
            shortfall = max(0, req - len(band_samples))
            if shortfall > 0:
                plan_entry["shortfall"] = shortfall

        # Reallocate leftover to bands with remaining availability
        leftover = int(total_count) - int(actually_sampled)
        if leftover > 0:
            indices = rng.permutation(len(band_iters))
            for idx in indices:
                if leftover <= 0:
                    break
                if band_iters[idx] is None and not attempted[idx]:
                    attempted[idx] = True
                    it = self._get_or_build_iterator(
                        spec_key,
                        band_configs[idx]["min_optimal_length"],
                        band_configs[idx]["max_optimal_length"],
                        band_configs[idx]["min_robots_moved"],
                        band_configs[idx]["max_robots_moved"],
                        band_seeds[idx],
                    )
                    band_iters[idx] = it
                it = band_iters[idx]
                if it is None:
                    continue
                taken = 0
                while leftover > 0:
                    data = self._next_row(it)
                    if data is None:
                        break
                    out.append(PuzzleMetadata.from_dict(data))
                    actually_sampled += 1
                    leftover -= 1
                    taken += 1
                if taken > 0:
                    plan_details[idx]["extra_sampled"] = int(plan_details[idx].get("extra_sampled", 0) + taken)

        if warn_on_empty_band:
            import warnings

            for i, plan_entry in enumerate(plan_details):
                if allocations[i] > 0 and plan_entry.get("sampled", 0) == 0:
                    warnings.warn(
                        (
                            "No puzzles available for stratified band "
                            f"{plan_entry['band']}; redistributed allocation to other bands."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Monitoring snapshot for external logging
        self._last_stratified_plan = {
            "total_requested": int(total_count),
            "total_sampled": int(actually_sampled),
            "bands": plan_details,
        }
        if out:
            order = rng.permutation(len(out))
            out = [out[int(i)] for i in order]
        return out

    def get_last_stratified_plan(self) -> Optional[Dict[str, Any]]:
        """Return details of the last stratified sampling plan and outcomes."""
        return self._last_stratified_plan


def compute_board_hash(board: Board) -> bytes:
    """Compute deterministic binary hash of a board state.
    
    This avoids JSON serialization entirely and instead hashes a compact
    binary representation composed of geometry, walls, robots, goal and target.
    """
    buf = bytearray()
    # Dimensions
    buf += struct.pack('<II', int(board.height), int(board.width))
    # Walls (raw bytes, deterministic layout)
    buf += board.h_walls.tobytes()
    buf += board.v_walls.tobytes()
    # Robots: sorted by robot id
    for rid in sorted(board.robot_positions.keys()):
        r, c = board.robot_positions[rid]
        buf += struct.pack('<III', int(rid), int(r), int(c))
    # Goal and target
    gr, gc = board.goal_position
    buf += struct.pack('<III', int(board.target_robot), int(gr), int(gc))
    # Compute hash (use SHA-256 as fallback if blake3 not available)
    try:
        return hashlib.blake3(buf).digest()
    except AttributeError:
        return hashlib.sha256(buf).digest()


def compute_layout_hash(board: Board) -> bytes:
    """Compute deterministic binary hash of the board layout excluding target_robot.

    Includes dimensions, walls, robots positions, and goal position, but excludes
    which robot is target to allow multi-round episodes on same layout.
    """
    buf = bytearray()
    buf += struct.pack('<II', int(board.height), int(board.width))
    buf += board.h_walls.tobytes()
    buf += board.v_walls.tobytes()
    for rid in sorted(board.robot_positions.keys()):
        r, c = board.robot_positions[rid]
        buf += struct.pack('<III', int(rid), int(r), int(c))
    gr, gc = board.goal_position
    buf += struct.pack('<II', int(gr), int(gc))
    try:
        return hashlib.blake3(buf).digest()
    except AttributeError:
        return hashlib.sha256(buf).digest()


def count_robots_moved(actions_encoded: List[int]) -> int:
    """Count number of distinct robots moved in action sequence.
    
    Args:
        actions_encoded: List of encoded actions (robot_id * 4 + direction)
        
    Returns:
        Number of distinct robots moved
    """
    robot_ids = set()
    for action in actions_encoded:
        robot_id = action // 4
        robot_ids.add(robot_id)
    return len(robot_ids)


def create_fixed_layout_from_metadata(metadata: PuzzleMetadata) -> FixedLayout:
    """Create FixedLayout from puzzle metadata.
    
    This requires the layout_blob to be present in the metadata.
    
    Args:
        metadata: Puzzle metadata with layout_blob
        
    Returns:
        FixedLayout for use with environment
        
    Raises:
        ValueError: If layout_blob is not available
    """
    if metadata.layout_blob is None:
        raise ValueError("Cannot create FixedLayout without layout_blob")
    return deserialize_layout_blob(metadata.layout_blob)


def create_fixed_layout_from_seed(metadata: PuzzleMetadata) -> FixedLayout:
    """Create FixedLayout by regenerating board from seed.
    
    Args:
        metadata: Puzzle metadata with seed and spec_key
        
    Returns:
        FixedLayout for use with environment
    """
    # Generate board directly from spec and seed without constructing an environment
    seed_to_use = int(metadata.board_seed) if getattr(metadata, 'board_seed', None) is not None else int(metadata.seed)
    board = generate_board_from_spec_with_seed(
        height=metadata.spec_key.height,
        width=metadata.spec_key.width,
        num_robots=metadata.spec_key.num_robots,
        edge_t_per_quadrant=metadata.spec_key.edge_t_per_quadrant,
        central_l_per_quadrant=metadata.spec_key.central_l_per_quadrant,
        seed=seed_to_use,
    )
    
    # Validate hash matches
    computed_hash = compute_board_hash(board)
    if computed_hash != metadata.board_hash:
        # If layout_hash is available, provide additional diagnostics and try layout-level check
        if getattr(metadata, 'layout_hash', None) is not None:
            comp_layout = compute_layout_hash(board)
            if comp_layout == metadata.layout_hash:
                raise ValueError(f"Board target mismatch for seed {seed_to_use}: layout matches but target_robot differs")
        raise ValueError(f"Board hash mismatch for seed {seed_to_use}")
    
    # Create FixedLayout
    return FixedLayout(
        height=board.height,
        width=board.width,
        h_walls=board.h_walls,
        v_walls=board.v_walls,
        robot_positions=board.robot_positions,
        goal_position=board.goal_position,
        target_robot=board.target_robot
    )


def serialize_layout_to_blob(board: Board) -> bytes:
    """Serialize a Board layout into a compact binary blob.

    Schema (little-endian):
    - uint32 height, uint32 width, uint32 num_robots
    - h_walls bytes ( (H+1)*W, bool packed as uint8 )
    - v_walls bytes ( H*(W+1), bool packed as uint8 )
    - for rid in sorted(robots): uint32 rid, uint32 r, uint32 c
    - uint32 goal_r, uint32 goal_c
    - uint32 target_robot
    """
    buf = bytearray()
    h = int(board.height)
    w = int(board.width)
    buf += struct.pack('<III', h, w, int(board.num_robots))
    buf += board.h_walls.astype(np.uint8).tobytes()
    buf += board.v_walls.astype(np.uint8).tobytes()
    for rid in sorted(board.robot_positions.keys()):
        r, c = board.robot_positions[rid]
        buf += struct.pack('<III', int(rid), int(r), int(c))
    gr, gc = board.goal_position
    buf += struct.pack('<II', int(gr), int(gc))
    buf += struct.pack('<I', int(board.target_robot))
    return bytes(buf)


def deserialize_layout_blob(blob: bytes) -> FixedLayout:
    """Deserialize a layout blob into a FixedLayout."""
    offset = 0
    def read(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, blob, offset)
        offset += size
        return vals
    h, w, nrobots = read('<III')
    hw_size = (h + 1) * w
    vw_size = h * (w + 1)
    h_walls = np.frombuffer(blob, dtype=np.uint8, count=hw_size, offset=offset).copy().astype(bool).reshape(h + 1, w)
    offset += hw_size
    v_walls = np.frombuffer(blob, dtype=np.uint8, count=vw_size, offset=offset).copy().astype(bool).reshape(h, w + 1)
    offset += vw_size
    robot_positions: Dict[int, Tuple[int, int]] = {}
    for _ in range(int(nrobots)):
        rid, r, c = read('<III')
        robot_positions[int(rid)] = (int(r), int(c))
    gr, gc = read('<II')
    (target_robot,) = read('<I')
    return FixedLayout(
        height=int(h),
        width=int(w),
        h_walls=h_walls,
        v_walls=v_walls,
        robot_positions=robot_positions,
        goal_position=(int(gr), int(gc)),
        target_robot=int(target_robot),
    )
