"""Generate a solver-labelled puzzle bank parquet compatible with puzzles.load_puzzles.

Usage:
    .venv-phase2/bin/python -m src.groundup.build_bank \
        --height 12 --width 12 --num-robots 3 --edge-t 2 --central-l 2 \
        --attempts 60000 --max-depth 8 --max-nodes 300000 --workers 8 \
        --output artifacts/groundup_bank_12x12/size-12x12_numR3_ET2_CL2.parquet

Deterministic: attempt seeds are 0..attempts-1; rows are written in seed order,
so the same arguments always produce a byte-identical table (fingerprint
printed at the end). Reuses the vetted generator and BFS solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from src.env.puzzle_bank import (
    compute_layout_hash,
    count_robots_moved,
    serialize_layout_to_blob,
)
from src.env.ricochet_env import generate_board_from_spec_with_seed
from src.env.solver import solve_bfs_with_metadata

_ARGS = None  # set per worker via initializer


def _init_worker(args_dict):
    global _ARGS
    _ARGS = args_dict


def _attempt(seed: int):
    a = _ARGS
    board = generate_board_from_spec_with_seed(
        height=a["height"], width=a["width"], num_robots=a["num_robots"],
        edge_t_per_quadrant=a["edge_t"], central_l_per_quadrant=a["central_l"],
        seed=seed,
    )
    result = solve_bfs_with_metadata(board, max_depth=a["max_depth"], max_nodes=a["max_nodes"])
    if not result.found or result.depth < 1:
        return None
    actions = [int(rid) * 4 + int(direction) for rid, direction in result.actions]
    return {
        "seed": seed,
        "board_seed": seed,
        "optimal_length": int(result.depth),
        "robots_moved": int(count_robots_moved(actions)),
        "nodes_expanded": int(result.nodes_expanded),
        "actions_encoded": actions,
        "layout_hash": compute_layout_hash(board).hex(),
        "layout_blob": serialize_layout_to_blob(board).hex(),
        "solved_within_limits": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-robots", type=int, required=True)
    parser.add_argument("--edge-t", type=int, default=2)
    parser.add_argument("--central-l", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=60_000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    args_dict = dict(
        height=args.height, width=args.width, num_robots=args.num_robots,
        edge_t=args.edge_t, central_l=args.central_l,
        max_depth=args.max_depth, max_nodes=args.max_nodes,
    )
    with Pool(args.workers, initializer=_init_worker, initargs=(args_dict,)) as pool:
        rows = [r for r in pool.imap(_attempt, range(args.attempts), chunksize=64) if r]

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    fingerprint = hashlib.sha256(
        json.dumps(df[["seed", "optimal_length", "layout_hash"]].values.tolist()).encode()
    ).hexdigest()[:16]
    print(f"accepted {len(df)}/{args.attempts} puzzles -> {out}")
    print("by optimal_length:", df.groupby("optimal_length").size().to_dict())
    print("by robots_moved:", df.groupby("robots_moved").size().to_dict())
    print(f"fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
