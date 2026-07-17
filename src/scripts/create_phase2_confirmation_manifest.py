"""Create a sealed confirmation suite from unused held-out test layouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase2 import Phase2Config, build_confirmation_manifest  # noqa: E402
from src.env.puzzle_bank import PuzzleBank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20_260_717)
    parser.add_argument("--max-per-stratum", type=int, default=20)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Phase 2 config must be a mapping")
    config = Phase2Config.from_mapping(raw)
    parent = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    manifest = build_confirmation_manifest(
        PuzzleBank(args.bank_dir),
        parent,
        args.output,
        config,
        selection_seed=args.selection_seed,
        max_per_stratum=args.max_per_stratum,
    )
    print(
        f"confirmation manifest: {len(manifest['selected_puzzle_ids'])} puzzles, "
        f"fingerprint={manifest['fingerprint']}"
    )


if __name__ == "__main__":
    main()
