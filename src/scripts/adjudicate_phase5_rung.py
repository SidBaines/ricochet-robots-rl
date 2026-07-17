"""Apply predeclared Phase 5 halving rules and emit exact-resume configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase5_selection import (  # noqa: E402
    SuccessiveHalvingRules,
    adjudicate_successive_halving,
    promotion_config_mapping,
)


def _candidate_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=/path/to/results.json")
    return name, Path(path)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase5_allocation_rules.yaml",
    )
    parser.add_argument("--rung", type=int, required=True)
    parser.add_argument(
        "--candidate",
        type=_candidate_argument,
        action="append",
        required=True,
        help="repeat as NAME=/path/to/results.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(args.rules.read_text(encoding="utf-8")) or {}
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported allocation-rule schema")
    rungs = raw.get("rungs", {})
    rung = rungs.get(args.rung, rungs.get(str(args.rung)))
    if not isinstance(rung, dict):
        raise ValueError(f"No predeclared allocation rung at {args.rung}")
    candidates = {name: _load_json(path) for name, path in args.candidate}
    if len(candidates) != len(args.candidate):
        raise ValueError("Candidate names must be unique")
    rules = SuccessiveHalvingRules.from_mapping(raw.get("selection", {}))
    decision = adjudicate_successive_halving(
        candidates,
        rung_timesteps=args.rung,
        target_survivors=int(rung["target_survivors"]),
        rules=rules,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision["rules_path"] = str(args.rules)
    decision["promotion_checkpoints"] = [int(value) for value in rung["promotion_checkpoints"]]
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in decision["selected"]:
        promotion = promotion_config_mapping(
            candidates[name],
            candidate_name=name,
            rung_timesteps=args.rung,
            target_checkpoints=rung["promotion_checkpoints"],
            plateau=rung.get("plateau"),
        )
        (args.output_dir / f"promote_{name}.yaml").write_text(
            yaml.safe_dump(promotion, sort_keys=False),
            encoding="utf-8",
        )
    print(
        f"{decision['outcome']}: selected={decision['selected']} "
        f"eliminated={decision['eliminated']}"
    )
    if decision["ambiguity_reasons"]:
        print("Ambiguity: " + "; ".join(decision["ambiguity_reasons"]))


if __name__ == "__main__":
    main()
