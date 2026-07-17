"""Run or resume the bounded Phase 5 BC training-lever quality screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase2 import Phase2Config, run_phase2_benchmark  # noqa: E402


def _load_mapping(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def build_variant_configs(manifest_path: Path) -> tuple[Mapping[str, Any], Dict[str, Phase2Config]]:
    manifest = _load_mapping(manifest_path)
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported BC lever-screen schema")
    base_path = PROJECT_ROOT / manifest["base_config"]
    base = _load_mapping(base_path)
    common = dict(manifest.get("common_overrides", {}))
    variants = {}
    for name, overrides in manifest["variants"].items():
        values = {**base, **common, **dict(overrides or {})}
        variants[str(name)] = Phase2Config.from_mapping(values)
    return manifest, variants


def _quality_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    seed = result["seeds"][0]
    bc = seed["bc_pretrain"]
    selected = next(
        point for point in seed["curve"]
        if int(point["actual_timesteps"]) == int(seed["selected_checkpoint_timesteps"])
    )
    return {
        "bc_total_seconds": float(bc["total_seconds"]),
        "dataset_seconds": float(bc["dataset_seconds"]),
        "precision": bc["precision"],
        "selected_epoch": int(bc["selected_epoch"]),
        "best_validation_loss": float(bc["best_validation_loss"]),
        "best_validation_accuracy": float(bc["best_validation_accuracy"]),
        "validation_deterministic_success": float(selected["deterministic"]["success_rate"]),
        "validation_stochastic_success": float(selected["stochastic"]["success_rate"]),
        "parameter_count": int(seed["policy"]["parameter_count"]),
    }


def _reference_summary(path: Path, epochs: int) -> Dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    seed = result["seeds"][0]
    bc = seed["bc_pretrain"]
    history = bc["epochs"][:epochs]
    selected_index = min(
        range(len(history)),
        key=lambda index: (
            float(history[index]["validation_loss"]),
            -float(history[index]["validation_accuracy"]),
        ),
    )
    selected = history[selected_index]
    return {
        "source": str(path),
        "epochs_compared": epochs,
        "selected_epoch": selected_index + 1,
        "best_validation_loss": float(selected["validation_loss"]),
        "best_validation_accuracy": float(selected["validation_accuracy"]),
        "note": "launch-time canonical code; weights are parity-equivalent through these epochs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase5_bc_training_lever_screen.yaml",
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase4_bank",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest, variants = build_variant_configs(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, config in variants.items():
        output_dir = args.output_root / name
        result_path = output_dir / "results.json"
        if result_path.exists():
            print(f"[bc-lever] reusing complete {name}: {result_path}", flush=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            print(f"[bc-lever] starting {name}", flush=True)
            result = run_phase2_benchmark(config, bank_dir=args.bank_dir, output_dir=output_dir)
        summaries[name] = _quality_summary(result)
        (args.output_root / "screen_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "bounded five-epoch quality screen; not a recipe promotion",
                    "reference": _reference_summary(
                        PROJECT_ROOT / manifest["reference_result"],
                        next(iter(variants.values())).bc_pretrain_epochs,
                    ),
                    "variants": summaries,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {args.output_root / 'screen_summary.json'}")


if __name__ == "__main__":
    main()
