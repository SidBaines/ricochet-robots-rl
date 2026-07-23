"""Collate DRC landing probes with the task-learning curve.

Writes a Markdown table and a two-panel chart sharing the environment-step
axis. Optional W&B synchronization mirrors the P1 probe collator.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _steps_from_path(path: Path) -> int:
    match = re.search(r"(?:probe_t|model_)(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer timesteps from {path}")
    return int(match.group(1))


def load_probe_series(run_dir: str | Path) -> List[Dict[str, object]]:
    run_dir = Path(run_dir)
    rows = []
    dataset_hash = None
    for path in sorted(run_dir.glob("probe_t*.json"), key=_steps_from_path):
        row = json.loads(path.read_text())
        if row.get("schema_version") != 1 or row.get("probe_type") != "drc_landing_zero_state":
            raise ValueError(f"Unsupported probe schema: {path}")
        if row.get("random_init_control"):
            continue
        current_hash = row.get("dataset_sha256")
        if dataset_hash is None:
            dataset_hash = current_hash
        elif current_hash != dataset_hash:
            raise ValueError(f"Probe dataset changed at {path}")
        row["timesteps"] = int(row.get("timesteps") or _steps_from_path(path))
        rows.append(row)
    return rows


def load_task_curve(run_dir: str | Path) -> List[Dict[str, object]]:
    path = Path(run_dir) / "curve.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_at_or_before(curve: List[Dict[str, object]], timesteps: int) -> Optional[float]:
    candidates = [row for row in curve if int(row["timesteps"]) <= timesteps]
    if not candidates:
        return None
    return float(max(candidates, key=lambda row: int(row["timesteps"]))["val_det"]["success_rate"])


def _accuracy(row: Dict[str, object], representation: str, bucket: str) -> Optional[float]:
    result = row["representations"][representation]
    if bucket == "overall":
        return float(result["overall_acc"])
    value = result["by_slide_length"][bucket]["acc"]
    return None if value is None else float(value)


def write_outputs(
    run_dir: str | Path,
    rows: List[Dict[str, object]],
    *,
    control: Optional[Dict[str, object]] = None,
    p1_run_dir: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("No DRC probe rows to collate")
    run_dir = Path(run_dir)
    curve = load_task_curve(run_dir)
    representations = list(rows[0]["representations"])
    for row in rows:
        if list(row["representations"]) != representations:
            raise ValueError("Probe representation schema changed across checkpoints")

    columns = ["steps (M)", "val det"]
    for representation in representations:
        columns.extend([f"{representation} mid", f"{representation} long"])
    table = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        steps = int(row["timesteps"])
        val = _task_at_or_before(curve, steps)
        values = [f"{steps / 1e6:.1f}", "—" if val is None else f"{val:.3f}"]
        for representation in representations:
            for bucket in ("mid(4-7)", "long(8+)"):
                acc = _accuracy(row, representation, bucket)
                values.append("—" if acc is None else f"{acc:.3f}")
        table.append("| " + " | ".join(values) + " |")
    summary_path = run_dir / "drc_probe_summary.md"
    summary_path.write_text("\n".join(table) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (task_ax, probe_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    if curve:
        task_ax.plot(
            [int(row["timesteps"]) / 1e6 for row in curve],
            [float(row["val_det"]["success_rate"]) for row in curve],
            label="DRC validation",
            linewidth=2,
        )
    if p1_run_dir is not None:
        p1_curve = load_task_curve(p1_run_dir)
        if p1_curve:
            task_ax.plot(
                [int(row["timesteps"]) / 1e6 for row in p1_curve],
                [float(row["val_det"]["success_rate"]) for row in p1_curve],
                label="P1 feed-forward validation",
                color="black",
                alpha=0.65,
            )
    task_ax.set_ylabel("Deterministic success")
    task_ax.set_ylim(0, 1)
    task_ax.grid(alpha=0.25)
    task_ax.legend(loc="best")

    x = [int(row["timesteps"]) / 1e6 for row in rows]
    colors = plt.cm.viridis_r(
        [index / max(1, len(representations) - 1) for index in range(len(representations))]
    )
    for color, representation in zip(colors, representations):
        mid = [_accuracy(row, representation, "mid(4-7)") for row in rows]
        long = [_accuracy(row, representation, "long(8+)") for row in rows]
        probe_ax.plot(x, mid, color=color, label=f"{representation} mid")
        probe_ax.plot(x, long, color=color, linestyle="--", label=f"{representation} long")
        if control is not None and representation in control.get("representations", {}):
            control_mid = _accuracy(control, representation, "mid(4-7)")
            control_long = _accuracy(control, representation, "long(8+)")
            if control_mid is not None:
                probe_ax.axhline(control_mid, color=color, alpha=0.18)
            if control_long is not None:
                probe_ax.axhline(control_long, color=color, linestyle="--", alpha=0.18)
    probe_ax.set_xlabel("Environment steps (millions)")
    probe_ax.set_ylabel("Landing-cell accuracy")
    probe_ax.set_ylim(bottom=0)
    probe_ax.grid(alpha=0.25)
    probe_ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle("DRC task success and tick-wise landing-cell decodability")
    fig.tight_layout()
    chart_path = run_dir / "drc_probe_curve.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return summary_path, chart_path


def sync_wandb(rows: List[Dict[str, object]], *, project: str, run_name: str) -> None:
    import wandb

    run = wandb.init(project=project, name=run_name, job_type="probe-collation")
    for row in rows:
        step = int(row["timesteps"])
        payload = {"probe/timesteps": step}
        for representation in row["representations"]:
            payload[f"probe/{representation}/overall"] = _accuracy(
                row, representation, "overall"
            )
            payload[f"probe/{representation}/mid"] = _accuracy(
                row, representation, "mid(4-7)"
            )
            payload[f"probe/{representation}/long"] = _accuracy(
                row, representation, "long(8+)"
            )
        run.log(payload, step=step)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--p1-run-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_probe_series(run_dir)
    control_path = run_dir / "probe_control.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    summary, chart = write_outputs(
        run_dir,
        rows,
        control=control,
        p1_run_dir=args.p1_run_dir,
    )
    if args.wandb_project:
        sync_wandb(
            rows,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"{run_dir.name}-probes",
        )
    print(json.dumps({"summary": str(summary), "chart": str(chart), "rows": len(rows)}))


if __name__ == "__main__":
    main()
