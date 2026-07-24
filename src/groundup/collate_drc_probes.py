"""Collate DRC landing probes with the task-learning curve.

Writes a Markdown table and a two-panel chart sharing the environment-step
axis. Optional W&B synchronization mirrors the P1 probe collator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


BUCKETS = ("overall", "blocked(0)", "short(1-3)", "mid(4-7)", "long(8+)")


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


def _bucket_slug(bucket: str) -> str:
    return {
        "overall": "overall",
        "blocked(0)": "blocked",
        "short(1-3)": "short",
        "mid(4-7)": "mid",
        "long(8+)": "long",
    }[bucket]


def build_probe_records(
    run_dir: str | Path,
    rows: List[Dict[str, object]],
    *,
    control: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Flatten probe results into stable, machine-readable records."""
    curve = load_task_curve(run_dir)
    records: List[Dict[str, object]] = []
    for row in rows:
        steps = int(row["timesteps"])
        record: Dict[str, object] = {
            "timesteps": steps,
            "steps_m": steps / 1e6,
            "val_det_success": _task_at_or_before(curve, steps),
            "dataset_sha256": row["dataset_sha256"],
        }
        for representation in row["representations"]:
            for bucket in BUCKETS:
                slug = _bucket_slug(bucket)
                prefix = f"{representation}_{slug}"
                value = _accuracy(row, representation, bucket)
                control_value = (
                    _accuracy(control, representation, bucket)
                    if control is not None and representation in control.get("representations", {})
                    else None
                )
                record[prefix] = value
                record[f"{prefix}_control"] = control_value
                record[f"{prefix}_gain"] = None if value is None or control_value is None else value - control_value
        records.append(record)
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_machine_outputs(
    run_dir: str | Path,
    rows: List[Dict[str, object]],
    *,
    control: Optional[Dict[str, object]] = None,
) -> tuple[Path, Path]:
    """Write a flat CSV and checksum manifest for durable probe archiving."""
    if not rows:
        raise ValueError("No DRC probe rows to archive")
    run_dir = Path(run_dir)
    records = build_probe_records(run_dir, rows, control=control)
    fieldnames = list(records[0])
    csv_path = run_dir / "drc_probe_series.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)

    raw_probe_paths = sorted(run_dir.glob("probe_t*.json"), key=_steps_from_path)
    source_paths = [
        path
        for path in (
            run_dir / "probe_dataset.npz",
            run_dir / "probe_control.json",
            run_dir / "curve.jsonl",
            run_dir / "report.json",
        )
        if path.exists()
    ] + raw_probe_paths
    manifest = {
        "schema_version": 1,
        "probe_type": "drc_landing_zero_state",
        "checkpoint_count": len(rows),
        "first_timesteps": int(rows[0]["timesteps"]),
        "last_timesteps": int(rows[-1]["timesteps"]),
        "dataset_sha256": rows[0]["dataset_sha256"],
        "dataset_hash_count": len({row["dataset_sha256"] for row in rows}),
        "series_csv": {
            "path": csv_path.name,
            "sha256": _file_sha256(csv_path),
        },
        "source_files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in source_paths
        },
    }
    manifest_path = run_dir / "drc_probe_manifest.json"
    _write_text_atomic(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return csv_path, manifest_path


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
    colors = plt.cm.viridis_r([index / max(1, len(representations) - 1) for index in range(len(representations))])
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


def sync_wandb(
    run_dir: str | Path,
    rows: List[Dict[str, object]],
    *,
    project: str,
    run_name: str,
    control: Optional[Dict[str, object]] = None,
    entity: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_alias: str = "latest",
) -> Dict[str, object]:
    """Backfill probe metrics and a durable artifact into W&B.

    When ``run_id`` is supplied, append to that existing run without using
    environment timesteps as W&B's internal step. Probe charts use the custom
    ``probe/timesteps`` axis, so historical checkpoints can be added safely
    after the training run has finished.
    """
    import wandb

    run_dir = Path(run_dir)
    csv_path, manifest_path = write_machine_outputs(
        run_dir,
        rows,
        control=control,
    )
    records = build_probe_records(run_dir, rows, control=control)
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must" if run_id is not None else "allow",
        name=run_name,
        job_type="probe-collation",
        dir=str(run_dir),
    )
    run.define_metric("probe/timesteps")
    metric_names = set()
    payloads = []
    for row in rows:
        payload = {
            "probe/timesteps": int(row["timesteps"]),
            "probe/val_det_success": _task_at_or_before(
                load_task_curve(run_dir),
                int(row["timesteps"]),
            ),
        }
        for representation in row["representations"]:
            for bucket in BUCKETS:
                slug = _bucket_slug(bucket)
                value = _accuracy(row, representation, bucket)
                control_value = (
                    _accuracy(control, representation, bucket)
                    if control is not None and representation in control.get("representations", {})
                    else None
                )
                metric = f"probe/{representation}/{slug}"
                control_metric = f"probe/{representation}/{slug}_control"
                gain_metric = f"probe/{representation}/{slug}_gain"
                payload[metric] = value
                payload[control_metric] = control_value
                payload[gain_metric] = None if value is None or control_value is None else value - control_value
                metric_names.update((metric, control_metric, gain_metric))
        metric_names.add("probe/val_det_success")
        payloads.append(payload)
    for metric_name in sorted(metric_names):
        run.define_metric(metric_name, step_metric="probe/timesteps")
    for payload in payloads:
        # Do not pass W&B's internal step here. A resumed training run already
        # has history through 60M; the custom probe/timesteps axis preserves
        # the true historical x-coordinate without causing step regression.
        run.log(payload)

    table = wandb.Table(
        columns=list(records[0]),
        data=[[record[column] for column in records[0]] for record in records],
    )
    media = {"probe/trajectory_table": table}
    chart_path = run_dir / "drc_probe_curve.png"
    if chart_path.exists():
        media["probe/trajectory_chart"] = wandb.Image(str(chart_path))
    run.log(media)

    manifest = json.loads(manifest_path.read_text())
    artifact = wandb.Artifact(
        f"{run_dir.name}-probes",
        type="probe-results",
        description=(
            "Fixed-dataset zero-state DRC landing-cell probes, random-init "
            "control, consolidated trajectory, and 60M validation context."
        ),
        metadata={
            "checkpoint_count": manifest["checkpoint_count"],
            "first_timesteps": manifest["first_timesteps"],
            "last_timesteps": manifest["last_timesteps"],
            "dataset_sha256": manifest["dataset_sha256"],
            "manifest_sha256": _file_sha256(manifest_path),
        },
    )
    artifact.add_file(str(csv_path), name=csv_path.name)
    artifact.add_file(str(manifest_path), name=manifest_path.name)
    for name in (
        "probe_dataset.npz",
        "probe_control.json",
        "drc_probe_summary.md",
        "drc_probe_curve.png",
        "curve.jsonl",
        "report.json",
        "config.json",
    ):
        path = run_dir / name
        if path.exists():
            artifact.add_file(str(path), name=name)
    for path in sorted(run_dir.glob("probe_t*.json"), key=_steps_from_path):
        artifact.add_file(str(path), name=f"checkpoints/{path.name}")
    logged_artifact = run.log_artifact(
        artifact,
        aliases=[artifact_alias, "latest"] if artifact_alias != "latest" else ["latest"],
    )
    logged_artifact.wait()

    final = records[-1]
    run.summary["probe/checkpoint_count"] = len(records)
    run.summary["probe/first_timesteps"] = records[0]["timesteps"]
    run.summary["probe/last_timesteps"] = final["timesteps"]
    run.summary["probe/dataset_sha256"] = final["dataset_sha256"]
    run.summary["probe/artifact"] = logged_artifact.qualified_name
    run.summary["probe/manifest_sha256"] = _file_sha256(manifest_path)
    run_url = run.url
    artifact_name = logged_artifact.qualified_name
    run.finish()
    return {
        "run_url": run_url,
        "artifact": artifact_name,
        "rows": len(rows),
        "csv": str(csv_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--p1-run-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--artifact-alias", default="latest")
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
    csv_path, manifest_path = write_machine_outputs(
        run_dir,
        rows,
        control=control,
    )
    wandb_result = None
    if args.wandb_project:
        wandb_result = sync_wandb(
            run_dir,
            rows,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"{run_dir.name}-probes",
            control=control,
            entity=args.wandb_entity,
            run_id=args.wandb_run_id,
            artifact_alias=args.artifact_alias,
        )
    print(
        json.dumps(
            {
                "summary": str(summary),
                "chart": str(chart),
                "csv": str(csv_path),
                "manifest": str(manifest_path),
                "rows": len(rows),
                "wandb": wandb_result,
            }
        )
    )


if __name__ == "__main__":
    main()
