"""Incremental W&B persistence for an in-flight DRC probe series."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .collate_drc_probes import (
    BUCKETS,
    _bucket_slug,
    _file_sha256,
    build_probe_records,
)


STATE_FILENAME = ".drc_probe_wandb_state.json"


def build_wandb_payloads(
    run_dir: str | Path,
    rows: List[Dict[str, object]],
    *,
    control: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Convert probe records to the stable W&B metric namespace."""
    records = build_probe_records(run_dir, rows, control=control)
    payloads: List[Dict[str, object]] = []
    for row, record in zip(rows, records):
        payload: Dict[str, object] = {
            "probe/timesteps": int(record["timesteps"]),
            "probe/val_det_success": record["val_det_success"],
        }
        for representation in row["representations"]:
            for bucket in BUCKETS:
                slug = _bucket_slug(bucket)
                prefix = f"{representation}_{slug}"
                payload[f"probe/{representation}/{slug}"] = record[prefix]
                payload[f"probe/{representation}/{slug}_control"] = record[
                    f"{prefix}_control"
                ]
                payload[f"probe/{representation}/{slug}_gain"] = record[
                    f"{prefix}_gain"
                ]
        payloads.append(payload)
    return payloads


def artifact_due(
    last_timesteps: int,
    artifact_timesteps: int,
    interval: int,
) -> bool:
    if interval <= 0:
        return False
    if artifact_timesteps <= 0:
        return True
    return last_timesteps // interval > artifact_timesteps // interval


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ProbeWandbLogger:
    """Long-lived W&B writer for a dedicated probe-monitor run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        project: str,
        run_name: str,
        run_id: str,
        entity: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        artifact_every: int = 5_000_000,
    ) -> None:
        import wandb

        self.wandb = wandb
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / STATE_FILENAME
        self.run_id = run_id
        self.artifact_every = artifact_every
        state: Dict[str, object] = {}
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            state_run_id = state.get("run_id")
            if state_run_id not in (None, run_id):
                raise ValueError(
                    f"Probe W&B state belongs to run {state_run_id}, not {run_id}"
                )

        self.run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume="allow",
            name=run_name,
            job_type="probe-monitor",
            group=parent_run_id,
            dir=str(self.run_dir),
            config={
                "source_training_run_id": parent_run_id,
                "probe_artifact_every": artifact_every,
            },
        )
        self.run.define_metric("probe/timesteps")
        self.run.define_metric("probe/*", step_metric="probe/timesteps")

        remote_last = int(self.run.summary.get("probe/last_timesteps", 0) or 0)
        remote_artifact = int(
            self.run.summary.get("probe/artifact_timesteps", 0) or 0
        )
        self.last_timesteps = max(
            remote_last,
            int(state.get("last_timesteps", 0) or 0),
        )
        self.artifact_timesteps = max(
            remote_artifact,
            int(state.get("artifact_timesteps", 0) or 0),
        )
        self.artifact_name = str(state.get("artifact", "") or "")

    def _persist_state(self) -> None:
        _write_json_atomic(
            self.state_path,
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "last_timesteps": self.last_timesteps,
                "artifact_timesteps": self.artifact_timesteps,
                "artifact": self.artifact_name,
            },
        )

    def _log_media(
        self,
        rows: List[Dict[str, object]],
        *,
        control: Optional[Dict[str, object]],
    ) -> None:
        records = build_probe_records(self.run_dir, rows, control=control)
        table = self.wandb.Table(
            columns=list(records[0]),
            data=[[record[column] for column in records[0]] for record in records],
        )
        media: Dict[str, object] = {
            "probe/timesteps": self.last_timesteps,
            "probe/trajectory_table": table,
        }
        chart_path = self.run_dir / "drc_probe_curve.png"
        if chart_path.exists():
            media["probe/trajectory_chart"] = self.wandb.Image(str(chart_path))
        self.run.log(media)

    def _log_artifact(self, rows: List[Dict[str, object]]) -> None:
        manifest_path = self.run_dir / "drc_probe_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifact = self.wandb.Artifact(
            f"{self.run_dir.name}-probes",
            type="probe-results",
            description=(
                "Fixed-dataset zero-state DRC landing-cell probes, random-init "
                "control, consolidated trajectory, and validation context."
            ),
            metadata={
                "checkpoint_count": manifest["checkpoint_count"],
                "first_timesteps": manifest["first_timesteps"],
                "last_timesteps": manifest["last_timesteps"],
                "dataset_sha256": manifest["dataset_sha256"],
                "manifest_sha256": _file_sha256(manifest_path),
            },
        )
        for name in (
            "drc_probe_series.csv",
            "drc_probe_manifest.json",
            "probe_dataset.npz",
            "probe_control.json",
            "drc_probe_summary.md",
            "drc_probe_curve.png",
            "curve.jsonl",
            "report.json",
            "config.json",
        ):
            path = self.run_dir / name
            if path.exists():
                artifact.add_file(str(path), name=name)
        for path in sorted(self.run_dir.glob("probe_t*.json")):
            artifact.add_file(str(path), name=f"checkpoints/{path.name}")

        alias = f"{self.last_timesteps // 1_000_000}m"
        logged = self.run.log_artifact(artifact, aliases=[alias, "latest"])
        logged.wait()
        self.artifact_timesteps = self.last_timesteps
        self.artifact_name = logged.qualified_name
        self.run.summary["probe/artifact_timesteps"] = self.artifact_timesteps
        self.run.summary["probe/artifact"] = self.artifact_name

    def sync(
        self,
        rows: List[Dict[str, object]],
        *,
        control: Optional[Dict[str, object]] = None,
        force_artifact: bool = False,
    ) -> int:
        pending_rows = [
            row for row in rows if int(row["timesteps"]) > self.last_timesteps
        ]
        for payload in build_wandb_payloads(
            self.run_dir,
            pending_rows,
            control=control,
        ):
            self.run.log(payload)
            self.last_timesteps = int(payload["probe/timesteps"])
            self.run.summary["probe/last_timesteps"] = self.last_timesteps
            self.run.summary["probe/checkpoint_count"] = len(
                [row for row in rows if int(row["timesteps"]) <= self.last_timesteps]
            )
            self._persist_state()

        if pending_rows:
            self._log_media(rows, control=control)
        if rows and (
            force_artifact
            or artifact_due(
                self.last_timesteps,
                self.artifact_timesteps,
                self.artifact_every,
            )
        ):
            self._log_artifact(rows)
            self._persist_state()
        return len(pending_rows)

    def finish(self) -> None:
        self.run.finish()
