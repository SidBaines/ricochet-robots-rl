"""Machine-readable provenance for every training run."""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable

from src.env.environment_spec import EnvironmentSpec


TRACKED_PACKAGES = (
    "numpy",
    "gymnasium",
    "torch",
    "stable-baselines3",
    "sb3-contrib",
    "pandas",
    "pyarrow",
)


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions(packages: Iterable[str] | None = None) -> Dict[str, str]:
    if packages is None:
        return dict(
            sorted(
                (dist.metadata["Name"], dist.version) for dist in metadata.distributions() if dist.metadata.get("Name")
            )
        )
    result: Dict[str, str] = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def create_run_manifest(
    *,
    project_root: Path,
    requested_environment_spec: EnvironmentSpec,
    effective_environment_spec: EnvironmentSpec,
    model_type: str,
    policy: Any,
    policy_kwargs: Dict[str, Any],
    actual_model: Dict[str, Any],
    observation_space: Any,
    action_space: Any,
    initial_timesteps: int,
    requested_additional_timesteps: int,
    target_total_timesteps: int,
    seed: int | None,
    resolved_config: Dict[str, Any],
    bank_split: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create the initial manifest from resolved runtime objects."""
    status = _git_value(project_root, "status", "--porcelain")
    tracked_diff = _git_value(project_root, "diff", "HEAD") or ""
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_value(project_root, "rev-parse", "HEAD"),
            "branch": _git_value(project_root, "branch", "--show-current"),
            "dirty": bool(status) if status is not None else None,
            "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest(),
            "status": status.splitlines() if status else [],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _versions(),
        },
        "seed": seed,
        "seed_derivation": "SB3 make_vec_env uses base_seed + worker_rank; bank reset(seed) restarts each worker sampler",
        "requested_environment": requested_environment_spec.to_dict(),
        "environment": effective_environment_spec.to_dict(),
        "resolved_config": resolved_config,
        "model": {
            "type": model_type,
            "requested_policy": policy if isinstance(policy, str) else getattr(policy, "__name__", str(policy)),
            "requested_policy_kwargs": policy_kwargs,
            **actual_model,
        },
        "spaces": {
            "observation": repr(observation_space),
            "action": repr(action_space),
        },
        "bank_split": bank_split,
        "timesteps": {
            "initial": int(initial_timesteps),
            "requested_additional": int(requested_additional_timesteps),
            "target_total": int(target_total_timesteps),
            "final_total": None,
        },
    }


def write_run_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Atomically write a manifest so interrupted runs leave valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
