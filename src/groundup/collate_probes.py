"""Collate landing-cell probe results into a chart, a table, and W&B.

Reads probe_t<steps>.json files from a run dir plus curve.jsonl (task
success), writes probe_curve.png and probe_summary.md, and optionally
appends new points to a dedicated W&B run (id stable across invocations, so
the probe daemon can call this after every checkpoint).

Usage:
    .venv-phase2/bin/python -m src.groundup.collate_probes \
        --run-dir artifacts/groundup/p1_12x12_planefree --wandb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BUCKETS = ["blocked(0)", "short(1-3)", "mid(4-7)", "long(8+)"]


def load_series(run_dir: Path):
    rows = []
    for path in sorted(run_dir.glob("probe_t*.json")):
        match = re.match(r"probe_t(\d+)\.json", path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "t": int(match.group(1)),
                "overall": data["overall_acc"],
                **{b: data["by_slide_length"][b]["acc"] for b in BUCKETS},
            }
        )
    rows.sort(key=lambda r: r["t"])
    control = None
    control_path = run_dir / "probe_control.json"
    if control_path.exists():
        c = json.loads(control_path.read_text())
        control = {"overall": c["overall_acc"], **{b: c["by_slide_length"][b]["acc"] for b in BUCKETS}}
    success = []
    curve = run_dir / "curve.jsonl"
    if curve.exists():
        for line in curve.read_text().splitlines():
            e = json.loads(line)
            success.append((e["timesteps"], e["val_det"]["success_rate"]))
    return rows, control, success


def write_outputs(run_dir: Path, rows, control, success):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    ts = [r["t"] / 1e6 for r in rows]
    for key, style in [("overall", "k-o"), ("blocked(0)", "-s"), ("short(1-3)", "-^"),
                       ("mid(4-7)", "-v"), ("long(8+)", "-d")]:
        ax1.plot(ts, [r[key] for r in rows], style, label=key, markersize=4)
    if control:
        ax1.axhline(control["overall"], color="gray", linestyle="--", label="random-init control (overall)")
    ax1.axhline(1 / 144, color="red", linestyle=":", label="chance (1/144)")
    ax1.set_xlabel("training steps (M)"); ax1.set_ylabel("exact landing-cell probe accuracy")
    ax1.set_title("Linear landing-cell probes by slide length"); ax1.legend(fontsize=8); ax1.set_ylim(0, None)

    if success:
        ax2.plot([t / 1e6 for t, _ in success], [s for _, s in success], "g-", label="val det success")
    if rows:
        ax2.plot(ts, [r["mid(4-7)"] for r in rows], "-v", label="probe mid(4-7)")
        ax2.plot(ts, [r["long(8+)"] for r in rows], "-d", label="probe long(8+)")
    ax2.set_xlabel("training steps (M)"); ax2.set_ylabel("rate")
    ax2.set_title("Task success vs long-range dynamics decodability"); ax2.legend(fontsize=8); ax2.set_ylim(0, 1)
    fig.savefig(run_dir / "probe_curve.png", dpi=140)
    plt.close(fig)

    lines = ["| steps (M) | overall | " + " | ".join(BUCKETS) + " |",
             "|---|---|" + "---|" * len(BUCKETS)]
    for r in rows:
        lines.append(
            f"| {r['t']/1e6:.0f} | {r['overall']:.3f} | "
            + " | ".join(f"{r[b]:.3f}" for b in BUCKETS) + " |"
        )
    if control:
        lines.append("| random-init control | " + f"{control['overall']:.3f} | "
                     + " | ".join(f"{control[b]:.3f}" for b in BUCKETS) + " |")
    (run_dir / "probe_summary.md").write_text("\n".join(lines) + "\n")


def sync_wandb(run_dir: Path, rows, project: str):
    import wandb

    state_path = run_dir / ".probe_wandb_state.json"
    last = json.loads(state_path.read_text())["last_t"] if state_path.exists() else -1
    new = [r for r in rows if r["t"] > last]
    if not new:
        return
    run = wandb.init(project=project, id=f"probes-{run_dir.name}", name=f"groundup-{run_dir.name}-probes",
                     resume="allow", dir=str(run_dir))
    run.define_metric("train/timesteps")
    run.define_metric("probe/*", step_metric="train/timesteps")
    for r in new:
        run.log({"train/timesteps": r["t"], "probe/overall": r["overall"],
                 **{f"probe/{b}": r[b] for b in BUCKETS}}, step=r["t"])
    run.finish()
    state_path.write_text(json.dumps({"last_t": rows[-1]["t"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", default="ricochet-robots-rl")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows, control, success = load_series(run_dir)
    if not rows:
        raise SystemExit("no probe_t*.json files found")
    write_outputs(run_dir, rows, control, success)
    if args.wandb:
        sync_wandb(run_dir, rows, args.project)
    print(f"collated {len(rows)} probe points -> {run_dir}/probe_curve.png, probe_summary.md")


if __name__ == "__main__":
    main()
