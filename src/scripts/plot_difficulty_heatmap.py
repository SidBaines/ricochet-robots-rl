"""Joint (optimal-length × robots-moved) success heatmaps.

Difficulty in Ricochet Robots is two-dimensional: how many moves the optimal
solution takes, and how many distinct robots that solution has to move. Many
cells are structurally impossible (e.g. robots_moved > optimal_length, or a
1-move solution can only move 1 robot), so the reachable region is a ragged
triangle rather than a full grid.

This renders success rate over that grid from per-episode records:
  - `--records`  : a results.json — uses the final-test records (always stored)
                   and, if per-checkpoint joint tables are present in the curve
                   (newer runs), a row of heatmaps across training checkpoints.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1a1a1a"
MUTED = "#8a8a8a"
IMPOSSIBLE = "#ededed"


def _joint_from_records(records):
    """records -> {(length, robots): (rate, n)} plus axis ranges."""
    cells = defaultdict(list)
    for rec in records:
        cells[(int(rec["optimal_length"]), int(rec["robots_moved"]))].append(bool(rec["success"]))
    lengths = sorted({length for length, _ in cells})
    robots = sorted({r for _, r in cells})
    table = {key: (float(np.mean(v)), len(v)) for key, v in cells.items()}
    return table, lengths, robots


def _draw_heatmap(ax, table, lengths, robots, title):
    """table: {(length, robots): (rate, n)}. Rows = robots, cols = length."""
    cmap = matplotlib.colormaps["Blues"]
    grid = np.full((len(robots), len(lengths)), np.nan)
    for (length, r), (rate, _n) in table.items():
        grid[robots.index(r), lengths.index(length)] = rate
    # Grey backdrop for impossible/absent cells, then paint present cells.
    ax.set_facecolor("white")
    for i in range(len(robots)):
        for j in range(len(lengths)):
            val = grid[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=IMPOSSIBLE, edgecolor="white", lw=1.5))
                continue
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(0.12 + 0.83 * val), edgecolor="white", lw=1.5))
            rate, n = table[(lengths[j], robots[i])]
            ax.text(j, i, f"{rate:.0%}\n(n={n})", ha="center", va="center", fontsize=8,
                    color="white" if val > 0.5 else INK)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(length) for length in lengths])
    ax.set_yticks(range(len(robots)))
    ax.set_yticklabels([str(r) for r in robots])
    ax.set_xlim(-0.5, len(lengths) - 0.5)
    ax.set_ylim(-0.5, len(robots) - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("optimal solution length", color=INK, fontsize=9)
    ax.set_ylabel("robots moved", color=INK, fontsize=9)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="a results.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args()

    result = json.loads(args.records.read_text(encoding="utf-8"))
    seed_result = result["seeds"][args.seed_index]
    args.out.mkdir(parents=True, exist_ok=True)

    # Always available: final held-out test records (stochastic + deterministic).
    panels = []
    for mode in ("stochastic", "deterministic"):
        recs = seed_result["test"][mode]["records"]
        table, lengths, robots = _joint_from_records(recs)
        panels.append((f"Final test — {mode}", table, lengths, robots))

    # If the curve carries per-checkpoint joint tables (newer runs), add a row.
    curve = seed_result.get("curve", [])
    checkpoint_joints = [(c["actual_timesteps"], c.get("stochastic", {}).get("joint")) for c in curve]
    has_across_training = any(joint for _, joint in checkpoint_joints)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, table, lengths, robots) in zip(axes, panels):
        _draw_heatmap(ax, table, lengths, robots, title)
    fig.suptitle(
        "Held-out success over the 2-D difficulty grid (grey = no such puzzle)",
        color=INK, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "final_test_difficulty_heatmap.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote final-test heatmap to {args.out}")

    if has_across_training:
        present = [(ts, joint) for ts, joint in checkpoint_joints if joint]
        fig, axes = plt.subplots(1, len(present), figsize=(5.0 * len(present), 4.8))
        if len(present) == 1:
            axes = [axes]
        all_len = sorted({int(length) for _, joint in present for length, _ in (k.split(",") for k in joint)})
        for ax, (ts, joint) in zip(axes, present):
            table = {}
            robots_set = set()
            for key, cell in joint.items():
                length, r = (int(x) for x in key.split(","))
                robots_set.add(r)
                table[(length, r)] = (cell["rate"], cell["n"])
            _draw_heatmap(ax, table, all_len, sorted(robots_set), f"{ts/1e6:.1f}M steps")
        fig.suptitle("Held-out stochastic success over difficulty, across training",
                     color=INK, fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(args.out / "difficulty_heatmap_across_training.png", dpi=150, facecolor="white")
        plt.close(fig)
        print(f"wrote across-training heatmap row to {args.out}")
    else:
        print("note: no per-checkpoint joint tables in this run — across-training row skipped "
              "(re-run with the joint-persistence harness to produce it)")


if __name__ == "__main__":
    main()
