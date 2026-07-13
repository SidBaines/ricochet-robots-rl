"""Plot held-out success by puzzle difficulty across training.

Reads a Phase 2 benchmark ``results.json`` (which stores, per checkpoint, the
held-out validation success stratified ``by_optimal_length`` and
``by_robots_moved``) and renders difficulty-vs-training curves plus the final
test breakdown. Difficulty is an *ordered* variable, so lengths/robots use a
sequential colormap (light = easy, dark = hard) rather than a categorical one.

Usage:
    python src/scripts/plot_training_by_difficulty.py \
        --results artifacts/phase2_experiments/phase4b_bc_16x16/results.json \
        --out artifacts/phase2_experiments/phase4b_bc_16x16/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"
RANDOM_C = "#b0b0b0"
GATE_C = "#c04a3b"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(-0.03, 1.03)


def _series_by(curve, mode, field):
    """Return {stratum_key: [success_rate per checkpoint]} and the x timesteps."""
    xs = [c["actual_timesteps"] for c in curve]
    keys = sorted(
        {k for c in curve for k in c[mode][f"by_{field}"]},
        key=lambda k: int(k),
    )
    series = {
        key: [c[mode][f"by_{field}"].get(key, {}).get("success_rate", np.nan) for c in curve]
        for key in keys
    }
    return xs, series


def _plot_by(ax, curve, mode, field, label, cmap_name, baselines):
    xs, series = _series_by(curve, mode, field)
    xs_m = [x / 1e6 for x in xs]
    cmap = matplotlib.colormaps[cmap_name]
    keys = list(series)
    # Sequential difficulty: harder = darker. viridis/plasma run dark->light,
    # so the hardest stratum takes the low (dark) end and the easiest the high
    # (light) end. Range kept off both extremes so every line reads on white.
    positions = np.linspace(0.82, 0.12, len(keys))
    end_labels = []  # (final_value, color, text) for de-overlapped right-edge labels
    for pos, key in zip(positions, keys):
        color = cmap(pos)
        ys = series[key]
        ax.plot(xs_m, ys, "-", color=color, linewidth=2, zorder=3)
        ax.plot(xs_m, ys, "o", color=color, markersize=4, zorder=4)
        last = next((v for v in reversed(ys) if not np.isnan(v)), np.nan)
        if not np.isnan(last):
            end_labels.append((last, color, f" {label}{key}"))
    # Greedy vertical spread so near-equal final values don't overprint.
    min_gap = 0.028
    end_labels.sort(key=lambda item: item[0])
    placed_y = []
    for value, _, _ in end_labels:
        y = value if not placed_y else max(value, placed_y[-1] + min_gap)
        placed_y.append(y)
    for (value, color, text), y in zip(end_labels, placed_y):
        ax.annotate(text, xy=(xs_m[-1], y), color=color, fontsize=8, va="center", fontweight="bold")
    for name, value, color in baselines:
        ax.axhline(value, color=color, linewidth=1.3, linestyle="--", zorder=2)
        ax.annotate(
            name,
            xy=(xs_m[0], value),
            xytext=(2, 3),
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        )
    _style(ax)
    ax.set_xlim(-0.3, xs_m[-1] + xs_m[-1] * 0.16)
    ax.set_xlabel("training steps (millions)  ·  0 = post-BC, pre-PPO", color=INK, fontsize=9)
    ax.set_ylabel("held-out success rate", color=INK, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    seed_result = result["seeds"][args.seed_index]
    curve = seed_result["curve"]
    baselines = result.get("baselines", {})
    random_rate = baselines.get("random", {}).get("summary", {}).get("success_rate")
    gate = result.get("heldout_milestone", {}).get("acceptance_threshold")
    ref = []
    if random_rate is not None:
        ref.append(("random", random_rate, RANDOM_C))
    if gate is not None:
        ref.append(("target-only gate", gate, GATE_C))

    args.out.mkdir(parents=True, exist_ok=True)

    # Figure 1: stochastic held-out success by optimal length across training.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_by(ax, curve, "stochastic", "optimal_length", "L", "viridis", ref)
    ax.set_title(
        "Held-out success by optimal solution length across training\n"
        "(stochastic policy; darker = longer/harder puzzle)",
        color=INK,
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(args.out / "by_optimal_length_stochastic.png", dpi=150, facecolor="white")
    plt.close(fig)

    # Figure 2: 2x2 — length (stoch/det) and robots-moved (stoch), plus aggregate.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_by(axes[0, 0], curve, "stochastic", "optimal_length", "L", "viridis", ref)
    axes[0, 0].set_title("By optimal length — stochastic", color=INK, fontsize=11, loc="left")
    _plot_by(axes[0, 1], curve, "deterministic", "optimal_length", "L", "viridis", ref)
    axes[0, 1].set_title("By optimal length — deterministic", color=INK, fontsize=11, loc="left")
    _plot_by(axes[1, 0], curve, "stochastic", "robots_moved", "R", "plasma", ref)
    axes[1, 0].set_title("By robots moved — stochastic", color=INK, fontsize=11, loc="left")

    # Aggregate panel (two categorical series: stoch / det) + training probe.
    ax = axes[1, 1]
    xs_m = [c["actual_timesteps"] / 1e6 for c in curve]
    for key, color, lbl in (("stochastic", "#2c6fbb", "held-out stochastic"),
                            ("deterministic", "#e08a1e", "held-out deterministic")):
        ys = [c[key]["success_rate"] for c in curve]
        ax.plot(xs_m, ys, "-o", color=color, linewidth=2, markersize=4, label=lbl, zorder=3)
    ys_train = [c["training"]["stochastic"]["success_rate"] for c in curve]
    ax.plot(xs_m, ys_train, "--", color=MUTED, linewidth=1.6, label="training-suite stochastic", zorder=2)
    for name, value, color in ref:
        ax.axhline(value, color=color, linewidth=1.3, linestyle="--", zorder=1)
        ax.annotate(name, xy=(xs_m[0], value), xytext=(0, 3), textcoords="offset points",
                    color=color, fontsize=7.5, va="bottom")
    _style(ax)
    ax.set_xlabel("training steps (millions)  ·  0 = post-BC, pre-PPO", color=INK, fontsize=9)
    ax.set_ylabel("success rate", color=INK, fontsize=9)
    ax.set_title("Aggregate success across training", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(
        "Phase 4b (16×16, 4 robots): BC warmstart + staged PPO — held-out performance across training",
        color=INK, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out / "training_curves_panels.png", dpi=150, facecolor="white")
    plt.close(fig)

    # Figure 3: final test breakdown by length (bars).
    test = seed_result["test"]
    lengths = sorted(test["stochastic"]["summary"]["by_optimal_length"], key=lambda k: int(k))
    stoch = [test["stochastic"]["summary"]["by_optimal_length"][k]["success_rate"] for k in lengths]
    det = [test["deterministic"]["summary"]["by_optimal_length"][k]["success_rate"] for k in lengths]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(lengths))
    ax.bar(x - 0.2, stoch, width=0.38, color="#2c6fbb", label="stochastic", zorder=3)
    ax.bar(x + 0.2, det, width=0.38, color="#9ecae1", label="deterministic", zorder=3)
    for name, value, color in ref:
        ax.axhline(value, color=color, linewidth=1.3, linestyle="--", zorder=2)
        ax.annotate(name, xy=(len(lengths) - 1, value), xytext=(0, 3), textcoords="offset points",
                    color=color, fontsize=7.5, va="bottom", ha="right")
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{k}" for k in lengths])
    ax.set_xlabel("optimal solution length", color=INK, fontsize=9)
    ax.set_ylabel("test success rate", color=INK, fontsize=9)
    ax.set_title("Final held-out test success by difficulty (untouched test suite)", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "final_test_by_length.png", dpi=150, facecolor="white")
    plt.close(fig)

    print(f"wrote 3 figures to {args.out}")


if __name__ == "__main__":
    main()
