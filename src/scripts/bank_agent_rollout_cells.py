#%%
"""
Ricochet Robots - Bank Curriculum Rollout Viewer

Run this Jupyter-style script inside VS Code (or any editor that understands
`#%%` cells) to inspect a trained agent across every curriculum level. It uses
an SB3 checkpoint together with the YAML training config and a bank curriculum
JSON to recreate the evaluation environments and capture a few preview rollouts.
"""


#%%
# Imports
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

import json
import importlib
import math
from typing import Any, Dict, List, Optional, Tuple

import yaml
import matplotlib.pyplot as plt
import numpy as np

from src.env.puzzle_bank import PuzzleBank, SpecKey
from src.env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper
from src.env.visuals.mpl_renderer import MatplotlibBoardRenderer
from src.env.visuals.episode_recorder import EpisodeRecorder


#%%
# Configuration (edit these paths/values to match your setup)
# CHECKPOINT_PATH = (ARTIFACTS_ROOT / "checkpoints" / "ppo_model.zip").resolve()
CHECKPOINT_PATH = (ARTIFACTS_ROOT / "checkpoints" / "resnet_example.zip").resolve()
TRAIN_CONFIG_PATH = (PROJECT_ROOT / "configs" / "train_defaults.yaml").resolve()
CURRICULUM_CONFIG_PATH = (PROJECT_ROOT / "configs" / "curriculum_config_default.json").resolve()

EPISODES_PER_LEVEL = 10          # number of episodes captured for each curriculum level
DETERMINISTIC_POLICY = True     # set False to sample stochastic actions
DEVICE_OVERRIDE: Optional[str] = None  # use "cpu", "cuda", "mps" or None to honour config/device auto-detect
PLOT_MAX_FRAMES = 10            # cap how many frames to show per episode preview


#%%
# Helpers to load config files

def load_training_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in training config {path}, got {type(data).__name__}")
    return data


def load_curriculum_levels(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    levels = data.get("levels", [])
    out: List[Dict[str, Any]] = []
    for idx, raw_level in enumerate(levels):
        level = dict(raw_level)
        spec_dict = level.get("spec_key")
        if isinstance(spec_dict, dict):
            level["spec_key"] = SpecKey(
                height=int(spec_dict["height"]),
                width=int(spec_dict["width"]),
                num_robots=int(spec_dict["num_robots"]),
                edge_t_per_quadrant=int(spec_dict["edge_t_per_quadrant"]),
                central_l_per_quadrant=int(spec_dict["central_l_per_quadrant"]),
                generator_rules_version=spec_dict.get("generator_rules_version", 1),
            )
        if not isinstance(level.get("spec_key"), SpecKey):
            raise ValueError(f"Level {idx} missing spec_key after conversion")
        out.append(level)
    return out


#%%
# Load configuration data
train_cfg = load_training_config(TRAIN_CONFIG_PATH)
levels = load_curriculum_levels(CURRICULUM_CONFIG_PATH)

print(f"Loaded training config from {TRAIN_CONFIG_PATH}")
print(f"Loaded {len(levels)} curriculum levels from {CURRICULUM_CONFIG_PATH}")

if not CHECKPOINT_PATH.exists():
    print(f"Warning: checkpoint not found at {CHECKPOINT_PATH}")


#%%
# Curriculum/bank scaffolding
# bank_dir = (Path(train_cfg.get("bank_dir", str(ARTIFACTS_ROOT / "puzzle_bank")))).resolve()
bank_dir = (Path(str(PROJECT_ROOT) + '/' + train_cfg.get("bank_dir", str( "artifacts/puzzle_bank")))).resolve()
if not bank_dir.exists():
    raise FileNotFoundError(f"Puzzle bank directory not found: {bank_dir}")

bank = PuzzleBank(str(bank_dir))

curriculum_kwargs = dict(
    bank=bank,
    curriculum_levels=levels,
    success_rate_threshold=float(train_cfg.get("curriculum_success_threshold", 0.8)),
    min_episodes_per_level=int(train_cfg.get("curriculum_min_episodes", 100)),
    success_rate_window_size=int(train_cfg.get("curriculum_window_size", 200)),
    advancement_check_frequency=int(train_cfg.get("curriculum_check_freq", 50)),
    verbose=bool(train_cfg.get("curriculum_verbose", True)),
)


#%%
# Agent loader utilities

def detect_device(configured: str, override: Optional[str]) -> str:
    if override:
        return override
    device = configured
    if device == "auto":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return device


def ensure_custom_modules(train_config: Dict[str, Any]) -> None:
    """Import optional custom policies so SB3 can deserialise them."""
    if train_config.get("small_cnn", False):
        from src.models import policies  # noqa: F401
    if train_config.get("resnet", False):
        from src.models import resnet  # noqa: F401
    if train_config.get("convlstm", False):
        from src.models import convlstm  # noqa: F401
    if train_config.get("drc", False):
        from src.models import drc_policy  # noqa: F401


def load_agent(checkpoint: Path, train_config: Dict[str, Any], override_device: Optional[str]) -> Tuple[Any, str]:
    ensure_custom_modules(train_config)

    sb3 = importlib.import_module("stable_baselines3")
    PPO = sb3.PPO

    try:
        sb3c = importlib.import_module("sb3_contrib")
        RecurrentPPO = getattr(sb3c, "RecurrentPPO", None)
    except Exception:
        RecurrentPPO = None

    device = detect_device(train_config.get("device", "cpu"), override_device)

    use_recurrent = bool(train_config.get("convlstm") or train_config.get("drc"))
    if use_recurrent and RecurrentPPO is None:
        raise ImportError("Recurrent checkpoint requested but sb3-contrib is not installed")

    loader = RecurrentPPO.load if (use_recurrent and RecurrentPPO is not None) else PPO.load
    model = loader(str(checkpoint), device=device)
    model.policy.to(device)
    return model, device


#%%
# Load agent checkpoint
model, device_used = load_agent(CHECKPOINT_PATH, train_cfg, DEVICE_OVERRIDE)
print(f"Loaded model on device '{device_used}' from {CHECKPOINT_PATH}")
print(model.policy.__class__)


#%%
# Infer action metadata (noop + robot count)

def infer_action_metadata(model: Any) -> Tuple[Optional[bool], Optional[int]]:
    action_space = getattr(model, "action_space", None)
    action_n = getattr(action_space, "n", None)
    if action_n is None:
        return None, None
    if action_n % 4 == 0:
        return False, action_n // 4
    if action_n >= 5 and (action_n - 1) % 4 == 0:
        return True, (action_n - 1) // 4
    # Fallback if noop encoding is unusual
    return None, None


inferred_include_noop, inferred_num_robots = infer_action_metadata(model)
config_include_noop = bool(train_cfg.get("include_noop", True))
include_noop = inferred_include_noop if inferred_include_noop is not None else config_include_noop

if inferred_include_noop is not None and inferred_include_noop != config_include_noop:
    print(f"Warning: model inference suggests include_noop={inferred_include_noop}, config has {config_include_noop}")

if inferred_num_robots is not None:
    level_matches = [lvl for lvl in levels if lvl["spec_key"].num_robots == inferred_num_robots]
    if level_matches:
        if len(level_matches) != len(levels):
            print(f"Filtering curriculum to {len(level_matches)} levels matching num_robots={inferred_num_robots}")
        levels = level_matches
    else:
        print(f"Warning: no curriculum levels match num_robots={inferred_num_robots}")


#%%
# Environment factory per level
obs_mode = train_cfg.get("obs_mode", "rgb_image")
channels_first = bool(obs_mode in ("image", "rgb_image"))
render_mode = None  # EpisodeRecorder relies on board data, so render_mode can stay None here


def make_env_factory_for_level(level_index: int):
    def _factory() -> BankCurriculumWrapper:
        manager = create_bank_curriculum_manager(**curriculum_kwargs)
        manager.current_level = level_index
        manager.verbose = False
        return BankCurriculumWrapper(
            bank=bank,
            curriculum_manager=manager,
            obs_mode=obs_mode,
            channels_first=channels_first,
            include_noop=include_noop,
            render_mode=render_mode,
            verbose=False,
        )
    return _factory


#%%
# Capture episodes
renderer = MatplotlibBoardRenderer(cell_size=28)
recorder = EpisodeRecorder(renderer=renderer)

rollouts: Dict[int, Dict[str, Any]] = {}

for level_idx, level_spec in enumerate(levels):
    env_factory = make_env_factory_for_level(level_idx)
    level_traces = []
    level_frames = []
    for episode in range(EPISODES_PER_LEVEL):
        trace = recorder.record_single(env_factory, model, deterministic=DETERMINISTIC_POLICY)
        frames = recorder.to_rgb_frames(trace)
        level_traces.append(trace)
        level_frames.append(frames)
    rollouts[level_idx] = {
        "spec": level_spec,
        "traces": level_traces,
        "frames": level_frames,
    }
    successes = sum(1 for t in level_traces if t.info.get("is_success", False))
    print(f"Level {level_idx}: {level_spec['name']} — captured {len(level_traces)} episodes, successes={successes}")


#%%
# Quick summary table
from pprint import pprint

summary = []
for idx, data in rollouts.items():
    spec = data["spec"]
    wins = sum(1 for t in data["traces"] if t.info.get("is_success", False))
    avg_steps = sum(len(t.steps) for t in data["traces"]) / max(1, len(data["traces"]))
    summary.append({
        "level": idx,
        "name": spec.get("name", f"level_{idx}"),
        "episodes": len(data["traces"]),
        "successes": wins,
        "success_rate": wins / max(1, len(data["traces"])),
        "avg_steps": avg_steps,
        "optimal_length_hint": spec.get("max_optimal_length"),
    })

pprint(summary)


#%%
# Plot a handful of frames per level using matplotlib
for idx, data in rollouts.items():
    if not data["frames"]:
        continue
    num_to_plot = len(data['frames'])
    for episode_idx in range(num_to_plot):
        first_episode_frames = data["frames"][episode_idx]
        display_frames = first_episode_frames[: min(PLOT_MAX_FRAMES, len(first_episode_frames))]
        if not display_frames:
            continue
        nrows = 1 + int(len(display_frames)>5)
        ncols = (len(display_frames)-1) // nrows + 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2.3 * nrows))
        # fig, axes = plt.subplots(1, len(display_frames), figsize=(2 * len(display_frames), 8))
        if nrows == 1:
            # axes = [axes]
            axes = axes.reshape(1,-1)
        # for i, (ax, frame) in enumerate(zip(axes, display_frames)):
        for i in range(len(display_frames)):
            ax = axes[i//ncols, i%ncols]
            frame = display_frames[i]
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(f"Step {i}")
        spec = data["spec"]
        fig.suptitle(f"Level {idx}: {spec.get('name', f'level_{idx}')} — episode preview")
        # plt.tight_layout()
        plt.show()


#%%
# Optional: interactive Plotly viewer for a selected rollout
try:
    from plotly.subplots import make_subplots  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except Exception as exc:
    make_subplots = None
    go = None
    print(f"Plotly not available: {exc}")


def _axis_range(series: List[float]) -> Tuple[float, float]:
    finite = [val for val in series if val is not None and not math.isnan(val)]
    if not finite:
        return (-1.0, 1.0)
    min_v = min(finite)
    max_v = max(finite)
    if math.isclose(min_v, max_v):
        pad = max(1.0, abs(min_v) * 0.1 or 1.0)
        return (min_v - pad, max_v + pad)
    pad = (max_v - min_v) * 0.1
    return (min_v - pad, max_v + pad)


def _format_step_label(step_idx: int, total_steps: int, pause: bool = False) -> str:
    step_num = min(step_idx, total_steps)
    label = f"Step {step_num}/{total_steps}"
    if pause:
        label += " · reset"
    return label


def _step_annotations(step_idx: int, total_steps: int, pause: bool = False) -> List[Dict[str, Any]]:
    return [
        dict(
            text=_format_step_label(step_idx, total_steps, pause=pause),
            x=0.5,
            xref="paper",
            y=1.08,
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#ffffff"),
            align="center",
            bgcolor="rgba(0,0,0,0.6)",
            borderpad=4,
        )
    ]


#%%
def plot_episode_interactive(level_index: int, episode_index: int = 0):
    if make_subplots is None or go is None:
        raise RuntimeError("Plotly is not available in this environment")
    level_data = rollouts.get(level_index)
    if level_data is None:
        raise KeyError(f"No rollout data for level {level_index}")
    frames = level_data["frames"][episode_index]
    trace = level_data["traces"][episode_index]
    rewards = [step.reward for step in trace.steps]
    actions = [step.action for step in trace.steps]
    total_steps = len(trace.steps)
    state_values = list(getattr(trace, "values", []))
    title = f"Level {level_index} Episode {episode_index}"
    # Prepare cumulative rewards for a smoother chart
    cumulative_rewards: List[float] = []
    total = 0.0
    for r in rewards:
        total += float(r)
        cumulative_rewards.append(total)

    if len(state_values) != len(frames):
        state_values = [float("nan")] * len(frames)

    # Build figure with image, cumulative reward, and value estimate panels
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.55, 0.225, 0.225],
        specs=[[{"type": "image"}, {"type": "scatter"}, {"type": "scatter"}]],
    )
    if len(frames) == 0:
        raise ValueError("No frames to plot")
    fig.add_trace(go.Image(z=frames[0]), row=1, col=1)
    initial_reward = cumulative_rewards[0] if cumulative_rewards else 0.0
    initial_value = state_values[0] if state_values else float("nan")
    fig.add_trace(go.Scatter(x=[0], y=[initial_reward], mode="lines+markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[initial_value], mode="lines+markers"), row=1, col=3)

    # Axis titles and initial y-range with padding
    reward_range = _axis_range(cumulative_rewards[:1])
    value_range = _axis_range(state_values[:1])
    fig.update_layout(
        title=title,
        xaxis2=dict(title="Step"),
        yaxis2=dict(title="Cumulative reward", range=list(reward_range)),
        xaxis3=dict(title="Step"),
        yaxis3=dict(title="Value estimate", range=list(value_range)),
        annotations=_step_annotations(0, total_steps),
    )

    # Updatemenus and slider
    updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(label="⏮", method="animate", args=[["previous"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
            dict(label="⏭", method="animate", args=[["next"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
            dict(label="▶", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
        ],
        direction="left",
        pad={"r": 10, "t": 10},
        showactive=False,
        x=0.1,
        xanchor="right",
        y=1.1,
        yanchor="top"
    )]

    steps = []
    fig_frames = []
    for i in range(len(frames)):
        reward_points = cumulative_rewards[: min(i + 1, len(cumulative_rewards))]
        value_points = state_values[: i + 1]
        reward_range = _axis_range(reward_points)
        value_range = _axis_range(value_points)
        frame_layout = go.Layout(
            title_text=f"Step {i}{', Action: ' + str(actions[i]) if i < len(actions) else ''}",
            yaxis2=dict(range=list(reward_range)),
            yaxis3=dict(range=list(value_range)),
            annotations=_step_annotations(i, total_steps),
        )
        frame = go.Frame(
            data=[
                go.Image(z=frames[i]),
                go.Scatter(x=list(range(len(reward_points))), y=reward_points, mode="lines+markers"),
                go.Scatter(x=list(range(len(value_points))), y=value_points, mode="lines+markers"),
            ],
            layout=frame_layout,
            name=str(i)
        )
        fig_frames.append(frame)
        steps.append({
            "args": [[i], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": str(i),
            "method": "animate"
        })

    fig.frames = fig_frames
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {"font": {"size": 20}, "prefix": "Step: ", "visible": True, "xanchor": "right"},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": steps,
    }]

    fig.update_layout(updatemenus=updatemenus, sliders=sliders)
    return fig

found = False
for level_index in range(len(levels)):
    for episode_index in range(len(rollouts[level_index]["frames"])):
        # print(len(rollouts[level_index].get('frames')[episode_index]) )
        if len(rollouts[level_index].get('frames')[episode_index]) < 10:
            found=True
        if found:
            break
    if found:
        break
plot_episode_interactive(level_index=level_index, episode_index=episode_index)

# %%
# Save episodes as GIFs for deeper inspection
def save_episode_gifs(
    level_index: int,
    episode_index: int = 0,
    output_dir: Optional[Path] = None,
    fps: int = 4,
    pause_seconds: float = 0.75,
) -> Dict[str, Optional[Path]]:
    """Export the requested rollout as matplotlib and Plotly GIF animations."""

    level_data = rollouts.get(level_index)
    if level_data is None:
        raise KeyError(f"No rollout data for level {level_index}")
    if episode_index >= len(level_data["frames"]):
        raise IndexError(f"Episode index {episode_index} out of range for level {level_index}")

    frames = level_data["frames"][episode_index]
    if not frames:
        raise ValueError("No frames available for the requested episode")

    trace = level_data["traces"][episode_index]
    actions = [step.action for step in trace.steps]
    rewards = [step.reward for step in trace.steps]
    cumulative_rewards: List[float] = []
    total = 0.0
    for r in rewards:
        total += float(r)
        cumulative_rewards.append(total)
    state_values = list(getattr(trace, "values", []))
    if len(state_values) != len(frames):
        state_values = [float("nan")] * len(frames)
    total_steps = len(trace.steps)

    output_path = Path(output_dir) if output_dir is not None else (ARTIFACTS_ROOT / "rollout_gifs")
    output_path.mkdir(parents=True, exist_ok=True)
    level_name = level_data["spec"].get("name", f"level_{level_index}")
    safe_level = "".join(char if char.isalnum() or char in "-_" else "_" for char in level_name)
    base_name = f"level{level_index:02d}_{safe_level}_episode{episode_index:02d}"

    # Matplotlib-style GIF using recorded RGB frames
    mpl_path = output_path / f"{base_name}_mpl.gif"
    pause_frames = int(round(max(0.0, pause_seconds) * fps))
    try:
        import imageio.v2 as imageio  # type: ignore
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore

            font = ImageFont.load_default()
            # Try a larger TrueType font for clearer, bigger annotation text
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 22)
            except Exception:
                try:
                    font = ImageFont.truetype("Arial.ttf", 22)
                except Exception:
                    pass

            def annotate_step(frame_array: np.ndarray, step_idx: int, pause: bool = False) -> np.ndarray:
                base_image = Image.fromarray(frame_array)
                draw_probe = ImageDraw.Draw(base_image)
                label = _format_step_label(step_idx, total_steps, pause=pause)
                try:
                    bbox = draw_probe.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older Pillow: use textsize to measure text
                    text_w, text_h = draw_probe.textsize(label, font=font)
                padding = 4
                margin_top = text_h + (padding * 2) + 8
                new_image = Image.new("RGB", (base_image.width, base_image.height + margin_top), color=(0, 0, 0))
                # paste original frame below the top annotation band
                new_image.paste(base_image, (0, margin_top))
                draw = ImageDraw.Draw(new_image)
                # Center text horizontally within the new image width
                x0 = max(0, (new_image.width - text_w) // 2)
                y0 = padding
                # optional background box behind text for readability, already black band
                # Draw text in the top band
                draw.text((x0, y0), label, fill=(255, 255, 255), font=font)
                return np.asarray(new_image)

        except Exception as pillow_exc:  # pragma: no cover - optional dependency
            print(f"Warning: step overlay skipped (Pillow unavailable: {pillow_exc})")

            def annotate_step(frame_array: np.ndarray, step_idx: int, pause: bool = False) -> np.ndarray:
                return np.asarray(frame_array)

        annotated_frames = [annotate_step(np.asarray(frame), idx, pause=False) for idx, frame in enumerate(frames)]
        pause_frame = annotate_step(np.asarray(frames[-1]), total_steps, pause=True)

        with imageio.get_writer(mpl_path, mode="I", fps=fps, loop=0) as writer:
            for annotated in annotated_frames:
                writer.append_data(annotated)
            for _ in range(pause_frames):
                writer.append_data(pause_frame)
    except Exception as exc:
        print(f"Warning: failed to save matplotlib GIF: {exc}")
        mpl_path = None

    # Plotly GIF reusing the interactive figure definition
    plotly_path: Optional[Path] = output_path / f"{base_name}_plotly.gif"
    try:
        fig = plot_episode_interactive(level_index, episode_index)
        gif_fig = go.Figure(fig)
        gif_frames = list(gif_fig.frames)
        plotly_pause_frames = max(1, pause_frames) if pause_seconds > 0 else 0
        if gif_frames and plotly_pause_frames > 0:
            reward_range = _axis_range(cumulative_rewards)
            value_range = _axis_range(state_values)
            pause_data = [
                go.Image(z=frames[-1]),
                go.Scatter(x=list(range(len(cumulative_rewards))), y=cumulative_rewards, mode="lines+markers"),
                go.Scatter(x=list(range(len(state_values))), y=state_values, mode="lines+markers"),
            ]
            pause_layout = go.Layout(
                title_text=f"Step {total_steps}",
                yaxis2=dict(range=list(reward_range)),
                yaxis3=dict(range=list(value_range)),
                annotations=_step_annotations(total_steps, total_steps, pause=True),
            )
            for pause_idx in range(plotly_pause_frames):
                gif_frames.append(
                    go.Frame(
                        data=pause_data,
                        layout=pause_layout,
                        name=f"pause_{pause_idx}",
                    )
                )
            gif_fig.frames = tuple(gif_frames)

        gif_fig.write_image(
            str(plotly_path),
            format="gif",
        )
    except Exception as exc:
        print(f"Warning: failed to save Plotly GIF: {exc}")
        plotly_path = None

    return {"matplotlib": mpl_path, "plotly": plotly_path}


# Example usage:
save_episode_gifs(level_index=0, episode_index=0)

# %%
