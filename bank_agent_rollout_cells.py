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

import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import matplotlib.pyplot as plt

from env.puzzle_bank import PuzzleBank, SpecKey
from env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper
from env.visuals.mpl_renderer import MatplotlibBoardRenderer
from env.visuals.episode_recorder import EpisodeRecorder


#%%
# Configuration (edit these paths/values to match your setup)
# CHECKPOINT_PATH = Path("./checkpoints/ppo_model.zip").resolve()
CHECKPOINT_PATH = Path("./checkpoints/resnet_example.zip").resolve()
TRAIN_CONFIG_PATH = Path("./configs/train_defaults.yaml").resolve()
CURRICULUM_CONFIG_PATH = Path("./configs/curriculum_config_default.json").resolve()

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
bank_dir = Path(train_cfg.get("bank_dir", "./puzzle_bank")).resolve()
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
        from models import policies  # noqa: F401
    if train_config.get("resnet", False):
        from models import resnet  # noqa: F401
    if train_config.get("convlstm", False):
        from models import convlstm  # noqa: F401
    if train_config.get("drc", False):
        from models import drc_policy  # noqa: F401


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


def plot_episode_interactive(level_index: int, episode_index: int = 0) -> None:
    if make_subplots is None or go is None:
        raise RuntimeError("Plotly is not installed; pip install plotly to enable interactive viewer")
    level_data = rollouts.get(level_index)
    if level_data is None:
        raise KeyError(f"No rollout data for level {level_index}")
    frames = level_data["frames"][episode_index]
    trace = level_data["traces"][episode_index]
    rewards = [step.reward for step in trace.steps]
    cumulative = []
    total = 0.0
    for r in rewards:
        total += r
        cumulative.append(total)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{"type": "image"}, {"type": "scatter"}]])
    start_frame = frames[0]
    fig.add_trace(go.Image(z=start_frame), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(cumulative))), y=cumulative, mode="lines+markers", name="reward"), row=1, col=2)
    fig.update_layout(title=f"Level {level_index} Episode {episode_index}")
    frames_payload = []
    for step_idx, frame in enumerate(frames):
        frames_payload.append(go.Frame(data=[go.Image(z=frame), go.Scatter(x=list(range(step_idx + 1)), y=cumulative[: step_idx + 1], mode="lines+markers")], name=f"step_{step_idx}"))
    fig.frames = frames_payload
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
            ]
        }]
    )
    fig.show()

# Example usage (uncomment in a notebook):
plot_episode_interactive(level_index=0, episode_index=0)

#%%
def plot_episode_interactive(level_index: int, episode_index: int = 0):
    level_data = rollouts.get(level_index)
    if level_data is None:
        raise KeyError(f"No rollout data for level {level_index}")
    frames = level_data["frames"][episode_index]
    trace = level_data["traces"][episode_index]
    rewards = [step.reward for step in trace.steps]
    actions = [step.action for step in trace.steps]
    title = f"Level {level_index} Episode {episode_index}"
    # Prepare values as running (cumulative) reward for a smoother chart
    values = []
    total = 0.0
    for r in rewards:
        total += float(r)
        values.append(total)

    # Build figure with image on the left and line+markers on the right
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{"type": "image"}, {"type": "scatter"}]])
    if len(frames) == 0:
        raise ValueError("No frames to plot")
    fig.add_trace(go.Image(z=frames[0]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[values[0] if values else 0.0], mode='lines+markers'), row=1, col=2)

    # Axis titles and initial y-range with padding
    y_min = min(values) if values else 0.0
    y_max = max(values) if values else 0.0
    pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1.0
    fig.update_layout(
        title=title,
        xaxis2=dict(title="Step"),
        yaxis2=dict(title="Cumulative reward", range=[y_min - pad, y_max + pad]),
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
        curr_vals = values[:i+1] if values else [0.0]
        y_min = min(curr_vals)
        y_max = max(curr_vals)
        pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1.0
        frame_layout = go.Layout(title_text=f"Step {i}{', Action: ' + str(actions[i]) if i < len(actions) else ''}", yaxis2=dict(range=[y_min - pad, y_max + pad]))
        frame = go.Frame(
            data=[
                go.Image(z=frames[i]),
                go.Scatter(x=list(range(i+1)), y=curr_vals, mode='lines+markers')
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
