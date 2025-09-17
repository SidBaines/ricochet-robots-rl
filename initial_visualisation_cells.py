#%% imports and setup
import os
import json

from env.puzzle_bank import PuzzleBank, SpecKey
from env.curriculum import create_bank_curriculum_manager
from env.visuals.mpl_renderer import MatplotlibBoardRenderer
from env.visuals.plotly_renderer import PlotlyBoardRenderer
from env.visuals.episode_recorder import EpisodeRecorder

# SB3 lazy import
import importlib
sb3 = importlib.import_module("stable_baselines3")
PPO = sb3.PPO

#%% load curriculum config and bank
BANK_DIR = os.path.abspath("./puzzle_bank")
CFG_PATH = os.path.abspath("./curriculum_config_example2.json")

with open(CFG_PATH, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
levels = cfg.get("levels", [])
print(f"Levels: {levels}")

# Normalize spec_key entries to SpecKey objects expected by the bank
def _normalize_levels(level_list):
    out = []
    for lvl in level_list:
        lvl = dict(lvl)
        sk = lvl.get("spec_key")
        if isinstance(sk, dict):
            lvl["spec_key"] = SpecKey(
                height=sk["height"],
                width=sk["width"],
                num_robots=sk["num_robots"],
                edge_t_per_quadrant=sk["edge_t_per_quadrant"],
                central_l_per_quadrant=sk["central_l_per_quadrant"],
                generator_rules_version=sk.get("generator_rules_version", "1.0"),
            )
        out.append(lvl)
    return out

levels = _normalize_levels(levels)
print(f"Normalized levels: {levels}")

bank = PuzzleBank(BANK_DIR)

# Build a bank curriculum manager to access levels (thresholds here are dummies for viewing)
manager = create_bank_curriculum_manager(
    bank=bank,
    curriculum_levels=levels,
    success_rate_threshold=0.99,
    min_episodes_per_level=10,
    success_rate_window_size=50,
    advancement_check_frequency=10,
    verbose=True,
)

#%% helper: env factory like train_agent BankCurriculumWrapper
from env.curriculum import BankCurriculumWrapper

def make_env_for_level(level_index: int):
    # Build a manager restricted to the single requested level
    single_level = [levels[level_index]] if levels else None
    local_manager = create_bank_curriculum_manager(
        bank=bank,
        curriculum_levels=single_level,
        success_rate_threshold=1.0,
        min_episodes_per_level=1,
        success_rate_window_size=1,
        advancement_check_frequency=1,
        verbose=False,
    )
    return BankCurriculumWrapper(
        bank=bank,
        curriculum_manager=local_manager,
        obs_mode="rgb_image",
        channels_first=True,
        include_noop=bool(inferred_include_noop if inferred_include_noop is not None else True),
        render_mode=None,
        verbose=False,
    )

#%% load model checkpoint
# MODEL_PATH = os.path.abspath("checkpoints/convlstm-ppo-model-16chan-l2start_320000_steps.zip")
MODEL_PATH = os.path.abspath("checkpoints/convlstm-ppo-model-16chan-l2start_200000_steps.zip")
model = PPO.load(MODEL_PATH, device="cpu")

# Infer model's action space to align env settings
model_n = int(getattr(model.action_space, 'n', 0))
inferred_include_noop = None
inferred_num_robots = None
if model_n > 0:
    if model_n % 4 == 0:
        inferred_include_noop = False
        inferred_num_robots = model_n // 4
    elif (model_n - 1) % 4 == 0 and model_n >= 5:
        inferred_include_noop = True
        inferred_num_robots = (model_n - 1) // 4
    else:
        inferred_include_noop = True
        inferred_num_robots = max(1, (model_n - 1) // 4)

# Filter levels by robot count if determinable
if inferred_num_robots is not None:
    _levels_match = [lvl for lvl in levels if lvl.get("spec_key").num_robots == inferred_num_robots]
    if _levels_match:
        levels = _levels_match
    else:
        print(f"Warning: No levels found with num_robots={inferred_num_robots}. Using all levels.")

#%% record episodes with Matplotlib renderer
mpl_renderer = MatplotlibBoardRenderer(cell_size=24)
mpl_recorder = EpisodeRecorder(renderer=mpl_renderer)

# level_indices = [0, max(0, len(levels)//2), max(0, len(levels)-1)] if levels else [0]
level_indices = [0]
EPISODES_PER_LEVEL = 3  # increase to >1 to see variety per level
mpl_traces = []
mpl_frames_by_level = {}
for li in level_indices:
    env_factory = lambda li=li: make_env_for_level(li)
    for _ in range(EPISODES_PER_LEVEL):
        trace = mpl_recorder.record_single(env_factory, model, deterministic=True)
        mpl_traces.append((li, trace))
        mpl_frames_by_level.setdefault(li, []).append(mpl_recorder.to_rgb_frames(trace))

# Produce RGB frames for the first trace as a sanity check
mpl_frames_preview = mpl_frames_by_level[level_indices[0]][0] if mpl_traces else []

#%% record episodes with Plotly renderer
plotly_renderer = PlotlyBoardRenderer(cell_size=32, show_grid=True)
plotly_recorder = EpisodeRecorder(renderer=plotly_renderer)

plotly_traces = []
plotly_frames_by_level = {}
for li in level_indices:
    env_factory = lambda li=li: make_env_for_level(li)
    for _ in range(EPISODES_PER_LEVEL):
        trace = plotly_recorder.record_single(env_factory, model, deterministic=True)
        plotly_traces.append((li, trace))
        plotly_frames_by_level.setdefault(li, []).append(plotly_recorder.to_rgb_frames(trace))

plotly_frames_preview = plotly_frames_by_level[level_indices[0]][0] if plotly_traces else []

#%% display hints (in notebooks you could show with PIL or matplotlib imshow)
print({
    "mpl_levels": [li for li, _ in mpl_traces],
    "plotly_levels": [li for li, _ in plotly_traces],
    "mpl_frames_preview_shape": (len(mpl_frames_preview),) + (mpl_frames_preview[0].shape if mpl_frames_preview else (0,)),
    "plotly_frames_preview_shape": (len(plotly_frames_preview),) + (plotly_frames_preview[0].shape if plotly_frames_preview else (0,)),
})

#%% interactive plotly viewer combining frames + running reward
from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore

def plot_episode_interactive(frames, rewards, actions, title="Ricochet Robots Agent Play"):
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

# Example: visualize the first Matplotlib-rendered trace
if mpl_traces:
    first_li, first_trace = mpl_traces[0]
    rewards = [s.reward for s in first_trace.steps]
    actions = [s.action for s in first_trace.steps]
    first_frames = mpl_frames_by_level[first_li][0]
    fig = plot_episode_interactive(first_frames, rewards, actions, title="Ricochet Robots (Matplotlib frames)")
    try:
        fig.show()
    except Exception:
        pass


# %%
# Visualise all matplotlib and plotly traces
if mpl_traces:
    for li, trace in mpl_traces:
        rewards = [s.reward for s in trace.steps]
        actions = [s.action for s in trace.steps]
        fig = plot_episode_interactive(mpl_frames_preview, rewards, actions, title=f"Ricochet Robots (Matplotlib frames) - Level {li}")
        try:
            fig.show()
        except Exception:
            pass
if plotly_traces:
    for li, trace in plotly_traces:
        rewards = [s.reward for s in trace.steps]
        actions = [s.action for s in trace.steps]
        fig = plot_episode_interactive(plotly_frames_preview, rewards, actions, title=f"Ricochet Robots (Plotly frames) - Level {li}")
        try:
            fig.show()
        except Exception:
            pass

# %%
from matplotlib import pyplot as plt
plt.imshow(mpl_frames_preview[0]); plt.axis('off')
# %%
from PIL import Image
Image.fromarray(plotly_frames_preview[0])
# %%
import imageio.v2 as iio
iio.mimsave('episode_mpl.gif', mpl_frames_preview, fps=4)
# %%
import imageio.v2 as iio
iio.mimsave('episode_plotly.mp4', plotly_frames_preview, fps=4, codec='libx264')
# %%
