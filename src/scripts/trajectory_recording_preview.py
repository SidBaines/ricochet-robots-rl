#%%
"""
Trajectory Recording Preview

This Jupyter-style script mirrors the trajectory recording used in `train_agent.py`.
It instantiates a simple environment factory and a dummy random policy implementing
`predict`, then uses `TrajectoryRecorder` to capture episode frames and saves them
to disk so you can quickly verify video formats and visuals without training.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


#%%
# Configuration
OUT_DIR = ARTIFACTS_ROOT / "rollout_gifs"
EPISODES = 5
ENV_HEIGHT = 8
ENV_WIDTH = 8
ENV_ROBOTS = 2
MAX_STEPS = 12
OBS_MODE = "image"  # env obs; recorder renders from board or env.render()
CHANNELS_FIRST = False
CAPTURE_RENDER = True
USE_RENDERER = True  # let recorder's default Matplotlib renderer draw frames
TRY_SAVE_MP4 = True
TRY_SAVE_GIF = True
FPS = 4


#%%
# Imports
from typing import Any, Dict, List

import numpy as np

from src.env import RicochetRobotsEnv
from src.monitoring.evaluators import TrajectoryRecorder


#%%
# Utility: ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving trajectory previews to: {OUT_DIR}")


#%%
# Environment factory mirroring train defaults (no curriculum)
def make_env() -> RicochetRobotsEnv:
    # Prefer using env.get_board() + renderer path inside TrajectoryRecorder. We still set render_mode
    # to 'rgb' so that fallback env.render() works if renderer import fails.
    env = RicochetRobotsEnv(
        height=ENV_HEIGHT,
        width=ENV_WIDTH,
        num_robots=ENV_ROBOTS,
        include_noop=True,
        max_steps=MAX_STEPS,
        obs_mode=OBS_MODE,
        channels_first=CHANNELS_FIRST,
        render_mode="rgb",
    )
    return env


#%%
# Dummy random "model" implementing Stable-Baselines-like predict(obs)->(action, None)
class RandomPolicy:
    def __init__(self, action_space_n: int, rng: np.random.Generator | None = None) -> None:
        self._n = int(action_space_n)
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def predict(self, obs: Any, deterministic: bool = True):  # noqa: D401 - SB3-compatible signature
        # Return a random discrete action index and a dummy state (None)
        del obs, deterministic
        a = int(self._rng.integers(0, self._n))
        return a, None


#%%
# Prepare policy using a probe env to get action space size
_probe_env = make_env()
_ = _probe_env.reset()
action_space_n = int(_probe_env.action_space.n)
_probe_env.close()
policy = RandomPolicy(action_space_n)
print(f"Initialized RandomPolicy with action_space_n={action_space_n}")


#%%
# Record trajectories
renderer = None  # let TrajectoryRecorder lazily create MatplotlibBoardRenderer
if not USE_RENDERER:
    renderer = None

recorder = TrajectoryRecorder(episodes=EPISODES, capture_render=CAPTURE_RENDER, renderer=renderer)
trajectories: List[Dict[str, Any]] = recorder.record(make_env, policy)

num_ok = sum(1 for t in trajectories if t.get("frames") is not None and len(t["frames"]) > 0)
print(f"Captured {len(trajectories)} episodes; frames present for {num_ok}.")


#%%
# Save videos per episode
def _save_gif(frames: np.ndarray | List[np.ndarray], path: Path, fps: int = 4) -> bool:
    try:
        import imageio.v2 as imageio
        arr = frames
        if isinstance(arr, list):
            arr = np.asarray(arr)
        arr = arr.astype(np.uint8)
        imageio.mimsave(path, arr, fps=fps)
        return True
    except Exception as e:
        print(f"GIF save failed ({path.name}): {e}")
        return False


def _save_mp4(frames: np.ndarray | List[np.ndarray], path: Path, fps: int = 4) -> bool:
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
        try:
            for f in frames:
                writer.append_data(np.asarray(f, dtype=np.uint8))
        finally:
            writer.close()
        return True
    except Exception as e:
        print(f"MP4 save failed ({path.name}): {e}")
        return False


for idx, traj in enumerate(trajectories):
    frames = traj.get("frames")
    if not CAPTURE_RENDER or frames is None or len(frames) == 0:
        print(f"Episode {idx:02d}: no frames captured; skipping save.")
        continue
    # Ensure (T,H,W,C) uint8
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 3:  # grayscale (T,H,W) -> (T,H,W,1)
        arr = arr[:, :, :, np.newaxis]

    gif_path = OUT_DIR / f"trajectory_ep{idx:02d}.gif"
    mp4_path = OUT_DIR / f"trajectory_ep{idx:02d}.mp4"

    saved_any = False
    if TRY_SAVE_GIF:
        saved_any |= _save_gif(arr, gif_path, fps=FPS)
    if TRY_SAVE_MP4:
        saved_any |= _save_mp4(arr, mp4_path, fps=FPS)
    if not saved_any:
        # Fallback: save first frame as PNG for inspection
        try:
            import imageio.v2 as imageio
            png_path = OUT_DIR / f"trajectory_ep{idx:02d}_frame0.png"
            imageio.imwrite(png_path, arr[0])
            print(f"Saved fallback frame: {png_path}")
        except Exception:
            pass

print("Trajectory recording preview complete.")



# %%
