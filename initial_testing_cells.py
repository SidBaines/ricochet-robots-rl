#%%
"""
Ricochet Robots - Initial Environment Testing Cells

How to use:
- Run each cell independently in a Jupyter-aware editor (e.g., VS Code) that supports #%% cells.
- Edit parameters in-place and re-run cells to iterate.

This script lets you:
- Create fixed or random boards
- Render ASCII boards
- Step through custom action sequences
- Run the solver and replay its plan
- Do random rollouts for smoke testing
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np

from env.ricochet_env import RicochetRobotsEnv, FixedLayout
from env.ricochet_core import UP, DOWN, LEFT, RIGHT
from env.solver import solve_bfs, solve_astar


#%%
# Configuration: tweak these defaults and re-run this cell
HEIGHT: int = 16
WIDTH: int = 16
NUM_ROBOTS: int = 4
INCLUDE_NOOP: bool = True
OBS_MODE: str = "image"  # "image" or "symbolic"
CHANNELS_FIRST: bool = False
STEP_PENALTY: float = -0.01
NOOP_PENALTY: Optional[float] = None  # None -> use STEP_PENALTY
GOAL_REWARD: float = 1.0
MAX_STEPS: int = 100
RENDER_MODE: Optional[str] = "rgb"  # "ascii", "rgb" or None
SEED: Optional[int] = 123

# When ensuring solvable random boards
ENSURE_SOLVABLE: bool = False
SOLVER_MAX_DEPTH: int = 40
SOLVER_MAX_NODES: int = 40000


#%%
# Helper: build a simple fixed layout you can edit
def make_simple_fixed_layout() -> FixedLayout:
    H, W = HEIGHT, WIDTH
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    # Borders
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True

    # Example interior walls; edit these
    # Horizontal wall between rows 2-3 at column 3
    if H > 2 and W > 3:
        h_walls[2, 3] = True
    # Vertical wall between cols 1-2 at row 4
    if H > 4 and W > 2:
        v_walls[4 - 1, 2] = True  # note: v_walls shape is (H, W+1)

    # Robot positions; ensure unique cells and inside bounds
    robots: Dict[int, Tuple[int, int]] = {}
    robots[0] = (H - 2, 1)
    if NUM_ROBOTS >= 2:
        robots[1] = (1, W - 2)
    if NUM_ROBOTS >= 3:
        robots[2] = (H // 2, W // 2)

    goal: Tuple[int, int] = (1, 1)
    target_robot: int = 0

    return FixedLayout(
        height=H,
        width=W,
        h_walls=h_walls,
        v_walls=v_walls,
        robot_positions=robots,
        goal_position=goal,
        target_robot=target_robot,
    )


#%%
# Create an environment (choose fixed_layout or random with/without ensure_solvable)
USE_FIXED: bool = False

fixed_layout: Optional[FixedLayout] = make_simple_fixed_layout() if USE_FIXED else None

env = RicochetRobotsEnv(
    height=HEIGHT,
    width=WIDTH,
    num_robots=NUM_ROBOTS,
    include_noop=INCLUDE_NOOP,
    step_penalty=STEP_PENALTY,
    goal_reward=GOAL_REWARD,
    noop_penalty=NOOP_PENALTY,
    max_steps=MAX_STEPS,
    fixed_layout=fixed_layout,
    seed=SEED,
    ensure_solvable=ENSURE_SOLVABLE,
    solver_max_depth=SOLVER_MAX_DEPTH,
    solver_max_nodes=SOLVER_MAX_NODES,
    obs_mode=OBS_MODE,  # type: ignore[arg-type]
    channels_first=CHANNELS_FIRST,
    render_mode=RENDER_MODE,
)
obs, info = env.reset()
print("Environment reset. Info:", info)
frame = env.render()
if RENDER_MODE == "ascii" and frame is not None:
    print(frame)
elif RENDER_MODE == "rgb" and frame is not None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.title("Ricochet Robots - RGB Render")
        plt.axis("off")
        plt.show()
    except ImportError:
        print("matplotlib not available to display RGB frame; shape:", getattr(frame, "shape", None))


#%%
# Convenience: list action encoding and simple helpers
def encode_action(robot_id: int, direction: int) -> int:
    return robot_id * 4 + direction


DIRS = {"u": UP, "d": DOWN, "l": LEFT, "r": RIGHT}
DIRS_INVERSE = {v: k for k, v in DIRS.items()}
print("Action encoding: action = robot_id * 4 + direction (u=0,d=1,l=2,r=3), noop is last if enabled.")


#%%
# Step through a custom action sequence you edit here
# Each action is a tuple: (robot_id, direction), where direction in {UP, DOWN, LEFT, RIGHT}
custom_actions: List[Tuple[int, int]] = [
    (0, UP),
    (0, LEFT),
]

terminated_any = False
truncated_any = False
total_reward = 0.0

for rid, d in custom_actions:
    a = encode_action(rid, d)
    obs, reward, terminated, truncated, step_info = env.step(a)
    total_reward += reward
    frame = env.render()
    if RENDER_MODE == "ascii" and frame is not None:
        print(frame)
    print({"reward": reward, "terminated": terminated, "truncated": truncated, "info": step_info})
    terminated_any = terminated_any or terminated
    truncated_any = truncated_any or truncated
    if terminated or truncated:
        break

print(f"Cumulative reward: {total_reward:.3f}")


#%%
# Run solver on current board and replay its plan (BFS)
solution = solve_bfs(env.get_board(), max_depth=SOLVER_MAX_DEPTH, max_nodes=SOLVER_MAX_NODES)
if solution is None:
    print("Solver did not find a plan within limits.")
else:
    print(f"BFS plan length: {len(solution)} actions")
    # Replay from a fresh reset to ensure matching transitions
    env.reset()
    total_reward = 0.0
    for rid, d in solution:
        a = encode_action(rid, d)
        obs, reward, terminated, truncated, step_info = env.step(a)
        total_reward += reward
        frame = env.render()
        if RENDER_MODE == "ascii" and frame is not None:
            print(frame)
        print({"move": (rid, d), "reward": reward, "terminated": terminated, "truncated": truncated})
        if terminated or truncated:
            break
    print(f"Cumulative reward following solver plan: {total_reward:.3f}")




#%%
# Alternative: A* with selectable heuristic
H_MODE = "admissible_zero"  # one of: "admissible_zero", "admissible_one", "manhattan_cells"
H_MODE = "admissible_one"  # one of: "admissible_zero", "admissible_one", "manhattan_cells"
# Time how long it takes to solve 1000 times
import time
start_time = time.time()
for i in range(1):
    env.reset()
    frame = env.render()
    if RENDER_MODE == "ascii":
        print(frame)
    elif RENDER_MODE == "rgb" and frame is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure(figsize=(6, 6))
            plt.imshow(frame)
            plt.title("Ricochet Robots - RGB Render")
            plt.axis("off")
            plt.show()
        except ImportError:
            print("matplotlib not available to display RGB frame; shape:", getattr(frame, "shape", None))
    astar_sol = solve_astar(env.get_board(), max_depth=200, max_nodes=100000, h_mode=H_MODE)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
if astar_sol is None:
    print("A* did not find a plan within limits.")
else:
    print(f"A* plan length: {len(astar_sol)} (mode={H_MODE})")
# Print the moves in human readable format
for rid, d in astar_sol:
    print(f"Move: {rid}, Direction: {DIRS_INVERSE[d]}")
print(f"Solve length: {len(astar_sol)}")


#%%
# Random rollout for smoke testing
import random

env.reset()
steps = 20
total_reward = 0.0
for t in range(steps):
    a = random.randrange(env.action_space.n)
    obs, reward, terminated, truncated, step_info = env.step(a)
    total_reward += reward
    frame = env.render()
    if RENDER_MODE == "ascii" and frame is not None:
        print(frame)
    print({"t": t + 1, "a": a, "reward": reward, "terminated": terminated, "truncated": truncated})
    if terminated or truncated:
        break
print(f"Random rollout cumulative reward: {total_reward:.3f}")


#%%
# Inspect observation format and channel names (image mode) or vector (symbolic)
if OBS_MODE == "image":
    print("Image observation shape:", obs.shape)
    if hasattr(env, "channel_names"):
        print("Channels:", env.channel_names)
        # Show which cells are occupied by each robot channel
        if not CHANNELS_FIRST:
            base = 6
            for rid in range(NUM_ROBOTS):
                ch = base + rid
                count = int(np.count_nonzero(obs[:, :, ch]))
                print(f"robot_{rid} nonzeros:", count)
else:
    print("Symbolic observation vector length:", obs.shape)
    print("obs:", obs)


#%%
# Build a custom layout from scratch (advanced):
# - Edit the arrays h_walls and v_walls to draw edge walls.
# - Set robots and goal, then rebuild env with this layout.
H2, W2 = HEIGHT, WIDTH
h_walls2 = np.zeros((H2 + 1, W2), dtype=bool)
v_walls2 = np.zeros((H2, W2 + 1), dtype=bool)
# Borders
h_walls2[0, :] = True
h_walls2[H2, :] = True
v_walls2[:, 0] = True
v_walls2[:, W2] = True

# Example interior: make a small room top-left
if H2 > 3 and W2 > 3:
    h_walls2[1, 1] = True
    h_walls2[1, 2] = True
    v_walls2[1, 1] = True

robots2: Dict[int, Tuple[int, int]] = {i: (i % H2, (i * 2) % W2) for i in range(NUM_ROBOTS)}
goal2: Tuple[int, int] = (H2 - 2, W2 - 2)
target2: int = 0

layout2 = FixedLayout(
    height=H2,
    width=W2,
    h_walls=h_walls2,
    v_walls=v_walls2,
    robot_positions=robots2,
    goal_position=goal2,
    target_robot=target2,
)

env2 = RicochetRobotsEnv(
    fixed_layout=layout2,
    include_noop=INCLUDE_NOOP,
    step_penalty=STEP_PENALTY,
    goal_reward=GOAL_REWARD,
    obs_mode=OBS_MODE,  # type: ignore[arg-type]
    render_mode=RENDER_MODE,
)
env2.reset()
frame2 = env2.render()
if frame2 is not None:
    print(frame2)



# %%
