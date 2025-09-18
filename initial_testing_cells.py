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
OBS_MODE: str = "image"  # "image", "symbolic", or "rgb_image"
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



#%%
# =============================================================================
# CURRICULUM EXPLORATION CELLS
# =============================================================================
# These cells help you explore curriculum environments at different levels,
# understand level generation behavior, and test reset consistency.

#%%
# Create curriculum manager and explore available levels
from env.curriculum import create_curriculum_manager, create_default_curriculum

# Create curriculum manager with default configuration
curriculum_config = create_default_curriculum()
curriculum_manager = create_curriculum_manager(curriculum_config, initial_level=0, verbose=True)

print("Available curriculum levels:")
for i, level in enumerate(curriculum_config.levels):
    print(f"Level {i}: {level.name}")
    print(f"  Dimensions: {level.height}x{level.width}")
    print(f"  Robots: {level.num_robots}")
    print(f"  Wall complexity: {level.edge_t_per_quadrant} edge-T, {level.central_l_per_quadrant} central-L per quadrant")
    print(f"  Max optimal length: {level.max_optimal_length}")
    print(f"  Description: {level.description}")
    print()

print(f"Current curriculum level: {curriculum_manager.get_current_level()}")
print(f"Current success rate: {curriculum_manager.get_success_rate():.3f}")
print(f"Episodes at current level: {curriculum_manager.get_episodes_at_level()}")


#%%
# Explore a specific curriculum level in detail
TARGET_LEVEL = 0  # Change this to explore different levels (0-4)

# Create environment factory for the target level
def create_level_env(level_idx: int):
    """Create an environment for a specific curriculum level."""
    level = curriculum_config.levels[level_idx]
    from env.ricochet_env import RicochetRobotsEnv
    
    return RicochetRobotsEnv(
        height=level.height,
        width=level.width,
        num_robots=level.num_robots,
        edge_t_per_quadrant=level.edge_t_per_quadrant,
        central_l_per_quadrant=level.central_l_per_quadrant,
        solver_max_depth=level.solver_max_depth,
        solver_max_nodes=level.solver_max_nodes,
        ensure_solvable=True,
        obs_mode="image",
        channels_first=True,
        render_mode=RENDER_MODE,
        seed=42  # Fixed seed for reproducibility
    )

# Create and explore the target level
level_env = create_level_env(TARGET_LEVEL)
level_info = curriculum_config.levels[TARGET_LEVEL]

print(f"Exploring Level {TARGET_LEVEL}: {level_info.name}")
print(f"Environment parameters:")
print(f"  Height: {level_env.height}, Width: {level_env.width}")
print(f"  Robots: {level_env.num_robots}")
print(f"  Edge-T per quadrant: {level_env._edge_t_per_quadrant}")
print(f"  Central-L per quadrant: {level_env._central_l_per_quadrant}")
print(f"  Max steps: {level_env.max_steps}")
print()

# Reset and display the environment
obs, info = level_env.reset()
print("Environment reset info:")
for key, value in info.items():
    print(f"  {key}: {value}")
print()

# Render the board
frame = level_env.render()
if frame is not None:
    print("Generated board:")
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


#%%
# Test level generation consistency - multiple resets at same level
LEVEL_TO_TEST = 1  # Change this to test different levels
NUM_RESETS = 5  # Number of resets to test

print(f"Testing {NUM_RESETS} resets at Level {LEVEL_TO_TEST}")
print("=" * 50)

# Create environment for testing
test_env = create_level_env(LEVEL_TO_TEST)

# Store board states for comparison
board_states = []
reset_infos = []

for reset_idx in range(NUM_RESETS):
    print(f"\nReset {reset_idx + 1}:")
    obs, info = test_env.reset(seed=42)  # Same seed for all resets
    reset_infos.append(info)
    
    # Get board state
    board = test_env.get_board()
    board_states.append(board)
    
    # Display basic info
    print(f"  Episode seed: {info.get('episode_seed', 'N/A')}")
    print(f"  Optimal length: {info.get('optimal_length', 'N/A')}")
    print(f"  Target robot: {info.get('target_robot', 'N/A')}")
    print(f"  Goal position: {board.goal_position}")
    print(f"  Robot positions: {dict(board.robot_positions)}")
    
    # Show a small portion of the board
    frame = test_env.render()
    if frame is not None:
        lines = frame.split('\n')
        print("  Board preview (first 8 lines):")
        for line in lines[:8]:
            print(f"    {line}")

# Check if all resets produced the same board
print(f"\n" + "=" * 50)
print("CONSISTENCY ANALYSIS:")
print("=" * 50)

all_same = True
for i in range(1, len(board_states)):
    if board_states[i] != board_states[0]:
        all_same = False
        break

if all_same:
    print("✅ All resets produced IDENTICAL boards (deterministic with same seed)")
else:
    print("❌ Resets produced DIFFERENT boards (non-deterministic behavior)")

# Compare specific aspects
print(f"\nDetailed comparison:")
print(f"Goal positions: {[board.goal_position for board in board_states]}")
print(f"Target robots: {[board.target_robot for board in board_states]}")
print(f"Robot positions: {[dict(board.robot_positions) for board in board_states]}")

# Check if optimal lengths are consistent
optimal_lengths = [info.get('optimal_length') for info in reset_infos]
print(f"Optimal lengths: {optimal_lengths}")
if len(set(optimal_lengths)) == 1:
    print("✅ Optimal lengths are consistent")
else:
    print("❌ Optimal lengths vary between resets")


#%%
# Test level generation with different seeds - should produce different boards
LEVEL_TO_TEST = 2  # Change this to test different levels
NUM_DIFFERENT_SEEDS = 3

print(f"Testing Level {LEVEL_TO_TEST} with different seeds")
print("=" * 50)

# Create environment for testing
test_env = create_level_env(LEVEL_TO_TEST)

# Test with different seeds
seeds_to_test = [42, 123, 456]
board_states = []

for seed in seeds_to_test:
    print(f"\nTesting with seed {seed}:")
    obs, info = test_env.reset(seed=seed)
    
    # Get board state
    board = test_env.get_board()
    board_states.append(board)
    
    # Display basic info
    print(f"  Episode seed: {info.get('episode_seed', 'N/A')}")
    print(f"  Optimal length: {info.get('optimal_length', 'N/A')}")
    print(f"  Target robot: {info.get('target_robot', 'N/A')}")
    print(f"  Goal position: {board.goal_position}")
    print(f"  Robot positions: {dict(board.robot_positions)}")
    
    # Show a small portion of the board
    frame = test_env.render()
    if frame is not None:
        if RENDER_MODE == "ascii":
            lines = frame.split('\n')
            print("  Board preview (first 6 lines):")
            for line in lines[:6]:
                print(f"    {line}")
            # print(frame)
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

# Check if different seeds produced different boards
print(f"\n" + "=" * 50)
print("SEED VARIATION ANALYSIS:")
print("=" * 50)

all_different = True
for i in range(len(board_states)):
    for j in range(i + 1, len(board_states)):
        if board_states[i] == board_states[j]:
            all_different = False
            print(f"❌ Seeds {seeds_to_test[i]} and {seeds_to_test[j]} produced identical boards")

if all_different:
    print("✅ All different seeds produced different boards (good variation)")

# Compare specific aspects
print(f"\nDetailed comparison:")
print(f"Goal positions: {[board.goal_position for board in board_states]}")
print(f"Target robots: {[board.target_robot for board in board_states]}")
print(f"Robot positions: {[dict(board.robot_positions) for board in board_states]}")

# Check optimal lengths
optimal_lengths = [info.get('optimal_length') for info in reset_infos if 'optimal_length' in info]
print(f"Optimal lengths: {optimal_lengths}")


#%%
# Visualize curriculum progression - compare multiple levels side by side
LEVELS_TO_COMPARE = [0, 1, 2, 3]  # Levels to compare

print("CURRICULUM LEVEL COMPARISON")
print("=" * 60)

# Create environments for each level
level_envs = {}
for level_idx in LEVELS_TO_COMPARE:
    level_envs[level_idx] = create_level_env(level_idx)

# Reset all environments with the same seed for fair comparison
SEED_FOR_COMPARISON = 42

for level_idx in LEVELS_TO_COMPARE:
    level_info = curriculum_config.levels[level_idx]
    env = level_envs[level_idx]
    
    print(f"\nLevel {level_idx}: {level_info.name}")
    print("-" * 40)
    
    # Reset environment
    obs, info = env.reset(seed=SEED_FOR_COMPARISON)
    
    # Display level parameters
    print(f"Dimensions: {level_info.height}x{level_info.width}")
    print(f"Robots: {level_info.num_robots}")
    print(f"Wall complexity: {level_info.edge_t_per_quadrant} edge-T, {level_info.central_l_per_quadrant} central-L")
    print(f"Max optimal length: {level_info.max_optimal_length}")
    print(f"Actual optimal length: {info.get('optimal_length', 'N/A')}")
    
    # Display board
    frame = env.render()
    if frame is not None:
        if RENDER_MODE == "ascii":
            print("Board:")
            lines = frame.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... ({len(lines) - 10} more lines)")
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

print(f"\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
for level_idx in LEVELS_TO_COMPARE:
    level_info = curriculum_config.levels[level_idx]
    env = level_envs[level_idx]
    board = env.get_board()
    
    print(f"Level {level_idx}: {level_info.name}")
    print(f"  Board size: {board.height}x{board.width}")
    print(f"  Robots: {len(board.robot_positions)}")
    print(f"  Goal: {board.goal_position}")
    print(f"  Target robot: {board.target_robot}")


#%%
# Test curriculum wrapper functionality
print("TESTING CURRICULUM WRAPPER")
print("=" * 50)

# Create curriculum wrapper
from env.curriculum import create_curriculum_wrapper

def create_base_env_factory():
    """Factory function to create base environments."""
    from env.ricochet_env import RicochetRobotsEnv
    return RicochetRobotsEnv(
        height=16, width=16, num_robots=2,
        obs_mode="image", channels_first=True,
        render_mode=RENDER_MODE
    )

# Create curriculum wrapper
curriculum_wrapper = create_curriculum_wrapper(
    base_env_factory=create_base_env_factory,
    curriculum_manager=curriculum_manager,
    verbose=True
)

print(f"Curriculum wrapper created")
print(f"Current level: {curriculum_wrapper.get_current_level()}")
print(f"Success rate: {curriculum_wrapper.get_success_rate():.3f}")
print(f"Episodes at level: {curriculum_wrapper.get_episodes_at_level()}")

# Test wrapper reset
print(f"\nTesting wrapper reset:")
obs, info = curriculum_wrapper.reset(seed=42)
print(f"Reset info: {info}")

# Test wrapper step
print(f"\nTesting wrapper step:")
action = 0  # First action
obs, reward, terminated, truncated, step_info = curriculum_wrapper.step(action)
print(f"Step result: reward={reward}, terminated={terminated}, truncated={truncated}")
print(f"Step info: {step_info}")

# Test wrapper render
frame = curriculum_wrapper.render()
if frame is not None:
    if RENDER_MODE == "ascii":
        print(f"\nRendered board:")
        lines = frame.split('\n')
        for line in lines[:8]:
            print(f"  {line}")
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

# Test curriculum stats
stats = curriculum_wrapper.get_curriculum_stats()
print(f"\nCurriculum stats: {stats}")


#%%
# Test curriculum advancement simulation
print("TESTING CURRICULUM ADVANCEMENT SIMULATION")
print("=" * 60)

# Reset curriculum manager to level 0
curriculum_manager.current_level = 0
curriculum_manager.level_start_episode = 0
curriculum_manager.success_rate_window.clear()
curriculum_manager.episode_count = 0

print(f"Starting at level: {curriculum_manager.get_current_level()}")

# Simulate episodes with varying success rates
NUM_EPISODES = 300
SUCCESS_RATE_PATTERN = [0.3, 0.5, 0.7, 0.8, 0.9]  # Success rates for different phases

print(f"\nSimulating {NUM_EPISODES} episodes...")

for episode in range(NUM_EPISODES):
    # Determine success rate based on episode number
    phase = min(episode // 60, len(SUCCESS_RATE_PATTERN) - 1)
    success_rate = SUCCESS_RATE_PATTERN[phase]
    
    # Simulate episode result
    success = np.random.random() < success_rate
    curriculum_manager.record_episode_result(success)
    
    # Check for level changes
    if curriculum_manager.current_level != curriculum_manager.get_current_level():
        print(f"Episode {episode}: Advanced to level {curriculum_manager.get_current_level()}")
    
    # Print progress every 50 episodes
    if episode % 50 == 0:
        stats = curriculum_manager.get_curriculum_stats()
        print(f"Episode {episode}: Level {stats['current_level']}, "
              f"Success rate: {stats['success_rate']:.3f}, "
              f"Episodes at level: {stats['episodes_at_level']}")

print(f"\nFinal curriculum state:")
final_stats = curriculum_manager.get_curriculum_stats()
print(f"  Current level: {final_stats['current_level']}")
print(f"  Level name: {final_stats['level_name']}")
print(f"  Success rate: {final_stats['success_rate']:.3f}")
print(f"  Episodes at level: {final_stats['episodes_at_level']}")
print(f"  Total episodes: {final_stats['total_episodes']}")


#%%
# =============================================================================
# PUZZLE BANK SYSTEM EXPLORATION CELLS
# =============================================================================
# These cells demonstrate the new bank-based curriculum system that uses
# precomputed puzzles instead of online solving during training.

#%%
# Test basic puzzle bank functionality
print("TESTING PUZZLE BANK SYSTEM")
print("=" * 50)

from env.puzzle_bank import PuzzleBank, SpecKey
from env.precompute_pipeline import PuzzleGenerator
from env.criteria_env import CriteriaFilteredEnv, PuzzleCriteria
from env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper
if 0:
    # Create a temporary bank for testing
    import tempfile

    # Use a temporary directory for testing
    temp_bank_dir = tempfile.mkdtemp(prefix="test_bank_")
    print(f"Using temporary bank directory: {temp_bank_dir}")

    # Create bank
    bank = PuzzleBank(temp_bank_dir)
    print(f"Bank created with {bank.get_puzzle_count()} puzzles")
    # Generate some test puzzles
    print("GENERATING TEST PUZZLES")
    print("=" * 30)

    # Define a simple spec for testing
    test_spec = SpecKey(
        height=6, width=6, num_robots=1,
        edge_t_per_quadrant=1, central_l_per_quadrant=1
    )

    # Create generator
    solver_config = {"max_depth": 30, "max_nodes": 20000}
    generator = PuzzleGenerator(bank, solver_config)

    # Generate a small number of puzzles
    print(f"Generating puzzles for spec: {test_spec}")
    stats = generator.generate_puzzles_for_spec(test_spec, num_puzzles=10)

    print(f"Generation complete:")
    print(f"  Requested: {stats['requested']}")
    print(f"  Generated: {stats['generated']}")
    print(f"  Solved: {stats['solved']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")

    # Check bank stats
    bank_stats = bank.get_stats()
    print(f"\nBank statistics:")
    print(f"  Total puzzles: {bank_stats['total_puzzles']}")
    print(f"  Partitions: {bank_stats['partition_count']}")

# %%
if 0:
    from env.puzzle_bank import PuzzleBank, SpecKey

    bank = PuzzleBank("./puzzle_bank")
    spec = SpecKey(height=16, width=16, num_robots=1, edge_t_per_quadrant=2, central_l_per_quadrant=2)

    # Count puzzles matching specific criteria
    puzzles = list(bank.query_puzzles(
        spec_key=spec,
        min_optimal_length=1,
        max_optimal_length=5,
        min_robots_moved=1,
        max_robots_moved=1
    ))
    count = len(puzzles)
    print(f"Puzzles matching criteria: {count}")

#%%
# Test criteria filtering
print("TESTING CRITERIA FILTERING")
print("=" * 40)
bank = PuzzleBank("./puzzle_bank")

test_spec = SpecKey(
    height=16, width=16, num_robots=4,
    edge_t_per_quadrant=2, central_l_per_quadrant=2
)
# Define criteria for filtering
criteria = PuzzleCriteria(
    spec_key=test_spec,
    min_optimal_length=1,
    max_optimal_length=4,
    min_robots_moved=1,
    max_robots_moved=1
)

print(f"Criteria: {criteria}")
print(f"  Spec: {criteria.spec_key}")
print(f"  Optimal length: {criteria.min_optimal_length}-{criteria.max_optimal_length}")
print(f"  Robots moved: {criteria.min_robots_moved}-{criteria.max_robots_moved}")

# Query puzzles matching criteria
matching_puzzles = list(bank.query_puzzles(
    spec_key=criteria.spec_key,
    min_optimal_length=criteria.min_optimal_length,
    max_optimal_length=criteria.max_optimal_length,
    min_robots_moved=criteria.min_robots_moved,
    max_robots_moved=criteria.max_robots_moved
))

print(f"\nFound {len(matching_puzzles)} puzzles matching criteria:")
for i, puzzle in enumerate(matching_puzzles[:5]):  # Show first 5
    print(f"  Puzzle {i+1}: seed={puzzle.seed}, length={puzzle.optimal_length}, robots={puzzle.robots_moved}")

#%%
# Test criteria-filtered environment
print("TESTING CRITERIA-FILTERED ENVIRONMENT")
print("=" * 50)

# Create criteria-filtered environment
criteria_env = CriteriaFilteredEnv(
    bank=bank,
    criteria=criteria,
    obs_mode="rgb_image",  # Use RGB for fixed observation shape
    channels_first=True,
    verbose=True
)

print(f"Criteria-filtered environment created")
print(f"Action space: {criteria_env.action_space}")
print(f"Observation space: {criteria_env.observation_space}")

# Test reset
print(f"\nTesting reset:")
obs, info = criteria_env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Reset info keys: {list(info.keys())}")

if 'bank_metadata' in info and info['bank_metadata'] is not None:
    metadata = info['bank_metadata']
    print(f"Puzzle metadata:")
    print(f"  Seed: {metadata.seed}")
    print(f"  Optimal length: {metadata.optimal_length}")
    print(f"  Robots moved: {metadata.robots_moved}")

# Test step
print(f"\nTesting step:")
action = 0  # First action
obs, reward, terminated, truncated, step_info = criteria_env.step(action)
print(f"Step result: reward={reward}, terminated={terminated}, truncated={truncated}")

# Test render
frame = criteria_env.render()
if frame is not None:
    print(f"Render frame shape: {frame.shape}")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.title("Criteria-Filtered Environment - RGB Render")
        plt.axis("off")
        plt.show()
    except ImportError:
        print("matplotlib not available to display RGB frame")

#%%
# Test bank-based curriculum manager
print("TESTING BANK-BASED CURRICULUM MANAGER")
print("=" * 50)

# Create curriculum levels for testing
test_curriculum_levels = [
    {
        "level": 0,
        "name": "Test Easy",
        "spec_key": SpecKey(height=4, width=4, num_robots=1, edge_t_per_quadrant=0, central_l_per_quadrant=0),
        "min_optimal_length": 1,
        "max_optimal_length": 3,
        "min_robots_moved": 1,
        "max_robots_moved": 1
    },
    {
        "level": 1,
        "name": "Test Medium",
        "spec_key": SpecKey(height=6, width=6, num_robots=1, edge_t_per_quadrant=1, central_l_per_quadrant=1),
        "min_optimal_length": 2,
        "max_optimal_length": 5,
        "min_robots_moved": 1,
        "max_robots_moved": 1
    }
]

# Generate puzzles for both levels
for level in test_curriculum_levels:
    print(f"Generating puzzles for {level['name']}...")
    generator.generate_puzzles_for_spec(level['spec_key'], num_puzzles=15)

# Create bank curriculum manager
bank_manager = create_bank_curriculum_manager(
    bank=bank,
    curriculum_levels=test_curriculum_levels,
    success_rate_threshold=0.6,  # Lower threshold for testing
    min_episodes_per_level=5,
    success_rate_window_size=10,
    advancement_check_frequency=5,
    verbose=True
)

print(f"Bank curriculum manager created")
print(f"Current level: {bank_manager.get_current_level()}")
print(f"Success rate: {bank_manager.get_success_rate():.3f}")

#%%
# Test bank curriculum wrapper
print("TESTING BANK CURRICULUM WRAPPER")
print("=" * 50)

# Create bank curriculum wrapper
bank_wrapper = BankCurriculumWrapper(
    bank=bank,
    curriculum_manager=bank_manager,
    obs_mode="rgb_image",
    channels_first=True,
    verbose=True
)

print(f"Bank curriculum wrapper created")
print(f"Current level: {bank_wrapper.get_current_level()}")
print(f"Success rate: {bank_wrapper.get_success_rate():.3f}")

# Test wrapper functionality
print(f"\nTesting wrapper reset:")
obs, info = bank_wrapper.reset()
print(f"Observation shape: {obs.shape}")
print(f"Reset info keys: {list(info.keys())}")

if 'criteria' in info:
    criteria_info = info['criteria']
    print(f"Current criteria:")
    print(f"  Spec: {criteria_info['spec_key']}")
    print(f"  Optimal length: {criteria_info['min_optimal_length']}-{criteria_info['max_optimal_length']}")
    print(f"  Robots moved: {criteria_info['min_robots_moved']}-{criteria_info['max_robots_moved']}")

# Test step
print(f"\nTesting wrapper step:")
action = 0
obs, reward, terminated, truncated, step_info = bank_wrapper.step(action)
print(f"Step result: reward={reward}, terminated={terminated}, truncated={truncated}")

# Test render
frame = bank_wrapper.render()
if frame is not None:
    print(f"Render frame shape: {frame.shape}")

#%%
# Test curriculum advancement with bank
print("TESTING BANK CURRICULUM ADVANCEMENT")
print("=" * 50)

# Reset manager to level 0
bank_manager.current_level = 0
bank_manager.level_start_episode = 0
bank_manager.success_rate_window.clear()
bank_manager.episode_count = 0

print(f"Starting at level: {bank_manager.get_current_level()}")

# Simulate episodes with varying success rates
NUM_EPISODES = 50
SUCCESS_RATE_PATTERN = [0.4, 0.7, 0.9]  # Success rates for different phases

print(f"\nSimulating {NUM_EPISODES} episodes...")

for episode in range(NUM_EPISODES):
    # Determine success rate based on episode number
    phase = min(episode // 20, len(SUCCESS_RATE_PATTERN) - 1)
    success_rate = SUCCESS_RATE_PATTERN[phase]
    
    # Simulate episode result
    success = np.random.random() < success_rate
    bank_manager.record_episode_result(success)
    
    # Check for level changes
    current_level = bank_manager.get_current_level()
    if episode > 0 and current_level != bank_manager.current_level:
        print(f"Episode {episode}: Advanced to level {current_level}")
    
    # Print progress every 10 episodes
    if episode % 10 == 0:
        stats = bank_manager.get_stats()
        print(f"Episode {episode}: Level {stats['current_level']}, "
              f"Success rate: {stats['success_rate']:.3f}, "
              f"Episodes at level: {stats['episodes_at_level']}")

print(f"\nFinal bank curriculum state:")
final_stats = bank_manager.get_stats()
print(f"  Current level: {final_stats['current_level']}")
print(f"  Level name: {final_stats['level_name']}")
print(f"  Success rate: {final_stats['success_rate']:.3f}")
print(f"  Episodes at level: {final_stats['episodes_at_level']}")
print(f"  Total episodes: {final_stats['total_episodes']}")

#%%
# Compare bank vs online curriculum
print("COMPARING BANK VS ONLINE CURRICULUM")
print("=" * 50)

# Create online curriculum manager for comparison
from env.curriculum import create_curriculum_manager, create_default_curriculum

online_config = create_default_curriculum()
online_manager = create_curriculum_manager(online_config, initial_level=0, verbose=False)

print("Online curriculum levels:")
for i, level in enumerate(online_config.levels):
    print(f"  Level {i}: {level.name} ({level.height}x{level.width}, {level.num_robots} robots)")

print(f"\nBank curriculum levels:")
for i, level in enumerate(test_curriculum_levels):
    spec = level['spec_key']
    print(f"  Level {i}: {level['name']} ({spec.height}x{spec.width}, {spec.num_robots} robots)")

print(f"\nKey differences:")
print(f"  Online: Uses solver during training (slower, more diverse)")
print(f"  Bank: Uses precomputed puzzles (faster, more controlled)")
print(f"  Online: Variable observation shapes across levels")
print(f"  Bank: Fixed observation shapes with rgb_image mode")

#%%
# Test bank sampling strategies
print("TESTING BANK SAMPLING STRATEGIES")
print("=" * 50)

# Test sampling with different criteria
test_criteria = [
    PuzzleCriteria(test_spec, min_optimal_length=1, max_optimal_length=2),
    PuzzleCriteria(test_spec, min_optimal_length=3, max_optimal_length=4),
    PuzzleCriteria(test_spec, min_optimal_length=5, max_optimal_length=6),
]

for i, criteria in enumerate(test_criteria):
    print(f"\nTesting criteria {i+1}: length {criteria.min_optimal_length}-{criteria.max_optimal_length}")
    
    # Sample puzzles using bank.get_puzzles
    puzzles = list(bank.query_puzzles(
        spec_key=criteria.spec_key,
        min_optimal_length=criteria.min_optimal_length,
        max_optimal_length=criteria.max_optimal_length,
        min_robots_moved=criteria.min_robots_moved,
        max_robots_moved=criteria.max_robots_moved,
        limit=3
    ))
    
    if puzzles:
        puzzle = puzzles[0]  # Take first puzzle
        print(f"  Found puzzle: seed={puzzle.seed}, length={puzzle.optimal_length}, robots={puzzle.robots_moved}")
        print(f"  Found {len(puzzles)} total puzzles matching criteria")
    else:
        print(f"  No puzzle found matching criteria")

#%%
# Clean up temporary bank
print("CLEANING UP")
print("=" * 20)

import shutil
shutil.rmtree(temp_bank_dir)
print(f"Removed temporary bank directory: {temp_bank_dir}")

print(f"\nBank system testing complete!")
print(f"To use in training, run:")
print(f"  python precompute_bank.py --bank_dir ./puzzle_bank --verbose")
print(f"  python train_agent.py --curriculum --use_bank_curriculum --bank_dir ./puzzle_bank --obs-mode rgb_image")

# %%
# Single level render
bank = PuzzleBank("./puzzle_bank")
test_spec = SpecKey(
    height=16, width=16, num_robots=1,
    edge_t_per_quadrant=2, central_l_per_quadrant=2
)
# Define criteria for filtering
criteria = PuzzleCriteria(
    spec_key=test_spec,
    min_optimal_length=1,
    max_optimal_length=1,
    min_robots_moved=1,
    max_robots_moved=1
)

print(f"Criteria: {criteria}")
print(f"  Spec: {criteria.spec_key}")
print(f"  Optimal length: {criteria.min_optimal_length}-{criteria.max_optimal_length}")
print(f"  Robots moved: {criteria.min_robots_moved}-{criteria.max_robots_moved}")

# Query puzzles matching criteria
matching_puzzles = list(bank.query_puzzles(
    spec_key=criteria.spec_key,
    min_optimal_length=criteria.min_optimal_length,
    max_optimal_length=criteria.max_optimal_length,
    min_robots_moved=criteria.min_robots_moved,
    max_robots_moved=criteria.max_robots_moved
))

print(f"\nFound {len(matching_puzzles)} puzzles matching criteria:")
for i, puzzle in enumerate(matching_puzzles[:1]):  # Show first 1
    print(f"  Puzzle {i+1}: seed={puzzle.seed}, length={puzzle.optimal_length}, robots={puzzle.robots_moved}")
    # Actually use the seed to generate the puzzle
    env = RicochetRobotsEnv(
        height=puzzle.spec_key.height,
        width=puzzle.spec_key.width,
        num_robots=puzzle.spec_key.num_robots,
        edge_t_per_quadrant=puzzle.spec_key.edge_t_per_quadrant,
        central_l_per_quadrant=puzzle.spec_key.central_l_per_quadrant,
        seed=puzzle.seed,
        obs_mode="image",
        channels_first=True,
        render_mode="rgb"
    )
    env.reset()
    frame = env.render()
    if frame is not None:
        # print(f"Rendered puzzle: {frame}")
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure(figsize=(4, 4))
            plt.imshow(frame)
            plt.title("Ricochet Robots - RGB Render")
            plt.axis("off")
            plt.show()
        except ImportError:
            print("matplotlib not available to display RGB frame; shape:", getattr(frame, "shape", None))




# %%
# Loop over levels and sample/render some puzzles
# Generate the curriculum levels
from env.precompute_pipeline import CurriculumSpecGenerator
curriculum_levels = CurriculumSpecGenerator.create_curriculum_specs()
print(f"Curriculum levels: {curriculum_levels}")

# Loop over levels and sample/render some puzzles
for level in curriculum_levels:
    print(f"Level: {level['name']}")
    # print(f"  Spec: {level['spec_key']}")
    # print(f"  Optimal length: {level['min_optimal_length']}-{level['max_optimal_length']}")
    # print(f"  Robots moved: {level['min_robots_moved']}-{level['max_robots_moved']}")

    bank = PuzzleBank("./puzzle_bank")
    # Define criteria for filtering
    criteria = PuzzleCriteria(
        spec_key=level['spec_key'],
        min_optimal_length=level['min_optimal_length'],
        max_optimal_length=level['max_optimal_length'],
        min_robots_moved=level['min_robots_moved'],
        max_robots_moved=level['max_robots_moved']
    )

    # print(f"Criteria: {criteria}")
    # print(f"  Spec: {criteria.spec_key}")
    # print(f"  Optimal length: {criteria.min_optimal_length}-{criteria.max_optimal_length}")
    # print(f"  Robots moved: {criteria.min_robots_moved}-{criteria.max_robots_moved}")

    # Query puzzles matching criteria
    matching_puzzles = list(bank.query_puzzles(
        spec_key=criteria.spec_key,
        min_optimal_length=criteria.min_optimal_length,
        max_optimal_length=criteria.max_optimal_length,
        min_robots_moved=criteria.min_robots_moved,
        max_robots_moved=criteria.max_robots_moved
    ))

    print(f"\nFound {len(matching_puzzles)} puzzles matching criteria:")
    for i, puzzle in enumerate(matching_puzzles[:1]):  # Show first 1
        print(f"  Puzzle {i+1}: seed={puzzle.seed}, length={puzzle.optimal_length}, robots={puzzle.robots_moved}")
        # Actually use the seed to generate the puzzle
        env = RicochetRobotsEnv(
            height=puzzle.spec_key.height,
            width=puzzle.spec_key.width,
            num_robots=puzzle.spec_key.num_robots,
            edge_t_per_quadrant=puzzle.spec_key.edge_t_per_quadrant,
            central_l_per_quadrant=puzzle.spec_key.central_l_per_quadrant,
            seed=puzzle.seed,
            obs_mode="image",
            channels_first=True,
            render_mode="rgb"
        )
        env.reset()
        frame = env.render()
        if frame is not None:
            # print(f"Rendered puzzle: {frame}")
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.figure(figsize=(4, 4))
                plt.imshow(frame)
                plt.title("Ricochet Robots - RGB Render")
                plt.axis("off")
                plt.show()
            except ImportError:
                print("matplotlib not available to display RGB frame; shape:", getattr(frame, "shape", None))

# %%
