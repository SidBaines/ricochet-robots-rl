import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

import argparse

import numpy as np

from src.env.ricochet_env import RicochetRobotsEnv, FixedLayout
from src.env.ricochet_core import UP, DOWN, LEFT, RIGHT

ACTIONS_HELP = """
Enter an action:
  u/d/l/r  -> move current robot up/down/left/right
  s <id>   -> switch current robot (0-based)
  n        -> no-op (if enabled)
  q        -> quit
"""

def make_fixed_env() -> RicochetRobotsEnv:
    H, W = 5, 5
    walls_h = np.zeros((H + 1, W), dtype=bool)
    walls_v = np.zeros((H, W + 1), dtype=bool)
    walls_h[0, :] = True
    walls_h[H, :] = True
    walls_v[:, 0] = True
    walls_v[:, W] = True
    walls_h[1, 2] = True
    walls_v[0, 3] = True
    robots = {0: (3, 2), 1: (4, 4)}
    goal = (0, 2)
    layout = FixedLayout(height=H, width=W, h_walls=walls_h, v_walls=walls_v, robot_positions=robots, goal_position=goal, target_robot=0)
    env = RicochetRobotsEnv(fixed_layout=layout, include_noop=True, step_penalty=0.0, goal_reward=1.0, max_steps=50)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ricochet Robots CLI Demo")
    _ = parser.parse_args(argv)

    env = make_fixed_env()
    _, info = env.reset()
    current_robot = 0
    print(f"Initial board (target={info.get('target_robot')}):")
    print(env.render())
    print(ACTIONS_HELP)

    def to_action(robot_id: int, dir_code: int) -> int:
        return robot_id * 4 + dir_code

    while True:
        try:
            s = input(f"[robot {current_robot} target={info.get('target_robot')}] Action (u/d/l/r/s <id>/n/q): ").strip().lower()
        except EOFError:
            print()
            break
        if s == 'q':
            break
        if s.startswith('s'):
            parts = s.split()
            if len(parts) == 2 and parts[1].isdigit():
                new_id = int(parts[1])
                current_robot = max(0, min(new_id, getattr(env, 'num_robots', 1) - 1))
            else:
                print("Usage: s <robot_id>")
            continue
        if s == 'n':
            action = env.action_space.n - 1  # type: ignore[attr-defined]
        elif s == 'u':
            action = to_action(current_robot, UP)
        elif s == 'd':
            action = to_action(current_robot, DOWN)
        elif s == 'l':
            action = to_action(current_robot, LEFT)
        elif s == 'r':
            action = to_action(current_robot, RIGHT)
        else:
            print("Unknown input. Try again.")
            continue

        _, reward, terminated, truncated, info = env.step(action)
        print(env.render())
        print(f"reward={reward:.3f} terminated={terminated} truncated={truncated} info={info}")
        if terminated or truncated:
            print("Episode ended. Resetting.\n")
            _, info = env.reset()
            print(env.render())


if __name__ == "__main__":
    sys.exit(main())
