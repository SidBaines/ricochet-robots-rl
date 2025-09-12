# Package for Ricochet Robots environment
from .ricochet_env import RicochetRobotsEnv, FixedLayout  # re-export
from .curriculum import CurriculumWrapper, CurriculumConfig, create_curriculum_wrapper, create_default_curriculum  # re-export

# Deterministic milestone layouts for training sanity checks
def fixed_layout_v0_one_move() -> FixedLayout:
    """A 4x4 board where target robot reaches goal with a single RIGHT move.

    Layout:
    - Borders only; robot 0 starts at (0,1); goal at (0,3); target_robot=0
    - Moving RIGHT once slides to (0,3) with goal.
    """
    import numpy as np  # local import to avoid mandatory dependency on import time

    H, W = 4, 4
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    robots = {0: (0, 1)}
    goal = (0, 3)
    return FixedLayout(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=0)


def fixed_layouts_v1_four_targets() -> list[FixedLayout]:
    """Return four 5x5 layouts where optimal is to slide the target robot in one of four directions.

    Each layout has borders only; minimal plan is one action in the specified direction.
    """
    import numpy as np

    layouts: list[FixedLayout] = []

    H, W = 5, 5

    # UP: start (3,2) -> up stops at row 1 due to h_wall at (1,2); goal (1,2)
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    h_walls[1, 2] = True
    robots = {0: (3, 2)}
    goal = (1, 2)
    layouts.append(FixedLayout(height=H, width=W, h_walls=h_walls.copy(), v_walls=v_walls.copy(), robot_positions=dict(robots), goal_position=goal, target_robot=0))

    # DOWN: start (1,2) -> down stops at row 3 due to h_wall at (4,2); goal (3,2)
    h_w2 = np.zeros((H + 1, W), dtype=bool)
    v_w2 = np.zeros((H, W + 1), dtype=bool)
    h_w2[0, :] = True
    h_w2[H, :] = True
    v_w2[:, 0] = True
    v_w2[:, W] = True
    h_w2[4, 2] = True
    robots2 = {0: (1, 2)}
    goal2 = (3, 2)
    layouts.append(FixedLayout(height=H, width=W, h_walls=h_w2, v_walls=v_w2, robot_positions=robots2, goal_position=goal2, target_robot=0))

    # LEFT: start (2,3) -> left stops at col 1 due to v_wall at (2,1); goal (2,1)
    h_w3 = np.zeros((H + 1, W), dtype=bool)
    v_w3 = np.zeros((H, W + 1), dtype=bool)
    h_w3[0, :] = True
    h_w3[H, :] = True
    v_w3[:, 0] = True
    v_w3[:, W] = True
    v_w3[2, 1] = True
    robots3 = {0: (2, 3)}
    goal3 = (2, 1)
    layouts.append(FixedLayout(height=H, width=W, h_walls=h_w3, v_walls=v_w3, robot_positions=robots3, goal_position=goal3, target_robot=0))

    # RIGHT: start (2,1) -> right stops at col 3 due to v_wall at (2,4); goal (2,3)
    h_w4 = np.zeros((H + 1, W), dtype=bool)
    v_w4 = np.zeros((H, W + 1), dtype=bool)
    h_w4[0, :] = True
    h_w4[H, :] = True
    v_w4[:, 0] = True
    v_w4[:, W] = True
    v_w4[2, 4] = True
    robots4 = {0: (2, 1)}
    goal4 = (2, 3)
    layouts.append(FixedLayout(height=H, width=W, h_walls=h_w4, v_walls=v_w4, robot_positions=robots4, goal_position=goal4, target_robot=0))

    return layouts
