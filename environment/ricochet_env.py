import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any

from .board import Board
from .robot import Robot
from .utils import (
    DIRECTIONS, NORTH, EAST, SOUTH, WEST,
    ROBOT_COLORS, TARGET_MARKER, EMPTY_CELL,
    WALL_HORIZONTAL, WALL_VERTICAL, CORNER,
    DEFAULT_BOARD_SIZE, DEFAULT_NUM_ROBOTS
)

class RicochetRobotsEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self,
                 board_size: int = DEFAULT_BOARD_SIZE,
                 num_robots: int = DEFAULT_NUM_ROBOTS,
                 max_steps: int = 50,
                 board_walls_config: Optional[List[Tuple[Tuple[int, int], int]]] = None,
                 use_standard_walls: bool = True,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.height = board_size
        self.width = board_size
        self.num_robots = min(num_robots, len(ROBOT_COLORS)) # Cap at available colors
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.board = Board(self.height, self.width)
        if use_standard_walls and self.height == 16 and self.width == 16:
            self.board.add_standard_ricochet_walls() # Add some predefined walls
        if board_walls_config:
            for (r, c), direction_idx in board_walls_config:
                self.board.add_wall(r, c, direction_idx)

        self.robots: List[Robot] = [] # Will be populated in reset
        self.target_pos: Optional[Tuple[int, int]] = None
        self.target_robot_idx: Optional[int] = None
        self.current_step = 0

        # Action space: num_robots * 4 directions
        # Action = robot_idx * 4 + direction_idx
        self.action_space = spaces.Discrete(self.num_robots * 4)

        # Observation space:
        # - board_features: (num_robots + 1, height, width) binary tensor
        #   - Channel 0 to num_robots-1: position of robot i
        #   - Channel num_robots: position of the target
        # - target_robot_idx: Discrete(num_robots) indicating which robot is the active target
        self.observation_space = spaces.Dict({
            "board_features": spaces.Box(
                low=0, high=1,
                shape=(self.num_robots + 1, self.height, self.width),
                dtype=np.uint8
            ),
            "target_robot_idx": spaces.Discrete(self.num_robots)
        })
        
        self.np_random = None # Will be seeded in reset

    def _get_obs(self) -> Dict[str, Union[np.ndarray, int]]:
        board_features = np.zeros((self.num_robots + 1, self.height, self.width), dtype=np.uint8)
        for i, robot in enumerate(self.robots):
            r, c = robot.pos
            board_features[i, r, c] = 1
        
        if self.target_pos:
            tr, tc = self.target_pos
            board_features[self.num_robots, tr, tc] = 1
            
        return {
            "board_features": board_features,
            "target_robot_idx": self.target_robot_idx if self.target_robot_idx is not None else 0
        }

    def _get_info(self) -> Dict[str, Any]:
        robot_positions = [robot.pos for robot in self.robots]
        return {
            "current_step": self.current_step,
            "robot_positions": robot_positions,
            "target_pos": self.target_pos,
            "target_robot_id_val": self.target_robot_idx # Using a different key to avoid gym warning
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed) # Important for seeding self.np_random

        self.current_step = 0
        
        # Initialize robots at unique random positions
        all_possible_positions = [(r, c) for r in range(self.height) for c in range(self.width)]
        self.np_random.shuffle(all_possible_positions)
        
        self.robots = []
        occupied_positions = set()
        for i in range(self.num_robots):
            pos = all_possible_positions.pop()
            self.robots.append(Robot(robot_id=i, color=ROBOT_COLORS[i], initial_pos=pos))
            occupied_positions.add(pos)

        # Select target robot and target position (not under another robot)
        self.target_robot_idx = self.np_random.integers(self.num_robots)
        
        available_target_positions = [p for p in all_possible_positions if p not in occupied_positions]
        if not available_target_positions: # Should be rare, fallback if all cells taken
             available_target_positions = all_possible_positions # Allow target under robot if no other choice
        
        self.target_pos = self.np_random.choice(available_target_positions) if available_target_positions else all_possible_positions[0]
        # Ensure target_pos is a tuple of ints
        if isinstance(self.target_pos, np.ndarray):
             self.target_pos = tuple(self.target_pos.tolist())


        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def _get_robot_at(self, r: int, c: int) -> Optional[Robot]:
        for robot in self.robots:
            if robot.pos == (r, c):
                return robot
        return None

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        robot_to_move_idx = action // 4
        direction_idx = action % 4
        
        moving_robot = self.robots[robot_to_move_idx]
        dr, dc = DIRECTIONS[direction_idx]

        current_r, current_c = moving_robot.pos
        
        # Simulate movement until collision
        while True:
            if self.board.has_wall(current_r, current_c, direction_idx):
                break # Hit a wall from current cell in chosen direction

            next_r, next_c = current_r + dr, current_c + dc

            # Check bounds (should be covered by perimeter walls, but good for safety)
            if not (0 <= next_r < self.height and 0 <= next_c < self.width):
                break # Should not happen if perimeter walls are set up correctly

            # Check for collision with another robot at next_r, next_c
            other_robot = self._get_robot_at(next_r, next_c)
            if other_robot is not None and other_robot.id != moving_robot.id:
                break # Hit another robot

            # If no collision, move one step
            current_r, current_c = next_r, next_c
        
        moving_robot.set_position(current_r, current_c)
        self.current_step += 1

        # Check for goal condition
        terminated = False
        reward = -1.0  # Penalty for each step

        active_target_robot = self.robots[self.target_robot_idx]
        if active_target_robot.pos == self.target_pos:
            terminated = True
            reward += 50.0 # Bonus for reaching the target (can be tuned)

        # Check for truncation (max steps)
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated: # Penalize if max steps reached without solving
                 reward -= 10.0 

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[Union[str, np.ndarray]]:
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        grid_repr = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        for robot in self.robots:
            r, c = robot.pos
            grid_repr[r][c] = robot.color_char
        
        if self.target_pos:
            tr, tc = self.target_pos
            # If target is under a robot, show robot, otherwise show target marker
            if grid_repr[tr][tc] == ' ':
                grid_repr[tr][tc] = TARGET_MARKER
            else: # Target is under a robot, maybe indicate with lowercase if robot is target
                if self.robots[self.target_robot_idx].pos == self.target_pos:
                     grid_repr[tr][tc] = grid_repr[tr][tc].lower()


        output_str = ""
        for r in range(self.height):
            # Top border of the row
            row_str_top = CORNER
            for c in range(self.width):
                row_str_top += WALL_HORIZONTAL if self.board.has_wall(r, c, NORTH) else "   "
                row_str_top += CORNER
            output_str += row_str_top + "\n"

            # Cell contents and right walls
            row_str_mid = ""
            for c in range(self.width):
                row_str_mid += WALL_VERTICAL if self.board.has_wall(r, c, WEST) else " "
                row_str_mid += f" {grid_repr[r][c]} "
            row_str_mid += WALL_VERTICAL if self.board.has_wall(r, self.width - 1, EAST) else " " # Last cell's east wall
            output_str += row_str_mid + "\n"

        # Bottom border of the grid (south walls of the last row)
        row_str_bottom = CORNER
        for c in range(self.width):
            row_str_bottom += WALL_HORIZONTAL if self.board.has_wall(self.height - 1, c, SOUTH) else "   "
            row_str_bottom += CORNER
        output_str += row_str_bottom + "\n"
        
        if self.target_robot_idx is not None:
            output_str += f"Target Robot: {ROBOT_COLORS[self.target_robot_idx]} ({self.robots[self.target_robot_idx].color_char}) to {self.target_pos}\n"
        output_str += f"Step: {self.current_step}\n"


        if self.render_mode == "human":
            print(output_str)
        elif self.render_mode == "ansi":
            return output_str
        # For "rgb_array", one would typically use matplotlib or similar
        # For now, we'll skip implementing rgb_array fully.

    def close(self):
        pass # Cleanup if any resources were allocated (e.g., pygame window)

    # Helper to create a board with some specific walls for testing
    @staticmethod
    def get_test_board_env(size=5, num_robots=2, max_steps=20):
        env = RicochetRobotsEnv(board_size=size, num_robots=num_robots, max_steps=max_steps, use_standard_walls=False)
        # Add a simple wall
        # Wall to the East of (1,1) / West of (1,2)
        env.board.add_wall(1, 1, EAST)
        # Wall to the South of (2,2) / North of (3,2)
        env.board.add_wall(2, 2, SOUTH)
        return env 