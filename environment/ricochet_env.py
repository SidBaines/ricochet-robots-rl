import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import random
import torch
import json


from .board import Board
from .robot import Robot
from .utils import (
    DIRECTIONS, NORTH, EAST, SOUTH, WEST,
    ROBOT_COLORS, TARGET_MARKER, EMPTY_CELL,
    WALL_HORIZONTAL, WALL_VERTICAL, CORNER,
    DEFAULT_BOARD_SIZE, DEFAULT_NUM_ROBOTS
)

def simulate_robot_move(
    robot_pos: Tuple[int, int],
    robot_id: int,
    direction_idx: int,
    walls: np.ndarray,
    robot_positions: List[Tuple[int, int]],
    robot_ids: List[int],
    board_height: int,
    board_width: int
) -> Tuple[int, int]:
    """Simulate moving a robot in a given direction and return its final position.
    
    Args:
        robot_pos: Current (row, col) position of the robot to move
        robot_id: ID of the robot to move
        direction_idx: Direction to move in (0=NORTH, 1=EAST, 2=SOUTH, 3=WEST)
        walls: 4D numpy array of walls (height, width, 4) where last dimension is [N,E,S,W]
        robot_positions: List of (row, col) positions of all robots
        robot_ids: List of IDs for all robots
        board_height: Height of the board
        board_width: Width of the board
        
    Returns:
        Tuple[int, int]: The final (row, col) position the robot would end up at
    """
    dr, dc = DIRECTIONS[direction_idx]
    current_r, current_c = robot_pos
    
    # Create a mapping of positions to robot IDs for quick lookup
    pos_to_robot = {pos: rid for pos, rid in zip(robot_positions, robot_ids)}
    
    # Simulate movement until collision
    while True:
        if walls[current_r, current_c, direction_idx]:
            break # Hit a wall from current cell in chosen direction

        next_r, next_c = current_r + dr, current_c + dc

        # Check bounds
        if not (0 <= next_r < board_height and 0 <= next_c < board_width):
            break

        # Check for collision with another robot
        if (next_r, next_c) in pos_to_robot and pos_to_robot[(next_r, next_c)] != robot_id:
            break # Hit another robot

        # If no collision, move one step
        current_r, current_c = next_r, next_c
        
    return current_r, current_c

class RicochetRobotsEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self,
                 board_size: Union[int, Tuple[int, int]] = DEFAULT_BOARD_SIZE,
                 num_robots: int = DEFAULT_NUM_ROBOTS,
                 max_steps: int = 200,
                 board_walls_config: Optional[List[Tuple[Tuple[int, int], int]]] = None,
                 use_standard_walls: bool = True,
                 num_edge_walls_per_quadrant: Optional[int] = None,
                 num_floating_walls_per_quadrant: Optional[int] = None,
                 display_step: bool = False,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()

        self.display_step = display_step
        if isinstance(board_size, int):
            self.height, self.width = board_size, board_size
        else:
            self.height, self.width = board_size
        
        self.num_robots = min(num_robots, len(ROBOT_COLORS)) # Max robots based on available colors
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.board = Board(self.height, self.width)

        if board_walls_config is not None:
            for (r, c), direction_idx in board_walls_config:
                self.board.add_wall(r, c, direction_idx)
            # If specific walls are given, standard walls might be implicitly overridden or complemented.
            # We might decide not to call add_standard_ricochet_walls if board_walls_config is present.
            # For now, let's assume board_walls_config defines the *entire* non-perimeter wall setup.
            # So, if board_walls_config is given, we might skip use_standard_walls.
            # However, the current Board.add_standard_ricochet_walls is idempotent for its specific walls.
            if use_standard_walls: # Allow standard walls to be added alongside config if desired
                 self.board.add_standard_ricochet_walls()
        elif use_standard_walls:
            self.board.add_standard_ricochet_walls()
            
        
        self.board.add_middle_blocked_walls() # Always add the middle blocked walls
        
        # Now, apply random wall generation if parameters are provided
        if num_edge_walls_per_quadrant is not None and num_edge_walls_per_quadrant > 0 or \
           num_floating_walls_per_quadrant is not None and num_floating_walls_per_quadrant > 0:
            # Ensure defaults if one is None but the other is set
            edge_walls_to_gen = num_edge_walls_per_quadrant if num_edge_walls_per_quadrant is not None else 0
            floating_walls_to_gen = num_floating_walls_per_quadrant if num_floating_walls_per_quadrant is not None else 0
            
            self.board.generate_random_walls(
                num_edge_walls_per_quadrant=edge_walls_to_gen,
                num_floating_walls_per_quadrant=floating_walls_to_gen,
                rng=self._np_random  # Pass the seeded RNG
            )

        self.robots: List[Robot] = [] # Will be populated in reset
        self.target_pos: Optional[Tuple[int, int]] = None
        self.target_robot_idx: Optional[int] = None
        self.current_step = 0

        # Action space: num_robots * 4 directions
        # Action = robot_idx * 4 + direction_idx
        self.action_space = spaces.Discrete(self.num_robots * 4)

        # Observation space:
        # - board_features: (num_robots + 1 + 4, height, width) binary tensor
        #   - Channel 0 to num_robots-1: position of robot i
        #   - Channel num_robots: position of the target
        #   - Channels num_robots+1 to num_robots+4: wall information (N, E, S, W)
        # - target_robot_idx: Discrete(num_robots) indicating which robot is the active target
        self.observation_space = spaces.Dict({
            "board_features": spaces.Box(
                low=0, high=1,
                shape=(self.num_robots + 1 + 4, self.height, self.width),
                dtype=np.uint8
            ),
            "target_robot_idx": spaces.Discrete(self.num_robots)
        })
        
        self.np_random = None # Will be seeded in reset
        self._seed = seed
        self._set_seed(seed)

    def _set_seed(self, seed):
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
        except Exception:
            pass

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # Create observation with additional channels for walls
        # Channels: [robot_0, robot_1, ..., target, wall_N, wall_E, wall_S, wall_W]
        board_features = np.zeros((self.num_robots + 1 + 4, self.height, self.width), dtype=np.uint8)
        
        # Set robot positions (first num_robots channels)
        for i, robot in enumerate(self.robots):
            r, c = robot.pos
            board_features[i, r, c] = 1
        
        # Set target position (channel num_robots)
        if self.target_pos:
            tr, tc = self.target_pos
            board_features[self.num_robots, tr, tc] = 1
        
        # Set wall information (channels num_robots+1 to num_robots+4)
        # For each direction (N, E, S, W), create a binary mask where 1 indicates a wall
        wall_channel_offset = self.num_robots + 1
        
        # Add walls for each direction
        for direction in range(4):  # NORTH, EAST, SOUTH, WEST
            for r in range(self.height):
                for c in range(self.width):
                    if self.board.has_wall(r, c, direction):
                        board_features[wall_channel_offset + direction, r, c] = 1
        
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
        if seed is not None:
            self._seed = seed
        self._set_seed(self._seed)

        self.current_step = 0
        
        # Initialize robots at unique random positions
        # However, robots cannot be placed inside the middle blocked out 2x2 square. 
        # Centre square is defined as (self.height // 2 - 1, self.width // 2 - 1) to
        # (self.height // 2, self.width // 2)
        blocked_positions = [
            (self.height // 2 - 1, self.width // 2 - 1),
            (self.height // 2 - 1, self.width // 2),
            (self.height // 2, self.width // 2 - 1),
            (self.height // 2, self.width // 2)
        ]
        all_possible_positions = [(r, c) for r in range(self.height) for c in range(self.width)]
        all_possible_positions = [p for p in all_possible_positions if p not in blocked_positions]
        
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
        if self.display_step:
            if self.render_mode == "human":
                self.render()
        return self._get_obs(), self._get_info()

    def _get_robot_at(self, r: int, c: int) -> Optional[Robot]:
        for robot in self.robots:
            if robot.pos == (r, c):
                return robot
        return None

    def _simulate_robot_move(self, robot_idx: int, direction_idx: int) -> Tuple[int, int]:
        """Simulate moving a robot in a given direction and return its final position.
        
        Args:
            robot_idx: Index of the robot to move
            direction_idx: Direction to move in (0=NORTH, 1=EAST, 2=SOUTH, 3=WEST)
            
        Returns:
            Tuple[int, int]: The final (row, col) position the robot would end up at
        """
        robot = self.robots[robot_idx]
        robot_positions = [r.pos for r in self.robots]
        robot_ids = [r.id for r in self.robots]
        
        return simulate_robot_move(
            robot_pos=robot.pos,
            robot_id=robot.id,
            direction_idx=direction_idx,
            walls=self.board.walls,
            robot_positions=robot_positions,
            robot_ids=robot_ids,
            board_height=self.height,
            board_width=self.width
        )

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        robot_to_move_idx = action // 4
        direction_idx = action % 4
        
        # Get final position from simulation
        final_r, final_c = self._simulate_robot_move(robot_to_move_idx, direction_idx)
        
        # Actually move the robot
        self.robots[robot_to_move_idx].set_position(final_r, final_c)
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
        if self.display_step:
            print(f"Step: {self.current_step}")
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
        env = RicochetRobotsEnv(
            board_size=size, 
            num_robots=num_robots, 
            max_steps=max_steps, 
            use_standard_walls=False,
            # Example of using new params for test env:
            # num_edge_walls_per_quadrant=1 
        )
        # Add a simple wall
        # Wall to the East of (1,1) / West of (1,2)
        env.board.add_wall(1, 1, EAST)
        # Wall to the South of (2,2) / North of (3,2)
        env.board.add_wall(2, 2, SOUTH)
        return env 

    def save_env(self, filepath: str):
        """Save the full environment state to a JSON file."""
        def to_pyint_tuple(t):
            return (int(t[0]), int(t[1]))
        data = {
            'board': self.board.to_dict(),
            'robots': [to_pyint_tuple(robot.pos) for robot in self.robots],
            'target_pos': to_pyint_tuple(self.target_pos) if self.target_pos is not None else None,
            'target_robot_idx': int(self.target_robot_idx) if self.target_robot_idx is not None else None,
            'current_step': int(self.current_step),
            'max_steps': int(self.max_steps),
            'num_robots': int(self.num_robots),
            'height': int(self.height),
            'width': int(self.width)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_env(self, filepath: str):
        """Load the full environment state from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.height = data['height']
        self.width = data['width']
        self.num_robots = data['num_robots']
        self.max_steps = data.get('max_steps', 200)
        self.board = Board.from_dict(data['board'])
        self.robots = [Robot(robot_id=i, color=ROBOT_COLORS[i], initial_pos=tuple(pos)) for i, pos in enumerate(data['robots'])]
        self.target_pos = tuple(data['target_pos']) if data['target_pos'] is not None else None
        self.target_robot_idx = data['target_robot_idx']
        self.current_step = data.get('current_step', 0)
        # Rebuild action/observation spaces if needed
        self.action_space = spaces.Discrete(self.num_robots * 4)
        self.observation_space = spaces.Dict({
            "board_features": spaces.Box(
                low=0, high=1,
                shape=(self.num_robots + 1 + 4, self.height, self.width),
                dtype=np.uint8
            ),
            "target_robot_idx": spaces.Discrete(self.num_robots)
        }) 