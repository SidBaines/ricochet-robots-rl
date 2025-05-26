# Class for a simple Ricochet Robots environment in which the target is one step away from the robot

from environment.ricochet_env import RicochetRobotsEnv
from environment.custom_difficulty_envs import RicochetRobotsEnvCustomDifficulty
from environment.utils import DIRECTIONS
from typing import Optional, Tuple, Dict
import numpy as np
from environment.board import Board


class RicochetRobotsEnvOneStepAway(RicochetRobotsEnv):
    def __init__(self, 
                 board_size=5, 
                 num_robots=2, 
                 max_steps=10, 
                 num_edge_walls_per_quadrant=0, 
                 num_floating_walls_per_quadrant=0,
                 seed=None,
                 **kwargs):
        super().__init__(
            board_size=board_size,
            num_robots=num_robots,
            max_steps=max_steps,
            num_edge_walls_per_quadrant=num_edge_walls_per_quadrant,
            num_floating_walls_per_quadrant=num_floating_walls_per_quadrant,
            seed=seed,
            **kwargs
        )
        self._seed = seed
        self._set_seed(seed)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self._set_seed(self._seed)
        '''
        Reset the environment to create a one-step solvable puzzle.
        '''
        obs_dict, info = super().reset(seed, options)
        # Move the target to be one step away from the robot. This involves checking directions until 
        # we find one where the robot actually moves, and then putting the target to wherever it would 
        # end up.
        orig_pos = self.robots[self.target_robot_idx].pos
        new_pos = orig_pos
        attempts = 0
        dirn = np.random.randint(4)
        while (new_pos == orig_pos) and (attempts < 6):
            dirn = (dirn+1)%4
            self.step(self.target_robot_idx*4 + dirn)
            new_pos = self.robots[self.target_robot_idx].pos
            attempts += 1
            self.robots[self.target_robot_idx].set_position(orig_pos[0], orig_pos[1])
        if attempts < 6:
            self.target_pos = new_pos
            self.current_step = 0
            return self._get_obs(), self._get_info()
        else:
            # TODO At the moment, this just calls the reset again, which could go into a loop if the environment is complicated enough, so should maybe fix this.
            # for _ in range(10):
            #     print("Failed to find a valid move for the target robot; need to move robot around to make space")
            # self.render_mode = "human"
            # self.render()
            # This means that the robot cannot move anywhere, so we need to move something (either a 
            # wall or a robot). In fact, it should not be a wall, because we should never have more 
            # than 2 walls surrounding a single square. So look for robots around the target robot.
            robot_moved = False
            for direction in range(4):
                new_pos = self.robots[self.target_robot_idx].pos + DIRECTIONS[direction]
                if self.board.is_valid_position(new_pos[0], new_pos[1]):
                    # Check if there is a robot in the way
                    robot_in_pos = self._get_robot_at(new_pos[0], new_pos[1])
                    if robot_in_pos is not None:
                        # Found a robot in the way, so we can swap the target robot with this one
                        old_pos = self.robots[self.target_robot_idx].pos
                        new_pos = robot_in_pos.pos
                        self.robots[self.target_robot_idx].set_position(old_pos[0], old_pos[1])
                        self.robots[robot_in_pos.id].set_position(new_pos[0], new_pos[1])
                        robot_moved = True
                        break
            assert robot_moved, "Should have moved a robot; need to check how we got here"
            # Now call reset again to ensure the target is one step away from the robot
            return self.reset(seed, options)
    
class RicochetRobotsEnvCornerTarget(RicochetRobotsEnvCustomDifficulty):
    """
    Corner Target Environment: Only central block of walls, with target adjacent to a corner.
    Contains at least two robots, with the target positioned in a square adjacent to one of the
    board corners.
    """
    
    def __init__(self, board_size: int = 8, num_robots: int = 2, max_steps: int = 20, **kwargs):
        # Ensure at least 2 robots
        num_robots = max(2, num_robots)
        
        super().__init__(
            board_size=board_size,
            num_robots=num_robots,
            max_steps=max_steps,
            use_standard_walls=False,  # No standard walls
            num_edge_walls_per_quadrant=0,  # No edge walls
            num_floating_walls_per_quadrant=0,  # No floating walls
            **kwargs
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset to create a puzzle with only central walls and target adjacent to a corner."""
        obs_dict, info = super().reset(seed, options)
        
        # Clear all walls except perimeter
        self.board = Board(self.height, self.width)  # This sets up perimeter walls
        
        # Add only the central block walls
        self.board.add_middle_blocked_walls()
        
        # Define corner-adjacent positions
        corner_adjacent_positions = [
            # Adjacent to top-left corner (0,0)
            (0, 1), (1, 0),
            # Adjacent to top-right corner (0, width-1)
            (0, self.width-2), (1, self.width-1),
            # Adjacent to bottom-left corner (height-1, 0)
            (self.height-2, 0), (self.height-1, 1),
            # Adjacent to bottom-right corner (height-1, width-1)
            (self.height-2, self.width-1), (self.height-1, self.width-2)
        ]
        
        # Filter out positions that might be invalid for very small boards
        valid_corner_adjacent = []
        for pos in corner_adjacent_positions:
            if (0 <= pos[0] < self.height and 0 <= pos[1] < self.width):
                valid_corner_adjacent.append(pos)
        
        if not valid_corner_adjacent:
            raise ValueError("Board is too small to place target adjacent to corner")
        
        # Choose a random corner-adjacent position for the target
        self.target_pos = valid_corner_adjacent[self.np_random.integers(len(valid_corner_adjacent))]
        
        # Choose a random robot as the target robot
        self.target_robot_idx = self.np_random.integers(self.num_robots)
        
        # Make sure robots are not placed on the target position
        for i, robot in enumerate(self.robots):
            if robot.pos == self.target_pos:
                # Find a new position for this robot
                available_positions = [(r, c) for r in range(self.height) for c in range(self.width) 
                                      if (r, c) != self.target_pos and 
                                      (r, c) not in [other_robot.pos for other_robot in self.robots]]
                
                if available_positions:
                    new_pos = available_positions[self.np_random.integers(len(available_positions))]
                    robot.set_position(new_pos[0], new_pos[1])
        
        self.current_step = 0
        return self._get_obs(), self._get_info()