# Class for a simple Ricochet Robots environment in which the target is one step away from the robot

from environment.ricochet_env import RicochetRobotsEnv
from environment.utils import DIRECTIONS
from typing import Optional, Tuple, Dict
import numpy as np


class RicochetRobotsEnvOneStepAway(RicochetRobotsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
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
    
    