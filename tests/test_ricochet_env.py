import pytest
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict as GymDict # Renamed to avoid conflict
from environment.ricochet_env import RicochetRobotsEnv
from environment.utils import NORTH, EAST, SOUTH, WEST, DEFAULT_BOARD_SIZE, DEFAULT_NUM_ROBOTS, TARGET_MARKER, ROBOT_COLORS

class TestRicochetRobotsEnv:

    @pytest.fixture
    def env_default(self):
        return RicochetRobotsEnv(render_mode=None) # No rendering for most tests

    @pytest.fixture
    def env_small_custom(self):
        # 5x5 board, 2 robots, specific walls
        env = RicochetRobotsEnv(board_size=5, num_robots=2, max_steps=20, use_standard_walls=False, render_mode=None)
        # Wall East of (1,1) -> (1,1) has East wall, (1,2) has West wall
        env.board.add_wall(1, 1, EAST)
        # Wall South of (2,2) -> (2,2) has South wall, (3,2) has North wall
        env.board.add_wall(2, 2, SOUTH)
        return env

    def test_env_initialization(self, env_default):
        assert env_default.height == DEFAULT_BOARD_SIZE
        assert env_default.width == DEFAULT_BOARD_SIZE
        assert env_default.num_robots == DEFAULT_NUM_ROBOTS
        assert isinstance(env_default.action_space, Discrete)
        assert env_default.action_space.n == DEFAULT_NUM_ROBOTS * 4
        
        assert isinstance(env_default.observation_space, GymDict)
        obs_space = env_default.observation_space.spaces
        assert "board_features" in obs_space
        assert "target_robot_idx" in obs_space
        
        assert isinstance(obs_space["board_features"], Box)
        assert obs_space["board_features"].shape == (DEFAULT_NUM_ROBOTS + 1, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)
        assert obs_space["board_features"].dtype == np.uint8
        
        assert isinstance(obs_space["target_robot_idx"], Discrete)
        assert obs_space["target_robot_idx"].n == DEFAULT_NUM_ROBOTS

    def test_reset(self, env_default):
        obs, info = env_default.reset(seed=42)
        
        assert env_default.observation_space.contains(obs)
        assert len(env_default.robots) == env_default.num_robots
        
        # Check robot positions are unique and within bounds
        robot_positions = set()
        for robot in env_default.robots:
            assert 0 <= robot.pos[0] < env_default.height
            assert 0 <= robot.pos[1] < env_default.width
            robot_positions.add(robot.pos)
        assert len(robot_positions) == env_default.num_robots

        assert env_default.target_pos is not None
        assert 0 <= env_default.target_pos[0] < env_default.height
        assert 0 <= env_default.target_pos[1] < env_default.width
        # Target should not be under an initial robot position if possible
        # (This test might be flaky if board is tiny and all spots taken, but unlikely for default)
        if len(robot_positions) < env_default.height * env_default.width:
             assert env_default.target_pos not in robot_positions 

        assert 0 <= obs["target_robot_idx"] < env_default.num_robots
        assert env_default.current_step == 0

        # Check observation content
        board_features = obs["board_features"]
        # Robot layers
        for i in range(env_default.num_robots):
            r, c = env_default.robots[i].pos
            assert board_features[i, r, c] == 1
            assert np.sum(board_features[i]) == 1 # Only one '1' per robot layer
        # Target layer
        tr, tc = env_default.target_pos
        assert board_features[env_default.num_robots, tr, tc] == 1
        assert np.sum(board_features[env_default.num_robots]) == 1

    def test_step_simple_move_no_collision(self, env_small_custom):
        env = env_small_custom
        # Manually set robot positions for predictable test
        # Robot 0 at (0,0), Robot 1 at (4,4)
        # Target for Robot 0 at (0,3)
        obs, _ = env.reset(seed=10)
        env.robots[0].set_position(0,0)
        env.robots[1].set_position(4,4)
        env.target_robot_idx = 0
        env.target_pos = (0,3)

        # Action: Robot 0 (idx 0) moves East (idx 1)
        # Action = 0 * 4 + 1 = 1
        action = 1 
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert env.robots[0].pos == (0, env.width -1) # Should move to the far East wall
        assert reward == -1.0
        assert not terminated
        assert not truncated
        assert env.current_step == 1
        assert next_obs["board_features"][0, 0, env.width -1] == 1 # Robot 0 new position

    def test_step_wall_collision(self, env_small_custom):
        env = env_small_custom # Has wall East of (1,1)
        obs, _ = env.reset(seed=20)
        # Robot 0 at (1,0), Robot 1 at (3,3)
        # Target for Robot 0
        env.robots[0].set_position(1,0)
        env.robots[1].set_position(3,3)
        env.target_robot_idx = 0
        env.target_pos = (4,4) # Arbitrary

        # Action: Robot 0 (idx 0) moves East (idx 1)
        # Should hit the wall at (1,1)'s East side
        action = 0 * 4 + EAST
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert env.robots[0].pos == (1,1) # Stopped by wall
        assert reward == -1.0

    def test_step_robot_collision(self, env_small_custom):
        env = env_small_custom
        obs, _ = env.reset(seed=30)
        # Robot 0 at (0,0), Robot 1 at (0,2)
        # Target for Robot 0
        env.robots[0].set_position(0,0)
        env.robots[1].set_position(0,2)
        env.target_robot_idx = 0
        env.target_pos = (4,4)

        # Action: Robot 0 (idx 0) moves East (idx 1)
        # Should stop at (0,1) because Robot 1 is at (0,2)
        action = 0 * 4 + EAST
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.robots[0].pos == (0,1) # Stopped by Robot 1
        assert reward == -1.0

    def test_goal_condition(self, env_small_custom):
        env = env_small_custom
        obs, _ = env.reset(seed=40)
        # Robot 0 at (0,0), Robot 1 at (4,4)
        # Target for Robot 0 is (0,1) (one step away)
        env.robots[0].set_position(0,0)
        env.robots[1].set_position(4,4)
        env.target_robot_idx = 0
        env.target_pos = (0,4) # For simplicity, assume it can reach in one step

        # Manually place robot 0 next to target to ensure it reaches
        env.robots[0].set_position(0,0) # Will move to (0,1) if moving East
        
        # Action: Robot 0 (idx 0) moves East (idx 1)
        action = 0 * 4 + EAST
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert env.robots[0].pos == (0,4) # Reached target
        assert terminated == True
        assert reward == -1.0 + 50.0 # Step penalty + goal reward
        assert env.current_step == 1

    def test_max_steps_truncation(self, env_small_custom):
        env = env_small_custom
        env.max_steps = 2 # Very short episode
        obs, _ = env.reset(seed=50)
        env.target_pos = (env.height -1, env.width -1) # Far away target

        # Take 2 steps that don't solve
        _, _, t1, tr1, _ = env.step(0) # Robot 0, North
        assert not t1 and not tr1
        assert env.current_step == 1
        
        _, reward, terminated, truncated, _ = env.step(0) # Robot 0, North again
        assert not terminated # Did not solve
        assert truncated == True # Max steps reached
        assert env.current_step == 2
        assert reward == -1.0 -10.0 # Step penalty + truncation penalty

    def test_render_ansi(self, env_small_custom):
        env = env_small_custom
        env.render_mode = "ansi" # Set for this test
        obs, _ = env.reset(seed=60)
        # Robot 0 at (1,1), Robot 1 at (3,3)
        # Target for Robot 0 at (0,0)
        env.robots[0].set_position(1,1)
        env.robots[1].set_position(3,3)
        env.target_robot_idx = 0
        env.target_pos = (0,0)
        
        output = env.render()
        assert isinstance(output, str)
        assert TARGET_MARKER in output # Target 'T' should be visible
        assert ROBOT_COLORS[0] in output # Robot 0 char 'R'
        assert ROBOT_COLORS[1] in output # Robot 1 char 'G'
        assert "Step: 0" in output
        assert "Target Robot: R (R) to (0, 0)" in output # Check target info line

        # Test a step and render again
        env.step(0) # Robot 0, North
        output_after_step = env.render()
        assert "Step: 1" in output_after_step

    def test_get_test_board_env(self):
        env = RicochetRobotsEnv.get_test_board_env(size=6, num_robots=3)
        assert env.height == 6
        assert env.num_robots == 3
        assert env.board.has_wall(1,1,EAST) # Check one of the custom walls
        assert env.board.has_wall(1,2,WEST)
        assert env.board.has_wall(2,2,SOUTH)
        assert env.board.has_wall(3,2,NORTH) 