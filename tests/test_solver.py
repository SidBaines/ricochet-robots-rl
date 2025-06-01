import unittest
import numpy as np
from environment.ricochet_env import RicochetRobotsEnv
from environment.custom_difficulty_envs import RicochetRobotsEnvCustomDifficulty
from solvers.astar_solver import AStarSolver
from solvers.utils import get_optimal_solution, compare_solution_to_optimal

class TestAStarSolver(unittest.TestCase):
    def setUp(self):
        # Create a simple test environment
        self.env = RicochetRobotsEnv(
            board_size=5,
            num_robots=2,
            max_steps=20,
            use_standard_walls=False
        )
        self.env.reset()
        
    def test_simple_solution(self):
        """Test solving a simple puzzle where target is one step away."""
        # Set up a simple puzzle where target is one step away
        self.env.robots[0].set_position(2, 3)  # Target robot
        self.env.robots[1].set_position(3, 3)  # Other robot
        self.env.target_robot_idx = 0
        self.env.target_pos = (2, 4)  # To the east of target robot
        
        # Get optimal solution
        solution = get_optimal_solution(self.env)
        
        # Verify solution
        self.assertIsNotNone(solution)
        self.assertEqual(len(solution), 1)  # Should take exactly one move
        
        # Verify the solution works
        for action in solution:
            self.env.step(action)
        self.assertEqual(self.env.robots[self.env.target_robot_idx].pos, self.env.target_pos)

    def test_difficult_solution(self):
        """Test solving a difficult puzzle where target is not one step away."""
        
        # Create a larger environment with custom difficulty
        env = RicochetRobotsEnvCustomDifficulty(
            board_size=8,
            num_robots=3,
            max_steps=50,
            use_standard_walls=True,
            render_mode="human"
        )
        
        # Set up robot positions
        robot_positions = [(2, 2), (5, 5), (3, 6)]
        target_robot_idx = 0
        target_pos = (6, 6)
        
        # Reset with our custom configuration
        env.reset_with_custom_config(
            robot_positions=robot_positions,
            target_robot_idx=target_robot_idx,
            target_pos=target_pos
        )

        # Get optimal solution
        solution = get_optimal_solution(env)
        
        # Verify solution
        self.assertIsNotNone(solution)
        self.assertEqual(len(solution), 8)  # Should take exactly eight moves

        
        
    def test_no_solution(self):
        """Test a puzzle with no solution."""
        self.env.robots[0].set_position(2, 2)  # Target robot
        self.env.robots[1].set_position(3, 3)  # Other robot
        self.env.target_robot_idx = 0
        self.env.target_pos = (2, 3)  # One step east of target robot, but not against a wall and not enough robots to get there
        
        # Try to get solution
        solution = get_optimal_solution(self.env)
        self.assertIsNone(solution)
        
    def test_compare_solution(self):
        """Test comparing a solution to the optimal solution."""
        # Set up a simple puzzle
        self.env.robots[0].set_position(2, 3)  # Target robot
        self.env.robots[1].set_position(3, 3)  # Other robot
        self.env.target_robot_idx = 0
        self.env.target_pos = (2, 4)  # To the east of target robot
        
        # Get optimal solution
        optimal_solution = get_optimal_solution(self.env)
        self.assertIsNotNone(optimal_solution)
        
        # Compare optimal solution to itself
        is_optimal, optimal_length = compare_solution_to_optimal(self.env, optimal_solution)
        self.assertTrue(is_optimal)
        self.assertEqual(optimal_length, len(optimal_solution))
        
        # Compare a longer solution
        longer_solution = optimal_solution + [0]  # Add an extra move
        is_optimal, optimal_length = compare_solution_to_optimal(self.env, longer_solution)
        self.assertFalse(is_optimal)
        self.assertEqual(optimal_length, len(optimal_solution))
        
if __name__ == '__main__':
    unittest.main() 