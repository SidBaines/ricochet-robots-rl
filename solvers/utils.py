from typing import List, Tuple, Optional
import numpy as np
from environment.ricochet_env import RicochetRobotsEnv
from .astar_solver import AStarSolver

def create_solver_from_env(env: RicochetRobotsEnv) -> AStarSolver:
    """Create an A* solver configured for the given environment."""
    return AStarSolver(
        board_size=(env.height, env.width),
        num_robots=env.num_robots
    )

def get_optimal_solution(env: RicochetRobotsEnv) -> Optional[List[int]]:
    """
    Get the optimal solution for the current environment state.
    
    Args:
        env: The RicochetRobotsEnv instance
        
    Returns:
        List of actions that solve the puzzle optimally, or None if no solution exists
    """
    solver = create_solver_from_env(env)
    
    # Get current state from environment
    robot_positions = [robot.pos for robot in env.robots]
    target_robot_idx = env.target_robot_idx
    target_pos = env.target_pos
    
    # Get walls from board
    walls = env.board.walls
    
    return solver.solve(robot_positions, target_robot_idx, target_pos, walls)

def compare_solution_to_optimal(env: RicochetRobotsEnv, solution: List[int]) -> Tuple[bool, Optional[int]]:
    """
    Compare a solution to the optimal solution.
    
    Args:
        env: The RicochetRobotsEnv instance
        solution: List of actions representing a solution
        
    Returns:
        Tuple of (is_optimal, optimal_length)
        - is_optimal: True if the solution is optimal
        - optimal_length: Length of the optimal solution, or None if no solution exists
    """
    solver = create_solver_from_env(env)
    
    # Get current state from environment
    robot_positions = [robot.pos for robot in env.robots]
    target_robot_idx = env.target_robot_idx
    target_pos = env.target_pos
    walls = env.board.walls
    
    # Find optimal solution
    optimal_solution = solver.solve(robot_positions, target_robot_idx, target_pos, walls)
    
    if optimal_solution is None:
        return False, None
        
    optimal_length = len(optimal_solution)
    return len(solution) == optimal_length, optimal_length 