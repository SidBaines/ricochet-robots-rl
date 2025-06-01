"""
Examples of using the A* solver for Ricochet Robots.

This file demonstrates various ways to use the solver:
1. Finding optimal solutions for different puzzle configurations
2. Comparing solutions to optimal ones
3. Using the solver with different environment types
"""

import numpy as np
from environment.ricochet_env import RicochetRobotsEnv
from environment.simpler_ricochet_env import RicochetRobotsEnvOneStepAway, RicochetRobotsEnvCornerTarget
from environment.custom_difficulty_envs import RicochetRobotsEnvCustomDifficulty
from environment.utils import action_to_human_readable
from solvers import get_optimal_solution, compare_solution_to_optimal

def example_simple_puzzle():
    """Example of solving a simple puzzle with a known solution."""
    print("\n=== Simple Puzzle Example ===")
    
    # Create a small environment with custom difficulty
    env = RicochetRobotsEnvCustomDifficulty(
        board_size=5,
        num_robots=2,
        max_steps=20,
        use_standard_walls=False,
        render_mode="human"
    )
    
    # Set up a simple puzzle where target is one step away
    robot_positions = [(2, 2), (3, 3)]  # Target robot at (2,2), other robot at (3,3)
    target_robot_idx = 0  # First robot is the target
    target_pos = (2, 3)  # One step east of target robot
    
    # Reset with our custom configuration
    env.reset_with_custom_config(
        robot_positions=robot_positions,
        target_robot_idx=target_robot_idx,
        target_pos=target_pos
    )
    
    # Print initial state
    print("\nInitial State:")
    env.render()
    
    # Get optimal solution
    solution = get_optimal_solution(env)
    if solution is None:
        print("\nNo solution found! Debugging info:")
        print(f"Target robot position: {env.robots[env.target_robot_idx].pos}")
        print(f"Target position: {env.target_pos}")
        print(f"Walls at target robot position: {env.board.walls[2, 2]}")
        print(f"Walls at target position: {env.board.walls[2, 3]}")
        return
        
    print(f"\nOptimal solution: {solution}")
    print("Solution in human-readable format:")
    for action in solution:
        print(f"- {action_to_human_readable(action)}")
    print(f"Solution length: {len(solution)}")
    
    # Verify the solution works
    for action in solution:
        env.step(action)
    print("\nFinal State:")
    env.render()
    print(f"Target robot reached target: {env.robots[env.target_robot_idx].pos == env.target_pos}")

def example_complex_puzzle():
    """Example of solving a more complex puzzle with walls."""
    print("\n=== Complex Puzzle Example ===")
    
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
    
    # Add some additional walls to make it interesting
    env.board.add_wall(3, 3, 1)  # East wall at (3,3)
    env.board.add_wall(3, 4, 3)  # West wall at (3,4)
    env.board.add_wall(4, 3, 2)  # South wall at (4,3)
    
    # Print initial state
    print("\nInitial State:")
    env.render()
    
    # Get optimal solution
    solution = get_optimal_solution(env)
    if solution:
        print(f"\nFound solution of length {len(solution)}")
        print("Solution in human-readable format:")
        for action in solution:
            print(f"- {action_to_human_readable(action)}")
        
        # Show final state
        for action in solution:
            env.step(action)
        print("\nFinal State:")
        env.render()
    else:
        print("No solution found!")

def example_compare_solutions():
    """Example of comparing different solutions to the optimal one."""
    print("\n=== Solution Comparison Example ===")
    
    # Create a simple environment with custom difficulty
    env = RicochetRobotsEnvCustomDifficulty(
        board_size=5,
        num_robots=2,
        max_steps=20,
        use_standard_walls=False,
        render_mode="human"
    )
    
    # Set up a simple puzzle
    robot_positions = [(2, 2), (3, 3)]  # Target robot at (2,2), other robot at (3,3)
    target_robot_idx = 0  # First robot is the target
    target_pos = (2, 3)  # One step east of target robot
    
    # Reset with our custom configuration
    env.reset_with_custom_config(
        robot_positions=robot_positions,
        target_robot_idx=target_robot_idx,
        target_pos=target_pos
    )
    
    # Print initial state
    print("\nInitial State:")
    env.render()
    
    # Get optimal solution
    optimal_solution = get_optimal_solution(env)
    if optimal_solution is None:
        print("No optimal solution found!")
        return
        
    print(f"\nOptimal solution length: {len(optimal_solution)}")
    print("Optimal solution in human-readable format:")
    for action in optimal_solution:
        print(f"- {action_to_human_readable(action)}")
    
    # Create a suboptimal solution by adding extra moves
    suboptimal_solution = optimal_solution + [0, 0]  # Add two extra moves
    print(f"\nSuboptimal solution length: {len(suboptimal_solution)}")
    print("Suboptimal solution in human-readable format:")
    for action in suboptimal_solution:
        print(f"- {action_to_human_readable(action)}")
    
    # Compare solutions
    is_optimal, optimal_length = compare_solution_to_optimal(env, optimal_solution)
    print(f"Optimal solution is optimal: {is_optimal}")
    
    is_optimal, optimal_length = compare_solution_to_optimal(env, suboptimal_solution)
    print(f"Suboptimal solution is optimal: {is_optimal}")
    print(f"Difference from optimal: {len(suboptimal_solution) - optimal_length} moves")
    
    # Show final state after optimal solution
    for action in optimal_solution:
        env.step(action)
    print("\nFinal State (after optimal solution):")
    env.render()

def example_one_step_away_env():
    """Example of using the solver with the OneStepAway environment."""
    print("\n=== One Step Away Environment Example ===")
    
    # Create the special environment
    env = RicochetRobotsEnvOneStepAway(
        board_size=5,
        num_robots=2,
        max_steps=10,
        render_mode="human"
    )
    env.reset()
    
    # Print initial state
    print("\nInitial State:")
    env.render()
    
    # Get optimal solution
    solution = get_optimal_solution(env)
    if solution is None:
        print("No solution found!")
        return
        
    print(f"\nSolution length: {len(solution)}")
    print("Solution in human-readable format:")
    for action in solution:
        print(f"- {action_to_human_readable(action)}")
    
    # Verify the solution works
    for action in solution:
        env.step(action)
    print("\nFinal State:")
    env.render()
    print(f"Target robot reached target: {env.robots[env.target_robot_idx].pos == env.target_pos}")

def example_corner_target_env():
    """Example of using the solver with the CornerTarget environment."""
    print("\n=== Corner Target Environment Example ===")
    
    # Create the special environment
    env = RicochetRobotsEnvCornerTarget(
        board_size=8,
        num_robots=3,
        max_steps=30,
        render_mode="human"
    )
    env = RicochetRobotsEnvCornerTarget(
        board_size=5,
        num_robots=2,
        max_steps=30,
        render_mode="human",
        seed=0
    )
    env.reset(seed=0)
    
    # Print initial state
    print("\nInitial State:")
    env.render()
    
    # Get optimal solution
    solution = get_optimal_solution(env)
    if solution:
        print(f"\nFound solution of length {len(solution)}")
        print("Solution in human-readable format:")
        for action in solution:
            print(f"- {action_to_human_readable(action)}")
        
        # Show final state
        for action in solution:
            env.step(action)
        print("\nFinal State:")
        env.render()
    else:
        print("No solution found!")

def main():
    """Run all examples."""
    print("Running Ricochet Robots Solver Examples")
    print("=======================================")
    
    example_simple_puzzle()
    example_complex_puzzle()
    example_compare_solutions()
    example_one_step_away_env()
    example_corner_target_env()

if __name__ == "__main__":
    main() 