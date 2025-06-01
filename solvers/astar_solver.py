import heapq
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from .base_solver import BaseSolver
from environment.ricochet_env import simulate_robot_move

class AStarSolver(BaseSolver):
    """A* solver for Ricochet Robots puzzles with early cutoff support."""
    
    def __init__(self, board_size: Tuple[int, int], num_robots: int):
        super().__init__(board_size, num_robots)
        self.solution = None
        self.solution_length = None
        
    def _get_robot_at(self, r: int, c: int, robot_positions: List[Tuple[int, int]]) -> Optional[int]:
        """Get the index of the robot at position (r,c), or None if no robot."""
        for i, pos in enumerate(robot_positions):
            if pos == (r, c):
                return i
        return None
    
    def _get_next_pos(self, r: int, c: int, direction: int, 
                     robot_positions: List[Tuple[int, int]], 
                     walls: np.ndarray) -> Tuple[int, int]:
        """Get the next position when moving in a direction until collision."""
        dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]  # N, E, S, W
        
        while True:
            if walls[r, c, direction]:
                break  # Hit a wall
                
            next_r, next_c = r + dr, c + dc
            
            # Check bounds
            if not (0 <= next_r < self.height and 0 <= next_c < self.width):
                break
                
            # Check for collision with another robot
            if self._get_robot_at(next_r, next_c, robot_positions) is not None:
                break
                
            r, c = next_r, next_c
            
        return (r, c)
    
    def _get_manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_state_key(self, robot_positions: List[Tuple[int, int]]) -> str:
        """Convert robot positions to a unique string key."""
        # return '|'.join(f"{r},{c}" for r, c in sorted(robot_positions))
        return '|'.join(f"{r},{c}" for r, c in robot_positions)
    
    def solve(self, 
              robot_positions: List[Tuple[int, int]], 
              target_robot_idx: int,
              target_pos: Tuple[int, int],
              walls: np.ndarray,
              depth_limit: Optional[int] = None) -> Optional[List[int]]:
        """
        Solve the puzzle using A* search with optional depth limit for early cutoff.
        
        Args:
            robot_positions: List of (row, col) positions for each robot
            target_robot_idx: Index of the robot that needs to reach the target
            target_pos: (row, col) position of the target
            walls: Boolean array of shape (height, width, 4) indicating walls
            depth_limit: Maximum depth to search (for curriculum learning early cutoff)
            
        Returns:
            List of actions that solve the puzzle, or None if no solution exists within depth limit
        """
        # Priority queue for A* search: (f_score, state_key, robot_positions, actions)
        # f_score = g_score (moves so far) + h_score (manhattan distance to target)
        pq = []
        initial_state = self._get_state_key(robot_positions)
        heapq.heappush(pq, (0, initial_state, robot_positions, []))
        
        # Keep track of visited states and their g_scores
        visited = {initial_state: 0}  # state_key -> g_score
        
        while pq:
            f_score, state_key, current_positions, actions = heapq.heappop(pq)
            
            # Early cutoff check - abort if we've exceeded the depth limit
            if depth_limit is not None and len(actions) >= depth_limit:
                continue  # Skip this path, try others
            
            # Check if target robot reached target
            if current_positions[target_robot_idx] == target_pos:
                self.solution = actions
                self.solution_length = len(actions)
                return actions
            
            # Try moving each robot in each direction
            for robot_idx in range(self.num_robots):
                for direction in range(4):
                    r, c = current_positions[robot_idx]
                    next_r, next_c = simulate_robot_move((r, c), robot_idx, direction, walls, current_positions, list(range(self.num_robots)), self.height, self.width)
                    
                    if (next_r, next_c) == (r, c):
                        continue  # No movement possible
                        
                    # Create new state with updated robot position
                    new_positions = current_positions.copy()
                    new_positions[robot_idx] = (next_r, next_c)
                    new_state = self._get_state_key(new_positions)
                    
                    # Calculate new g_score and f_score
                    new_g_score = len(actions) + 1
                    
                    # Early cutoff check - don't explore paths that exceed depth limit
                    if depth_limit is not None and new_g_score >= depth_limit:
                        continue
                    
                    h_score = self._get_manhattan_distance(new_positions[target_robot_idx], target_pos)
                    new_f_score = new_g_score + h_score
                    
                    # Only explore if this is a new state or we found a better path
                    if new_state not in visited or new_g_score < visited[new_state]:
                        visited[new_state] = new_g_score
                        new_actions = actions + [robot_idx * 4 + direction]
                        heapq.heappush(pq, (new_f_score, new_state, new_positions, new_actions))
        
        # No solution found within depth limit
        self.solution = None
        self.solution_length = None
        return None
    
    def solve_with_cutoff(self, 
                         robot_positions: List[Tuple[int, int]], 
                         target_robot_idx: int,
                         target_pos: Tuple[int, int],
                         walls: np.ndarray,
                         cutoff: int) -> Tuple[Optional[List[int]], bool]:
        """
        Solve puzzle with early cutoff, returning both solution and whether cutoff was hit.
        
        Convenience method for curriculum learning that makes the cutoff behavior explicit.
        
        Args:
            robot_positions: List of (row, col) positions for each robot
            target_robot_idx: Index of the robot that needs to reach the target
            target_pos: (row, col) position of the target
            walls: Boolean array of shape (height, width, 4) indicating walls
            cutoff: Maximum depth to search before aborting
            
        Returns:
            Tuple of (solution_actions, cutoff_hit)
            - solution_actions: List of actions if solved within cutoff, None otherwise
            - cutoff_hit: True if search was aborted due to cutoff, False if naturally exhausted
        """
        solution = self.solve(robot_positions, target_robot_idx, target_pos, walls, depth_limit=cutoff)
        
        # If we got a solution, cutoff wasn't hit
        if solution is not None:
            return solution, False
        
        # If no solution found, we need to determine if it was due to cutoff or no solution exists
        # Run without cutoff to check (but only if cutoff was reasonably small to avoid infinite search)
        if cutoff <= 20:  # Reasonable limit for checking
            full_solution = self.solve(robot_positions, target_robot_idx, target_pos, walls, depth_limit=None)
            cutoff_hit = full_solution is not None
            return None, cutoff_hit
        else:
            # For large cutoffs, assume cutoff was hit rather than running expensive full search
            return None, True
    
    def get_solution_length(self) -> Optional[int]:
        """Get the length of the solution found by the solver."""
        return self.solution_length 