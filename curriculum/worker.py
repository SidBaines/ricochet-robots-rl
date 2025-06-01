"""
Board generation worker for curriculum learning.

Multiprocessing worker that generates random boards, computes optimal solution lengths
using A* with early cutoff, and caches results for curriculum generation.
"""

import multiprocessing as mp
import queue
import random
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

from environment.board import Board
from environment.ricochet_env import RicochetRobotsEnv
from environment.simpler_ricochet_env import RicochetRobotsEnvOneStepAway
from solvers.astar_solver import AStarSolver
from utils.board_hash import canonical_hash
from utils.board_cache import BoardCache


class BoardGeneratorWorker:
    """
    Worker process that generates boards and computes optimal solution lengths.
    
    Each worker runs in its own process, generating random board configurations
    and solving them with A* search. Results are cached and valid boards
    (with optimal length ≤ target_k) are queued for training.
    """
    
    def __init__(self, 
                 worker_id: int,
                 output_queue: mp.Queue,
                 target_k_value: mp.Value,
                 cache_path: str,
                 board_config: Dict[str, Any],
                 stop_event: mp.Event):
        """
        Initialize board generator worker.
        
        Args:
            worker_id: Unique identifier for this worker
            output_queue: Queue to send valid boards to the main process
            target_k_value: Shared value containing current curriculum difficulty k
            cache_path: Path to LMDB cache for storing solution lengths
            board_config: Configuration dict with board generation parameters
            stop_event: Event to signal worker shutdown
        """
        self.worker_id = worker_id
        self.output_queue = output_queue
        self.target_k_value = target_k_value
        self.cache_path = cache_path
        self.board_config = board_config
        self.stop_event = stop_event
        
        # Statistics
        self.boards_generated = 0
        self.boards_solved = 0
        self.cache_hits = 0
        self.boards_enqueued = 0
        
    def _create_random_board(self, seed: int) -> Tuple[Board, list, int, tuple]:
        """
        Create a random board configuration with specified parameters.
        
        Args:
            seed: Random seed for reproducible generation
            
        Returns:
            Tuple of (board, robot_positions, target_robot_idx, target_pos)
        """
        # Set up random state
        local_random = random.Random(seed)
        local_np_random = np.random.Generator(np.random.PCG64(seed))
        
        # Extract board configuration
        board_size = self.board_config.get('board_size', 5)
        num_robots = self.board_config.get('num_robots', 3)
        use_standard_walls = self.board_config.get('use_standard_walls', False)
        num_edge_walls_per_quadrant = self.board_config.get('num_edge_walls_per_quadrant', 0)
        num_floating_walls_per_quadrant = self.board_config.get('num_floating_walls_per_quadrant', 0)
        
        # Create board
        board = Board(board_size, board_size)
        
        if use_standard_walls:
            board.add_standard_ricochet_walls()
        else:
            # Add middle blocked walls for simpler environments
            if board_size > 2:
                board.add_middle_blocked_walls()
        
        # Add random walls if specified
        if num_edge_walls_per_quadrant > 0 or num_floating_walls_per_quadrant > 0:
            board.generate_random_walls(
                num_edge_walls_per_quadrant=num_edge_walls_per_quadrant,
                num_floating_walls_per_quadrant=num_floating_walls_per_quadrant,
                rng=local_np_random
            )
        
        # Generate robot positions
        blocked_cells = board.get_blocked_cells()
        available_positions = [
            (r, c) for r in range(board_size) for c in range(board_size)
            if (r, c) not in blocked_cells
        ]
        
        if len(available_positions) < num_robots + 1:  # +1 for target
            raise ValueError(f"Not enough available positions for {num_robots} robots and target")
        
        # Sample positions without replacement
        chosen_positions = local_random.sample(available_positions, num_robots + 1)
        robot_positions = chosen_positions[:num_robots]
        target_pos = chosen_positions[num_robots]
        
        # Choose random target robot
        target_robot_idx = local_random.randint(0, num_robots - 1)
        
        return board, robot_positions, target_robot_idx, target_pos
    
    def _solve_board(self, board: Board, robot_positions: list, 
                     target_robot_idx: int, target_pos: tuple, 
                     cutoff: int) -> Optional[int]:
        """
        Solve a board using A* with early cutoff.
        
        Args:
            board: Game board
            robot_positions: List of robot positions
            target_robot_idx: Index of target robot
            target_pos: Target position
            cutoff: Maximum search depth
            
        Returns:
            Optimal solution length if solved within cutoff, None otherwise
        """
        solver = AStarSolver((board.height, board.width), len(robot_positions))
        
        solution, cutoff_hit = solver.solve_with_cutoff(
            robot_positions=robot_positions,
            target_robot_idx=target_robot_idx,
            target_pos=target_pos,
            walls=board.walls,
            cutoff=cutoff
        )
        
        if solution is not None:
            return len(solution)
        
        # If cutoff was hit, the board might still be solvable but requires > cutoff moves
        return None
    
    def run(self):
        """Main worker loop - generates and processes boards continuously."""
        print(f"Worker {self.worker_id} starting...")
        
        # Initialize cache (read-only for workers)
        cache = BoardCache(self.cache_path)
        
        # Initialize local random state
        worker_seed = int(time.time() * 1000) % (2**32) + self.worker_id * 1000
        local_random = random.Random(worker_seed)
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get current target difficulty
                    current_k = self.target_k_value.value
                    
                    # Generate random seed for this board
                    board_seed = local_random.randint(0, 2**31 - 1)
                    
                    # Create random board
                    board, robot_positions, target_robot_idx, target_pos = self._create_random_board(board_seed)
                    self.boards_generated += 1
                    
                    # Compute board hash
                    board_hash = canonical_hash(board, robot_positions, target_robot_idx, target_pos)
                    
                    # Check cache first
                    cached_length = cache.lookup(board_hash)
                    if cached_length is not None:
                        self.cache_hits += 1
                        optimal_length = cached_length
                    else:
                        # Solve with early cutoff
                        optimal_length = self._solve_board(
                            board, robot_positions, target_robot_idx, target_pos, 
                            cutoff=current_k + 5  # Allow some buffer beyond current k
                        )
                        
                        if optimal_length is not None:
                            # Cache the result
                            cache.insert(board_hash, optimal_length)
                            self.boards_solved += 1
                    
                    # If board is suitable for current curriculum level, enqueue it
                    if optimal_length is not None and optimal_length <= current_k:
                        board_data = {
                            'board': board,
                            'robot_positions': robot_positions,
                            'target_robot_idx': target_robot_idx,
                            'target_pos': target_pos,
                            'optimal_length': optimal_length,
                            'seed': board_seed
                        }
                        
                        try:
                            # Non-blocking put with timeout
                            self.output_queue.put(board_data, timeout=0.1)
                            self.boards_enqueued += 1
                        except queue.Full:
                            # Queue is full, skip this board
                            pass
                    
                    # Brief pause to prevent CPU overwhelming
                    time.sleep(0.001)
                    
                except Exception as e:
                    print(f"Worker {self.worker_id} error: {e}")
                    time.sleep(0.1)  # Brief pause on error
                    
        except KeyboardInterrupt:
            pass
        finally:
            cache.close()
            print(f"Worker {self.worker_id} stopping. Stats: "
                  f"Generated={self.boards_generated}, Solved={self.boards_solved}, "
                  f"Cache hits={self.cache_hits}, Enqueued={self.boards_enqueued}")


def start_workers(num_workers: int,
                  output_queue: mp.Queue,
                  target_k_value: mp.Value,
                  cache_path: str,
                  board_config: Dict[str, Any]) -> Tuple[list, mp.Event]:
    """
    Start board generator worker processes.
    
    Args:
        num_workers: Number of worker processes to start
        output_queue: Queue for workers to send generated boards
        target_k_value: Shared value containing current curriculum difficulty
        cache_path: Path to LMDB cache
        board_config: Board generation configuration
        
    Returns:
        Tuple of (worker_processes_list, stop_event)
    """
    # Ensure multiprocessing uses spawn method for cross-platform compatibility
    mp.set_start_method('spawn', force=True)
    
    stop_event = mp.Event()
    processes = []
    
    for worker_id in range(num_workers):
        worker = BoardGeneratorWorker(
            worker_id=worker_id,
            output_queue=output_queue,
            target_k_value=target_k_value,
            cache_path=cache_path,
            board_config=board_config,
            stop_event=stop_event
        )
        
        process = mp.Process(target=worker.run)
        process.start()
        processes.append(process)
    
    print(f"Started {num_workers} board generator workers")
    return processes, stop_event


def stop_workers(processes: list, stop_event: mp.Event, timeout: float = 5.0):
    """
    Stop board generator worker processes gracefully.
    
    Args:
        processes: List of worker processes
        stop_event: Event to signal workers to stop
        timeout: Maximum time to wait for graceful shutdown
    """
    print("Stopping board generator workers...")
    stop_event.set()
    
    # Wait for processes to finish gracefully
    for process in processes:
        process.join(timeout=timeout)
        if process.is_alive():
            print(f"Force terminating worker process {process.pid}")
            process.terminate()
            process.join()
    
    print("All workers stopped") 