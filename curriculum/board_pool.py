"""
Board pool manager for curriculum learning.

Maintains an in-memory queue of ready-to-use boards for training environments.
Automatically requests more boards from workers when pool runs low.
"""

import queue
import threading
import time
import multiprocessing as mp
from typing import Dict, Any, Optional
from collections import deque


class BoardPool:
    """
    Thread-safe pool of generated boards for curriculum learning.
    
    Maintains a bounded queue of boards ready for training, with automatic
    refill requests to worker processes when the pool runs low.
    """
    
    def __init__(self, 
                 maxsize: int,
                 low_watermark: Optional[int] = None,
                 worker_queue: Optional[mp.Queue] = None,
                 target_k_value: Optional[mp.Value] = None):
        """
        Initialize board pool.
        
        Args:
            maxsize: Maximum number of boards to keep in pool
            low_watermark: Trigger refill when pool size drops below this (default: maxsize // 4)
            worker_queue: Queue to receive boards from worker processes
            target_k_value: Shared value containing current curriculum difficulty
        """
        self.maxsize = maxsize
        self.low_watermark = low_watermark or max(1, maxsize // 4)
        self.worker_queue = worker_queue
        self.target_k_value = target_k_value
        
        # Thread-safe queue for storing boards
        self.board_queue = queue.Queue(maxsize=maxsize)
        
        # Statistics
        self.total_boards_received = 0
        self.total_boards_served = 0
        self.refill_requests = 0
        
        # Thread for receiving boards from workers
        self._receiver_thread = None
        self._stop_event = threading.Event()
        
        if worker_queue is not None:
            self._start_receiver_thread()
    
    def _start_receiver_thread(self):
        """Start background thread to receive boards from workers."""
        self._receiver_thread = threading.Thread(target=self._receive_boards, daemon=True)
        self._receiver_thread.start()
    
    def _receive_boards(self):
        """Background thread that receives boards from worker processes."""
        while not self._stop_event.is_set():
            try:
                # Get board from worker queue with timeout
                board_data = self.worker_queue.get(timeout=0.5)
                
                # Add to pool if there's space
                try:
                    self.board_queue.put(board_data, block=False)
                    self.total_boards_received += 1
                except queue.Full:
                    # Pool is full, discard this board
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Board pool receiver error: {e}")
                time.sleep(0.1)
    
    def get_board(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get a board from the pool.
        
        Args:
            timeout: Maximum time to wait for a board (seconds)
            
        Returns:
            Board data dict, or None if timeout exceeded
        """
        try:
            board_data = self.board_queue.get(timeout=timeout)
            self.total_boards_served += 1
            
            # Check if we need to request more boards
            current_size = self.board_queue.qsize()
            if current_size <= self.low_watermark:
                self.refill_requests += 1
                # In a full implementation, this would signal workers to generate more
                # For now, we rely on continuous generation
            
            return board_data
            
        except queue.Empty:
            return None
    
    def put_board(self, board_data: Dict[str, Any]) -> bool:
        """
        Add a board to the pool (for manual addition).
        
        Args:
            board_data: Board data dictionary
            
        Returns:
            True if added successfully, False if pool is full
        """
        try:
            self.board_queue.put(board_data, block=False)
            self.total_boards_received += 1
            return True
        except queue.Full:
            return False
    
    def size(self) -> int:
        """Get current number of boards in pool."""
        return self.board_queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if pool is empty."""
        return self.board_queue.empty()
    
    def is_full(self) -> bool:
        """Check if pool is full."""
        return self.board_queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool metrics
        """
        return {
            'current_size': self.size(),
            'maxsize': self.maxsize,
            'low_watermark': self.low_watermark,
            'total_received': self.total_boards_received,
            'total_served': self.total_boards_served,
            'refill_requests': self.refill_requests,
            'utilization': self.size() / self.maxsize if self.maxsize > 0 else 0.0
        }
    
    def set_target_k(self, new_k: int):
        """
        Update the target curriculum difficulty.
        
        Args:
            new_k: New curriculum difficulty level
        """
        if self.target_k_value is not None:
            self.target_k_value.value = new_k
    
    def clear(self):
        """Clear all boards from the pool."""
        while not self.board_queue.empty():
            try:
                self.board_queue.get_nowait()
            except queue.Empty:
                break
    
    def close(self):
        """Clean shutdown of the board pool."""
        self._stop_event.set()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=2.0)


class BoardPoolManager:
    """
    Manager class that coordinates board pool with worker processes.
    
    Simplifies setup and coordination between board generation workers
    and the board pool used by training environments.
    """
    
    def __init__(self, 
                 pool_size: int,
                 num_workers: int,
                 cache_path: str,
                 board_config: Dict[str, Any],
                 initial_k: int = 2):
        """
        Initialize board pool manager.
        
        Args:
            pool_size: Maximum boards to keep in pool
            num_workers: Number of worker processes to start
            cache_path: Path to LMDB cache
            board_config: Board generation configuration
            initial_k: Initial curriculum difficulty level
        """
        self.pool_size = pool_size
        self.num_workers = num_workers
        self.cache_path = cache_path
        self.board_config = board_config
        
        # Multiprocessing components
        self.worker_queue = mp.Queue(maxsize=pool_size * 2)
        self.target_k_value = mp.Value('i', initial_k)
        
        # Board pool
        self.board_pool = BoardPool(
            maxsize=pool_size,
            worker_queue=self.worker_queue,
            target_k_value=self.target_k_value
        )
        
        # Worker processes (will be initialized when started)
        self.worker_processes = []
        self.stop_event = None
    
    def start(self):
        """Start worker processes and board pool."""
        from .worker import start_workers
        
        print(f"Starting board pool manager with {self.num_workers} workers...")
        
        self.worker_processes, self.stop_event = start_workers(
            num_workers=self.num_workers,
            output_queue=self.worker_queue,
            target_k_value=self.target_k_value,
            cache_path=self.cache_path,
            board_config=self.board_config
        )
        
        print(f"Board pool manager started (pool size: {self.pool_size})")
    
    def stop(self):
        """Stop worker processes and clean up."""
        if self.stop_event is not None:
            from .worker import stop_workers
            stop_workers(self.worker_processes, self.stop_event)
        
        self.board_pool.close()
        print("Board pool manager stopped")
    
    def get_board(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a board from the pool."""
        return self.board_pool.get_board(timeout=timeout)
    
    def set_curriculum_level(self, k: int):
        """Update curriculum difficulty level."""
        self.board_pool.set_target_k(k)
        print(f"Curriculum level updated to k={k}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        pool_stats = self.board_pool.get_stats()
        pool_stats.update({
            'num_workers': self.num_workers,
            'current_k': self.target_k_value.value if self.target_k_value else None
        })
        return pool_stats
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_board_pool_manager(config: Dict[str, Any]) -> BoardPoolManager:
    """
    Factory function to create a board pool manager from configuration.
    
    Args:
        config: Configuration dictionary with curriculum settings
        
    Returns:
        Initialized BoardPoolManager
    """
    pool_size = config.get('pool_size', 20)
    num_workers = config.get('num_workers', min(8, mp.cpu_count() - 1))
    cache_path = config.get('cache_path', 'curriculum_cache.lmdb')
    initial_k = config.get('initial_k', 2)
    
    board_config = {
        'board_size': config.get('board_size', 5),
        'num_robots': config.get('num_robots', 3),
        'use_standard_walls': config.get('use_standard_walls', False),
        'num_edge_walls_per_quadrant': config.get('num_edge_walls_per_quadrant', 0),
        'num_floating_walls_per_quadrant': config.get('num_floating_walls_per_quadrant', 0)
    }
    
    return BoardPoolManager(
        pool_size=pool_size,
        num_workers=num_workers,
        cache_path=cache_path,
        board_config=board_config,
        initial_k=initial_k
    ) 