"""
Canonical board hashing for curriculum learning.

Provides deterministic hashing of board states including walls, robot positions,
and target configuration for efficient caching and curriculum generation.
"""

import hashlib
import numpy as np
from typing import List, Tuple
from environment.board import Board


def canonical_hash(board: Board, robot_positions: List[Tuple[int, int]], 
                   target_robot_idx: int, target_pos: Tuple[int, int]) -> str:
    """
    Generate a SHA1 hash over walls, robot positions and target in a deterministic byte layout.
    
    This hash captures the complete state of a Ricochet Robots puzzle instance,
    allowing for efficient caching of optimal solution lengths in curriculum learning.
    
    Args:
        board: The game board with wall configuration
        robot_positions: List of (row, col) positions for each robot
        target_robot_idx: Index of the robot that needs to reach the target
        target_pos: (row, col) position of the target
        
    Returns:
        40-character hexadecimal SHA1 hash string
    """
    # Start building deterministic byte representation
    hash_data = bytearray()
    
    # 1. Board dimensions (4 bytes each)
    hash_data.extend(board.height.to_bytes(4, byteorder='big'))
    hash_data.extend(board.width.to_bytes(4, byteorder='big'))
    
    # 2. Wall configuration - serialize the walls array
    # walls is shape (height, width, 4) boolean array
    walls_bytes = board.walls.astype(np.uint8).tobytes()
    hash_data.extend(walls_bytes)
    
    # 3. Robot positions - sort by position for deterministic ordering
    # to handle equivalent puzzle states with robots in different order
    sorted_positions = sorted(robot_positions)
    for r, c in sorted_positions:
        hash_data.extend(r.to_bytes(4, byteorder='big'))
        hash_data.extend(c.to_bytes(4, byteorder='big'))
    
    # 4. Target robot index (4 bytes)
    hash_data.extend(target_robot_idx.to_bytes(4, byteorder='big'))
    
    # 5. Target position (8 bytes)
    hash_data.extend(target_pos[0].to_bytes(4, byteorder='big'))
    hash_data.extend(target_pos[1].to_bytes(4, byteorder='big'))
    
    # 6. Number of robots for validation (4 bytes)
    hash_data.extend(len(robot_positions).to_bytes(4, byteorder='big'))
    
    # Generate SHA1 hash
    return hashlib.sha1(hash_data).hexdigest()


def board_state_to_bytes(board: Board, robot_positions: List[Tuple[int, int]], 
                        target_robot_idx: int, target_pos: Tuple[int, int]) -> bytes:
    """
    Convert board state to bytes for alternative storage/serialization.
    
    Returns the same deterministic byte representation used in canonical_hash
    but without the final hashing step, useful for debugging or alternative
    storage formats.
    """
    hash_data = bytearray()
    
    hash_data.extend(board.height.to_bytes(4, byteorder='big'))
    hash_data.extend(board.width.to_bytes(4, byteorder='big'))
    
    walls_bytes = board.walls.astype(np.uint8).tobytes()
    hash_data.extend(walls_bytes)
    
    sorted_positions = sorted(robot_positions)
    for r, c in sorted_positions:
        hash_data.extend(r.to_bytes(4, byteorder='big'))
        hash_data.extend(c.to_bytes(4, byteorder='big'))
    
    hash_data.extend(target_robot_idx.to_bytes(4, byteorder='big'))
    hash_data.extend(target_pos[0].to_bytes(4, byteorder='big'))
    hash_data.extend(target_pos[1].to_bytes(4, byteorder='big'))
    hash_data.extend(len(robot_positions).to_bytes(4, byteorder='big'))
    
    return bytes(hash_data)


def validate_hash_consistency():
    """
    Validation function to ensure hash consistency across different orderings
    and equivalent board states. Used in testing.
    """
    from environment.board import Board
    
    # Create a simple test board
    board = Board(5, 5)
    board.add_wall(1, 1, 0)  # Add some walls
    board.add_wall(2, 2, 1)
    
    # Test same configuration should produce same hash
    pos1 = [(0, 0), (1, 1), (2, 2)]
    pos2 = [(2, 2), (0, 0), (1, 1)]  # Different order
    
    hash1 = canonical_hash(board, pos1, 0, (3, 3))
    hash2 = canonical_hash(board, pos2, 0, (3, 3))
    
    assert hash1 == hash2, "Hashes should be identical for equivalent states"
    
    # Test different configurations should produce different hashes
    pos3 = [(0, 0), (1, 1), (2, 3)]  # Different robot position
    hash3 = canonical_hash(board, pos3, 0, (3, 3))
    
    assert hash1 != hash3, "Hashes should differ for different states"
    
    print("Hash consistency validation passed")


if __name__ == "__main__":
    validate_hash_consistency() 