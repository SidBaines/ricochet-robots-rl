"""
LMDB cache for optimal solution lengths in curriculum learning.

Provides fast, persistent storage of board hash -> optimal length mappings
for efficient curriculum generation without recomputing solutions.
"""

import lmdb
import os
from typing import Optional
import struct


class BoardCache:
    """
    Fast, persistent cache for board optimal solution lengths using LMDB.
    
    Keys: 20-byte SHA1 hash (first 20 bytes of hex string)
    Values: 1-byte optimal length (0-255, where 255 = unsolvable/unknown)
    
    Thread-safe for multiple readers, single writer. Designed for use with
    multiprocessing where workers read from cache and a single process writes.
    """
    
    def __init__(self, cache_path: str, map_size: int = 100 * 1024 * 1024):
        """
        Initialize the LMDB cache.
        
        Args:
            cache_path: Path to the LMDB database file/directory
            map_size: Maximum size of the database in bytes (default: 100MB)
        """
        self.cache_path = cache_path
        self.map_size = map_size
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
        
        # Open LMDB environment
        self.env = lmdb.open(
            cache_path,
            map_size=map_size,
            max_readers=32,  # Allow multiple worker processes to read
            readonly=False,  # Allow writes
            sync=True,       # Ensure durability
            metasync=True
        )
    
    def _hash_to_key(self, board_hash: str) -> bytes:
        """Convert 40-character hex hash to 20-byte binary key."""
        return bytes.fromhex(board_hash)
    
    def _length_to_value(self, optimal_length: int) -> bytes:
        """Convert optimal length to 1-byte value."""
        if optimal_length < 0 or optimal_length > 254:
            # Use 255 as sentinel for unsolvable/unknown
            return struct.pack('B', 255)
        return struct.pack('B', optimal_length)
    
    def _value_to_length(self, value: bytes) -> Optional[int]:
        """Convert 1-byte value back to optimal length."""
        if len(value) != 1:
            return None
        length = struct.unpack('B', value)[0]
        return None if length == 255 else length
    
    def lookup(self, board_hash: str) -> Optional[int]:
        """
        Look up optimal length for a board hash.
        
        Args:
            board_hash: 40-character SHA1 hex string
            
        Returns:
            Optimal solution length, or None if not found/unsolvable
        """
        key = self._hash_to_key(board_hash)
        
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                return None
            return self._value_to_length(value)
    
    def insert(self, board_hash: str, optimal_length: int) -> bool:
        """
        Insert optimal length for a board hash.
        
        Args:
            board_hash: 40-character SHA1 hex string
            optimal_length: Optimal solution length (0-254, or any negative for unsolvable)
            
        Returns:
            True if insertion successful, False otherwise
        """
        key = self._hash_to_key(board_hash)
        value = self._length_to_value(optimal_length)
        
        try:
            with self.env.begin(write=True) as txn:
                return txn.put(key, value)
        except Exception:
            return False
    
    def batch_insert(self, entries: list[tuple[str, int]]) -> int:
        """
        Insert multiple entries in a single transaction for efficiency.
        
        Args:
            entries: List of (board_hash, optimal_length) tuples
            
        Returns:
            Number of entries successfully inserted
        """
        inserted_count = 0
        
        try:
            with self.env.begin(write=True) as txn:
                for board_hash, optimal_length in entries:
                    key = self._hash_to_key(board_hash)
                    value = self._length_to_value(optimal_length)
                    if txn.put(key, value):
                        inserted_count += 1
        except Exception:
            pass  # Return partial count
        
        return inserted_count
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache size, hit rate estimates, etc.
        """
        with self.env.begin(write=False) as txn:
            stat = txn.stat()
            return {
                'entries': stat['entries'],
                'page_size': stat['psize'],
                'depth': stat['depth'],
                'branch_pages': stat['branch_pages'],
                'leaf_pages': stat['leaf_pages'],
                'overflow_pages': stat['overflow_pages']
            }
    
    def contains(self, board_hash: str) -> bool:
        """Check if a board hash exists in the cache."""
        key = self._hash_to_key(board_hash)
        
        with self.env.begin(write=False) as txn:
            return txn.get(key) is not None
    
    def clear(self):
        """Clear all entries from the cache."""
        with self.env.begin(write=True) as txn:
            # Delete all entries
            cursor = txn.cursor()
            cursor.first()
            while cursor.delete():
                pass
    
    def close(self):
        """Close the LMDB environment."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_cache(cache_path: str, map_size: int = 100 * 1024 * 1024) -> BoardCache:
    """
    Factory function to create a new board cache.
    
    Args:
        cache_path: Path where the LMDB database should be stored
        map_size: Maximum database size in bytes
        
    Returns:
        Initialized BoardCache instance
    """
    return BoardCache(cache_path, map_size)


if __name__ == "__main__":
    # Basic test
    import tempfile
    import shutil
    
    # Create temporary cache for testing
    temp_dir = tempfile.mkdtemp()
    cache_path = os.path.join(temp_dir, "test_cache.lmdb")
    
    try:
        cache = BoardCache(cache_path)
        
        # Test insertion and lookup
        test_hash = "a" * 40  # Valid 40-char hex string
        cache.insert(test_hash, 5)
        
        result = cache.lookup(test_hash)
        assert result == 5, f"Expected 5, got {result}"
        
        # Test non-existent key
        missing = cache.lookup("b" * 40)
        assert missing is None, f"Expected None, got {missing}"
        
        # Test batch insertion
        batch_entries = [("c" * 40, 3), ("d" * 40, 7)]
        inserted = cache.batch_insert(batch_entries)
        assert inserted == 2, f"Expected 2 insertions, got {inserted}"
        
        # Verify batch entries
        assert cache.lookup("c" * 40) == 3
        assert cache.lookup("d" * 40) == 7
        
        # Test stats
        stats = cache.get_stats()
        assert stats['entries'] >= 3, f"Expected at least 3 entries, got {stats['entries']}"
        
        cache.close()
        print("Cache tests passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir) 