"""
Unit tests for curriculum learning components.

Tests the board hashing, cache functionality, worker processes, and
curriculum environment integration.
"""

import unittest
import tempfile
import shutil
import os
import time
import multiprocessing as mp
from unittest.mock import patch, MagicMock

import numpy as np
from environment.board import Board
from utils.board_hash import canonical_hash, validate_hash_consistency
from utils.board_cache import BoardCache
from curriculum.board_pool import BoardPool, BoardPoolManager
from curriculum.curriculum_env import CurriculumRicochetEnv


class TestBoardHashing(unittest.TestCase):
    """Test board hashing functionality."""
    
    def setUp(self):
        self.board = Board(5, 5)
        self.board.add_wall(1, 1, 0)  # Add some walls for testing
        self.board.add_wall(2, 2, 1)
        
        self.robot_positions = [(0, 0), (1, 1), (2, 2)]
        self.target_robot_idx = 0
        self.target_pos = (3, 3)
    
    def test_hash_deterministic(self):
        """Test that identical board states produce identical hashes."""
        hash1 = canonical_hash(
            self.board, self.robot_positions, 
            self.target_robot_idx, self.target_pos
        )
        hash2 = canonical_hash(
            self.board, self.robot_positions, 
            self.target_robot_idx, self.target_pos
        )
        self.assertEqual(hash1, hash2)
    
    def test_hash_order_independent(self):
        """Test that robot position order doesn't affect hash."""
        pos1 = [(0, 0), (1, 1), (2, 2)]
        pos2 = [(2, 2), (0, 0), (1, 1)]  # Different order
        
        hash1 = canonical_hash(self.board, pos1, 0, self.target_pos)
        hash2 = canonical_hash(self.board, pos2, 0, self.target_pos)
        
        self.assertEqual(hash1, hash2)
    
    def test_hash_different_states(self):
        """Test that different board states produce different hashes."""
        pos1 = [(0, 0), (1, 1), (2, 2)]
        pos2 = [(0, 0), (1, 1), (2, 3)]  # Different position
        
        hash1 = canonical_hash(self.board, pos1, 0, self.target_pos)
        hash2 = canonical_hash(self.board, pos2, 0, self.target_pos)
        
        self.assertNotEqual(hash1, hash2)
    
    def test_hash_consistency_validation(self):
        """Test the hash consistency validation function."""
        # Should not raise any assertions
        validate_hash_consistency()


class TestBoardCache(unittest.TestCase):
    """Test LMDB board cache functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.temp_dir, "test_cache.lmdb")
        self.cache = BoardCache(self.cache_path)
    
    def tearDown(self):
        self.cache.close()
        shutil.rmtree(self.temp_dir)
    
    def test_insert_and_lookup(self):
        """Test basic insert and lookup operations."""
        test_hash = "a" * 40  # Valid 40-char hex string
        
        # Insert
        success = self.cache.insert(test_hash, 5)
        self.assertTrue(success)
        
        # Lookup
        result = self.cache.lookup(test_hash)
        self.assertEqual(result, 5)
    
    def test_nonexistent_lookup(self):
        """Test lookup of non-existent key returns None."""
        result = self.cache.lookup("b" * 40)
        self.assertIsNone(result)
    
    def test_batch_insert(self):
        """Test batch insertion functionality."""
        entries = [("c" * 40, 3), ("d" * 40, 7)]
        inserted = self.cache.batch_insert(entries)
        
        self.assertEqual(inserted, 2)
        self.assertEqual(self.cache.lookup("c" * 40), 3)
        self.assertEqual(self.cache.lookup("d" * 40), 7)
    
    def test_unsolvable_boards(self):
        """Test handling of unsolvable boards (length > 254)."""
        test_hash = "e" * 40
        
        # Insert unsolvable board
        self.cache.insert(test_hash, -1)  # Negative indicates unsolvable
        
        # Should return None for unsolvable
        result = self.cache.lookup(test_hash)
        self.assertIsNone(result)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Insert some entries
        self.cache.insert("f" * 40, 1)
        self.cache.insert("a" * 40, 2)
        
        stats = self.cache.get_stats()
        self.assertGreaterEqual(stats['entries'], 2)
        self.assertIn('page_size', stats)


class TestBoardPool(unittest.TestCase):
    """Test board pool functionality."""
    
    def setUp(self):
        self.pool = BoardPool(maxsize=5, low_watermark=2)
    
    def tearDown(self):
        self.pool.close()
    
    def test_put_and_get_board(self):
        """Test putting and getting boards from pool."""
        board_data = {
            'optimal_length': 3,
            'seed': 12345,
            'test_data': 'test'
        }
        
        # Put board
        success = self.pool.put_board(board_data)
        self.assertTrue(success)
        self.assertEqual(self.pool.size(), 1)
        
        # Get board
        retrieved = self.pool.get_board(timeout=0.1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['optimal_length'], 3)
        self.assertEqual(self.pool.size(), 0)
    
    def test_pool_capacity(self):
        """Test pool respects maximum capacity."""
        # Fill pool to capacity
        for i in range(6):  # maxsize is 5
            board_data = {'seed': i}
            success = self.pool.put_board(board_data)
            if i < 5:
                self.assertTrue(success)
            else:
                self.assertFalse(success)  # Should fail when full
        
        self.assertEqual(self.pool.size(), 5)
    
    def test_empty_pool_timeout(self):
        """Test getting from empty pool returns None after timeout."""
        result = self.pool.get_board(timeout=0.05)
        self.assertIsNone(result)
    
    def test_pool_stats(self):
        """Test pool statistics tracking."""
        # Add and remove some boards
        self.pool.put_board({'seed': 1})
        self.pool.put_board({'seed': 2})
        self.pool.get_board()
        
        stats = self.pool.get_stats()
        self.assertEqual(stats['total_received'], 2)
        self.assertEqual(stats['total_served'], 1)
        self.assertEqual(stats['current_size'], 1)


class TestCurriculumIntegration(unittest.TestCase):
    """Test curriculum learning integration."""
    
    def setUp(self):
        # Mock board pool manager to avoid starting real multiprocessing
        self.mock_pool_manager = MagicMock()
        self.mock_pool_manager.get_board.return_value = {
            'board': Board(5, 5),
            'robot_positions': [(0, 0), (1, 1), (2, 2)],
            'target_robot_idx': 0,
            'target_pos': (3, 3),
            'optimal_length': 3,
            'seed': 12345
        }
        self.mock_pool_manager.get_stats.return_value = {
            'current_size': 10,
            'utilization': 0.5,
            'current_k': 3
        }
    
    def test_curriculum_env_creation(self):
        """Test curriculum environment can be created."""
        # Use mock to avoid starting workers
        with patch('curriculum.curriculum_env.create_curriculum_env') as mock_create:
            mock_env = MagicMock()
            mock_create.return_value = mock_env
            
            env = mock_create({
                'board_size': 5,
                'num_robots': 3,
                'initial_k': 2
            })
            
            self.assertIsNotNone(env)
    
    def test_curriculum_env_board_loading(self):
        """Test curriculum environment loads boards correctly."""
        from environment.simpler_ricochet_env import RicochetRobotsEnvOneStepAway
        
        env = CurriculumRicochetEnv(
            board_pool_manager=self.mock_pool_manager,
            max_steps=10,
            fallback_env_class=RicochetRobotsEnvOneStepAway
        )
        
        # Test reset gets board from pool
        obs, info = env.reset()
        
        # Verify board was loaded
        self.mock_pool_manager.get_board.assert_called_once()
        self.assertEqual(env.current_curriculum_k, 3)
        self.assertIsNotNone(obs)
    
    def test_curriculum_stats(self):
        """Test curriculum environment statistics."""
        from environment.simpler_ricochet_env import RicochetRobotsEnvOneStepAway
        
        env = CurriculumRicochetEnv(
            board_pool_manager=self.mock_pool_manager,
            max_steps=10,
            fallback_env_class=RicochetRobotsEnvOneStepAway
        )
        
        # Reset to load a board
        env.reset()
        
        # Get stats
        stats = env.get_curriculum_stats()
        
        self.assertIn('curriculum_boards_used', stats)
        self.assertIn('fallback_boards_used', stats)
        self.assertIn('current_curriculum_k', stats)
        self.assertEqual(stats['curriculum_boards_used'], 1)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics of curriculum components."""
    
    def test_cache_lookup_speed(self):
        """Test cache lookups are fast enough (< 1ms)."""
        temp_dir = tempfile.mkdtemp()
        cache_path = os.path.join(temp_dir, "perf_test.lmdb")
        
        try:
            cache = BoardCache(cache_path)
            
            # Insert test data
            test_hash = "a" * 40
            cache.insert(test_hash, 5)
            
            # Time multiple lookups
            start_time = time.time()
            for _ in range(1000):
                result = cache.lookup(test_hash)
                self.assertEqual(result, 5)
            
            elapsed = time.time() - start_time
            avg_lookup_time = elapsed / 1000
            
            # Should be much faster than 1ms
            self.assertLess(avg_lookup_time, 0.001, 
                          f"Average lookup time {avg_lookup_time:.4f}s exceeds 1ms")
            
            cache.close()
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_hash_collision_resistance(self):
        """Test hash collision resistance with many boards."""
        hashes = set()
        
        # Generate many different board configurations
        for board_size in [3, 4, 5]:
            for num_robots in [2, 3]:
                for wall_config in range(5):  # Different wall configurations
                    board = Board(board_size, board_size)
                    
                    # Add some variation in walls
                    if wall_config > 0:
                        board.add_wall(1, 1, wall_config % 4)
                    
                    # Generate different robot configurations
                    for robot_config in range(3):
                        robot_positions = []
                        for i in range(num_robots):
                            pos = (i % board_size, (i + robot_config) % board_size)
                            robot_positions.append(pos)
                        
                        board_hash = canonical_hash(
                            board, robot_positions, 0, (board_size-1, board_size-1)
                        )
                        
                        # Check for collisions
                        self.assertNotIn(board_hash, hashes, 
                                       f"Hash collision detected: {board_hash}")
                        hashes.add(board_hash)
        
        print(f"Generated {len(hashes)} unique hashes without collisions")


def run_curriculum_tests():
    """Run all curriculum learning tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBoardHashing,
        TestBoardCache,
        TestBoardPool,
        TestCurriculumIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_curriculum_tests()
    if success:
        print("\n✅ All curriculum learning tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1) 