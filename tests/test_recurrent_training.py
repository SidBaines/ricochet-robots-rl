"""
Tests for recurrent training behavior, hidden state management, and RNN stability.

These tests verify that:
1. Hidden states are properly reset on episode boundaries per environment
2. Async episode terminations are handled correctly
3. RNN training remains stable on longer episodes
4. Memory usage is reasonable under typical batch sizes
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from models.recurrent_policy import RecurrentActorCriticPolicy
from models.convlstm import ConvLSTMFeaturesExtractor
from env import RicochetRobotsEnv
from gymnasium.spaces import Box, Discrete


class TestRecurrentPolicyHiddenStates:
    """Test hidden state lifecycle and per-env resets."""
    
    @pytest.fixture
    def policy(self):
        """Create a recurrent policy for testing."""
        obs_space = Box(0, 1, (3, 8, 8), dtype=np.float32)
        action_space = Discrete(4)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4
        
        policy = RecurrentActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ConvLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=64,
                lstm_channels=16,
                num_lstm_layers=2,
                num_repeats=1,
                use_pool_and_inject=True,
            ),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        )
        return policy
    
    def test_hidden_state_initialization(self, policy):
        """Test that hidden states are properly initialized."""
        batch_size = 4
        obs = torch.randn(batch_size, 3, 8, 8)
        
        # First forward pass should initialize hidden states
        action, value, log_prob = policy.forward(obs)
        
        assert action.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        
        # Hidden states should be initialized
        hidden_states = policy.get_hidden_states()
        assert hidden_states is not None
        assert len(hidden_states) == 2  # 2 LSTM layers
        
        for h, c in hidden_states:
            assert h.shape[0] == batch_size  # Batch dimension
            assert c.shape[0] == batch_size
    
    def test_episode_starts_reset(self, policy):
        """Test that episode_starts properly resets hidden states for specific envs."""
        batch_size = 4
        obs = torch.randn(batch_size, 3, 8, 8)
        
        # Forward pass to initialize hidden states
        policy.forward(obs)
        hidden_states_before = policy.get_hidden_states()
        
        # Set some environments to episode start
        episode_starts = [True, False, True, False]
        policy.set_episode_starts(episode_starts)
        
        # Forward pass should reset hidden states for envs 0 and 2
        policy.forward(obs)
        hidden_states_after = policy.get_hidden_states()
        
        # Check that envs 0 and 2 have been reset (should be zeros)
        for i, (h, c) in enumerate(hidden_states_after):
            # Env 0 should be reset
            assert torch.allclose(h[0], torch.zeros_like(h[0])), f"Env 0 hidden state not reset in layer {i}"
            assert torch.allclose(c[0], torch.zeros_like(c[0])), f"Env 0 cell state not reset in layer {i}"
            
            # Env 2 should be reset
            assert torch.allclose(h[2], torch.zeros_like(h[2])), f"Env 2 hidden state not reset in layer {i}"
            assert torch.allclose(c[2], torch.zeros_like(c[2])), f"Env 2 cell state not reset in layer {i}"
            
            # Envs 1 and 3 should retain their states (approximately)
            if not torch.allclose(h[1], hidden_states_before[i][0][1], atol=1e-6):
                # Allow some change due to detaching, but should not be zero
                assert not torch.allclose(h[1], torch.zeros_like(h[1])), f"Env 1 hidden state was reset in layer {i}"
            if not torch.allclose(h[3], hidden_states_before[i][0][3], atol=1e-6):
                assert not torch.allclose(h[3], torch.zeros_like(h[3])), f"Env 3 hidden state was reset in layer {i}"
    
    def test_batch_size_change_resets_states(self, policy):
        """Test that changing batch size resets all hidden states."""
        # First batch
        obs1 = torch.randn(2, 3, 8, 8)
        policy.forward(obs1)
        hidden_states_1 = policy.get_hidden_states()
        
        # Different batch size should reset
        obs2 = torch.randn(3, 3, 8, 8)
        policy.forward(obs2)
        hidden_states_2 = policy.get_hidden_states()
        
        # Should have different batch sizes
        assert hidden_states_1[0][0].shape[0] == 2
        assert hidden_states_2[0][0].shape[0] == 3
    
    def test_evaluate_actions_consistency(self, policy):
        """Test that evaluate_actions produces consistent results with forward."""
        batch_size = 3
        obs = torch.randn(batch_size, 3, 8, 8)
        actions = torch.randint(0, 4, (batch_size,))
        
        # Forward pass
        action_out, value_out, log_prob_out = policy.forward(obs)
        
        # Evaluate actions
        value_eval, log_prob_eval, entropy = policy.evaluate_actions(obs, actions)
        
        # Values should be identical
        assert torch.allclose(value_out, value_eval), "Values from forward and evaluate_actions differ"
        
        # Log probs should match for the same actions
        log_prob_forward = policy._get_action_dist_from_latent(policy.action_net(policy.features_extractor(obs)[0])).log_prob(actions)
        assert torch.allclose(log_prob_eval, log_prob_forward), "Log probs from evaluate_actions inconsistent"


class TestAsyncEpisodeTermination:
    """Test handling of asynchronous episode terminations across environments."""
    
    def test_mixed_episode_lengths(self):
        """Test that different episode lengths don't cause hidden state leakage."""
        # Create a simple environment that can terminate at different times
        obs_space = Box(0, 1, (3, 4, 4), dtype=np.float32)
        action_space = Discrete(4)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4
        
        policy = RecurrentActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ConvLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=32,
                lstm_channels=8,
                num_lstm_layers=1,
                num_repeats=1,
            ),
            net_arch=dict(pi=[32], vf=[32]),
            normalize_images=False,
        )
        
        batch_size = 3
        obs = torch.randn(batch_size, 3, 4, 4)
        
        # Simulate different episode termination patterns
        for step in range(5):
            # Some environments terminate at different steps
            if step == 2:
                # Env 0 terminates
                policy.set_episode_starts([True, False, False])
            elif step == 4:
                # Env 1 terminates
                policy.set_episode_starts([False, True, False])
            
            action, value, log_prob = policy.forward(obs)
            
            # Hidden states should be properly managed
            hidden_states = policy.get_hidden_states()
            assert hidden_states is not None
            assert len(hidden_states) == 1  # 1 LSTM layer
            
            # After termination, those envs should have reset states
            if step >= 2:
                h, c = hidden_states[0]
                # Env 0 should be reset after step 2
                if step > 2:
                    assert torch.allclose(h[0], torch.zeros_like(h[0])), f"Env 0 not reset at step {step}"
            
            if step >= 4:
                h, c = hidden_states[0]
                # Env 1 should be reset after step 4
                if step > 4:
                    assert torch.allclose(h[1], torch.zeros_like(h[1])), f"Env 1 not reset at step {step}"


class TestRNNTrainingStability:
    """Test RNN training stability on longer episodes."""
    
    def test_long_episode_stability(self):
        """Test that RNN remains stable during long episodes."""
        obs_space = Box(0, 1, (3, 6, 6), dtype=np.float32)
        action_space = Discrete(4)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4
        
        policy = RecurrentActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ConvLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=64,
                lstm_channels=16,
                num_lstm_layers=2,
                num_repeats=2,  # Multiple repeats per step
            ),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        )
        
        batch_size = 2
        obs = torch.randn(batch_size, 3, 6, 6)
        
        # Track hidden state norms over many steps
        hidden_norms = []
        values = []
        
        for step in range(50):  # Long episode simulation
            action, value, log_prob = policy.forward(obs)
            
            # Record hidden state norms
            hidden_states = policy.get_hidden_states()
            if hidden_states:
                h_norm = torch.norm(hidden_states[0][0]).item()
                hidden_norms.append(h_norm)
            
            values.append(value.mean().item())
            
            # Simulate some observation change
            obs = obs + 0.01 * torch.randn_like(obs)
        
        # Hidden state norms should not explode or vanish
        assert all(0.1 < norm < 100.0 for norm in hidden_norms), f"Hidden state norms out of range: {hidden_norms[:5]}"
        
        # Values should remain reasonable
        assert all(-10.0 < v < 10.0 for v in values), f"Values out of range: {values[:5]}"
        
        # No NaN or Inf values
        assert all(torch.isfinite(torch.tensor(hidden_norms))), "Hidden state norms contain NaN/Inf"
        assert all(torch.isfinite(torch.tensor(values))), "Values contain NaN/Inf"
    
    def test_gradient_stability(self):
        """Test that gradients remain stable during training."""
        obs_space = Box(0, 1, (3, 4, 4), dtype=np.float32)
        action_space = Discrete(4)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4
        
        policy = RecurrentActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ConvLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=32,
                lstm_channels=8,
                num_lstm_layers=1,
                num_repeats=1,
            ),
            net_arch=dict(pi=[32], vf=[32]),
            normalize_images=False,
        )
        
        batch_size = 4
        obs = torch.randn(batch_size, 3, 4, 4, requires_grad=True)
        actions = torch.randint(0, 4, (batch_size,))
        
        # Forward pass
        action, value, log_prob = policy.forward(obs)
        
        # Compute loss
        loss = -log_prob.mean() + 0.5 * (value - 1.0).pow(2).mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0.0
        param_count = 0
        
        for param in policy.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                
                # Individual parameter gradients should be reasonable
                assert not torch.isnan(param.grad).any(), "NaN gradients detected"
                assert not torch.isinf(param.grad).any(), "Inf gradients detected"
                assert grad_norm < 100.0, f"Gradient norm too large: {grad_norm}"
        
        total_grad_norm = (total_grad_norm ** 0.5) / max(param_count, 1)
        assert total_grad_norm < 50.0, f"Total gradient norm too large: {total_grad_norm}"


class TestMemoryUsage:
    """Test memory usage under typical batch sizes."""
    
    def test_memory_usage_batch_sizes(self):
        """Test memory usage across different batch sizes."""
        obs_space = Box(0, 1, (3, 8, 8), dtype=np.float32)
        action_space = Discrete(4)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4
        
        policy = RecurrentActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ConvLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=64,
                lstm_channels=16,
                num_lstm_layers=2,
                num_repeats=1,
            ),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        )
        
        batch_sizes = [1, 4, 8, 16]
        memory_usage = []
        
        for batch_size in batch_sizes:
            obs = torch.randn(batch_size, 3, 8, 8)
            
            # Measure memory before and after forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            action, value, log_prob = policy.forward(obs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append(memory_after - memory_before)
            else:
                # For CPU, just check that computation succeeds
                memory_usage.append(0)
            
            # Reset for next batch size
            policy.reset_hidden_states()
        
        # Memory usage should scale reasonably with batch size
        if torch.cuda.is_available() and any(m > 0 for m in memory_usage):
            # Memory usage should generally increase with batch size
            for i in range(1, len(memory_usage)):
                if memory_usage[i] > 0 and memory_usage[i-1] > 0:
                    # Allow some variance but should generally increase
                    assert memory_usage[i] >= memory_usage[i-1] * 0.5, f"Memory usage doesn't scale properly: {memory_usage}"
    
    def test_hidden_state_memory_cleanup(self):
        """Test that hidden states are properly cleaned up."""
        obs_space = Box(0, 1, (3, 4, 4), dtype=np.float32)
        action_space = Discrete(4)
        
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4
        
        policy = RecurrentActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ConvLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=32,
                lstm_channels=8,
                num_lstm_layers=1,
                num_repeats=1,
            ),
            net_arch=dict(pi=[32], vf=[32]),
            normalize_images=False,
        )
        
        # Forward pass with large batch
        obs = torch.randn(16, 3, 4, 4)
        policy.forward(obs)
        
        # Check that hidden states exist
        hidden_states = policy.get_hidden_states()
        assert hidden_states is not None
        
        # Reset hidden states
        policy.reset_hidden_states()
        
        # Hidden states should be cleared
        assert policy.get_hidden_states() is None
        
        # New forward pass should work
        obs_small = torch.randn(2, 3, 4, 4)
        action, value, log_prob = policy.forward(obs_small)
        
        # Should have new hidden states with correct batch size
        new_hidden_states = policy.get_hidden_states()
        assert new_hidden_states is not None
        assert new_hidden_states[0][0].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__])
