import numpy as np
import pytest

from env import RicochetRobotsEnv, fixed_layout_v0_one_move


def _sb3_available():
    try:
        import importlib
        importlib.import_module("stable_baselines3")
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sb3_available(), reason="SB3/torch not installed")
def test_policy_output_shapes():
    """Test that policies produce correct output shapes for both obs modes."""
    import importlib
    PPO = importlib.import_module("stable_baselines3").PPO  # type: ignore[attr-defined]
    
    layout = fixed_layout_v0_one_move()
    
    # Test image obs with MlpPolicy (tiny grid fallback)
    env_img = RicochetRobotsEnv(
        fixed_layout=layout,
        obs_mode="image",
        channels_first=True,
        include_noop=False,
    )
    
    model_img = PPO("MlpPolicy", env_img, policy_kwargs=dict(net_arch=dict(pi=[64], vf=[64])), verbose=0)
    
    obs, _ = env_img.reset()
    action, _ = model_img.predict(obs, deterministic=True)
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < env_img.action_space.n
    
    # Test symbolic obs with MlpPolicy
    env_sym = RicochetRobotsEnv(
        fixed_layout=layout,
        obs_mode="symbolic",
        include_noop=False,
    )
    
    model_sym = PPO("MlpPolicy", env_sym, policy_kwargs=dict(net_arch=dict(pi=[64], vf=[64])), verbose=0)
    
    obs, _ = env_sym.reset()
    action, _ = model_sym.predict(obs, deterministic=True)
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < env_sym.action_space.n


@pytest.mark.skipif(not _sb3_available(), reason="SB3/torch not installed")
def test_small_cnn_policy():
    """Test custom SmallCNN feature extractor if available."""
    try:
        from models.policies import SmallCNN  # type: ignore
    except ImportError:
        pytest.skip("SmallCNN not available")
    
    import importlib
    PPO = importlib.import_module("stable_baselines3").PPO  # type: ignore[attr-defined]
    
    layout = fixed_layout_v0_one_move()
    env = RicochetRobotsEnv(
        fixed_layout=layout,
        obs_mode="image",
        channels_first=True,
        include_noop=False,
    )
    
    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[64], vf=[64])
    )
    
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
    
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < env.action_space.n


@pytest.mark.skipif(not _sb3_available(), reason="SB3/torch not installed")
def test_ppo_smoke_run():
    """Brief PPO training run to ensure no runtime errors."""
    import importlib
    PPO = importlib.import_module("stable_baselines3").PPO  # type: ignore[attr-defined]
    make_vec_env = importlib.import_module("stable_baselines3.common.env_util").make_vec_env  # type: ignore[attr-defined]
    
    layout = fixed_layout_v0_one_move()
    
    def make_env():
        return RicochetRobotsEnv(
            fixed_layout=layout,
            obs_mode="image",
            channels_first=True,
            include_noop=False,
            max_steps=5,
        )
    
    vec_env = make_vec_env(make_env, n_envs=2, seed=42)
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[64], vf=[64])),
        verbose=0,
    )
    
    # Very short training run
    model.learn(total_timesteps=100)
    
    # Test prediction
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (2,)  # n_envs=2
    assert all(0 <= a < vec_env.action_space.n for a in action)
