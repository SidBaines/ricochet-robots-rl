import numpy as np
import pytest
import torch


pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available"),
]


def test_stock_cnn_learns_saves_and_reloads_on_mps(tmp_path) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    from src.env.ricochet_env import RicochetRobotsEnv

    def make_env():
        return RicochetRobotsEnv(
            height=8,
            width=8,
            num_robots=2,
            include_noop=False,
            max_steps=8,
            obs_mode="rgb_cell_image",
            channels_first=True,
            cell_obs_pixel_size=40,
            seed=17,
        )

    vec_env = make_vec_env(make_env, n_envs=1, seed=17)
    model = PPO(
        "CnnPolicy",
        vec_env,
        device="mps",
        seed=17,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"normalize_images": True},
        verbose=0,
    )

    assert next(model.policy.parameters()).device.type == "mps"
    model.learn(total_timesteps=8)
    assert all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters())

    checkpoint = tmp_path / "mps_model"
    model.save(checkpoint)
    loaded = PPO.load(checkpoint, env=vec_env, device="mps")
    action, _ = loaded.predict(vec_env.reset(), deterministic=True)

    assert next(loaded.policy.parameters()).device.type == "mps"
    assert np.asarray(action).shape == (1,)
    vec_env.close()
