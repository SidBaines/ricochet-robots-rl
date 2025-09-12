import pytest

from env import RicochetRobotsEnv, fixed_layout_v0_one_move, fixed_layouts_v1_four_targets


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def make_env_from_layout(layout, obs_mode="image", include_noop=False):
    return RicochetRobotsEnv(
        fixed_layout=layout,
        include_noop=include_noop,
        step_penalty=-0.01,
        goal_reward=1.0,
        max_steps=5,
        obs_mode=obs_mode,
        channels_first=True,
    )


def _sb3_available():
    try:
        import importlib
        importlib.import_module("stable_baselines3")
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sb3_available(), reason="SB3/torch not installed")
def test_v0_learns_single_move_quickly():
    try:
        import importlib
        PPO = importlib.import_module("stable_baselines3").PPO  # type: ignore[attr-defined]
        make_vec_env = importlib.import_module("stable_baselines3.common.env_util").make_vec_env  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("SB3 not available at runtime")

    layout = fixed_layout_v0_one_move()
    def _make():
        env = make_env_from_layout(layout, obs_mode="image", include_noop=False)
        return env

    vec_env = make_vec_env(_make, n_envs=4)

    policy_kwargs = dict(
        net_arch=dict(pi=[64], vf=[64]),
        normalize_images=False,
    )
    # Use MlpPolicy for tiny 4x4 grid to avoid NatureCNN kernel constraints
    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=64, batch_size=128, n_epochs=4,
                gamma=0.99, ent_coef=0.01, policy_kwargs=policy_kwargs, device="cpu")

    model.learn(total_timesteps=2000)

    # Evaluate greedily
    eval_env = _make()
    obs, info = eval_env.reset()
    success = 0
    episodes = 20
    for _ in range(episodes):
        done = False
        truncated = False
        obs, info = eval_env.reset()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = eval_env.step(int(action))
        if info.get("is_success", False):
            success += 1
    # Expect high success on this trivial task
    assert success >= int(0.9 * episodes)


@pytest.mark.skipif(not _sb3_available(), reason="SB3/torch not installed")
def test_v1_learns_direction_selection():
    try:
        import importlib
        PPO = importlib.import_module("stable_baselines3").PPO  # type: ignore[attr-defined]
        DummyVecEnv = importlib.import_module("stable_baselines3.common.vec_env").DummyVecEnv  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("SB3 not available at runtime")

    layouts = fixed_layouts_v1_four_targets()

    def _make_idx(idx: int):
        def fn():
            return make_env_from_layout(layouts[idx], obs_mode="image", include_noop=False)
        return fn

    # Mix four envs equally in the vectorized env
    makers = [_make_idx(i) for i in range(4)]
    # SB3's make_vec_env can't mix factories, so we use DummyVecEnv manually
    vec_env = DummyVecEnv(makers)

    policy_kwargs = dict(net_arch=dict(pi=[128], vf=[128]), normalize_images=False)
    # Use MlpPolicy on small 5x5 grids
    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=64, batch_size=128, n_epochs=4,
                gamma=0.99, ent_coef=0.01, policy_kwargs=policy_kwargs, device="cpu")

    model.learn(total_timesteps=6000)

    # Evaluate on each layout
    success_counts = []
    episodes = 20
    for idx in range(4):
        env = makers[idx]()
        success = 0
        for _ in range(episodes):
            obs, info = env.reset()
            done = False
            trunc = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(int(action))
            if info.get("is_success", False):
                success += 1
        success_counts.append(success)

    # Expect high success on each of the four single-step layouts
    assert all(s >= int(0.8 * episodes) for s in success_counts)


