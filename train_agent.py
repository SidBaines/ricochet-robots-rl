from __future__ import annotations

import argparse
import os
from typing import Callable

from env import RicochetRobotsEnv, fixed_layout_v0_one_move, fixed_layouts_v1_four_targets
try:
    from models.policies import SmallCNN  # type: ignore
    from models.convlstm import ConvLSTMFeaturesExtractor  # type: ignore
except ImportError:
    SmallCNN = None  # type: ignore
    ConvLSTMFeaturesExtractor = None  # type: ignore


def make_env_factory(args: argparse.Namespace) -> Callable[[], RicochetRobotsEnv]:
    """Return a thunk to create an environment instance based on CLI args."""
    mode = args.env_mode

    if mode == "v0":
        layout = fixed_layout_v0_one_move()
        def _fn():
            return RicochetRobotsEnv(
                height=layout.height,
                width=layout.width,
                num_robots=len(layout.robot_positions),
                fixed_layout=layout,
                include_noop=args.include_noop,
                step_penalty=args.step_penalty,
                goal_reward=args.goal_reward,
                max_steps=args.max_steps,
                obs_mode=args.obs_mode,
                channels_first=True,
            )
        return _fn

    if mode == "v1":
        layouts = fixed_layouts_v1_four_targets()
        # round-robin over layouts to add variety across envs
        counter = {"i": -1}
        def _fn():
            counter["i"] = (counter["i"] + 1) % len(layouts)
            layout = layouts[counter["i"]]
            return RicochetRobotsEnv(
                height=layout.height,
                width=layout.width,
                num_robots=len(layout.robot_positions),
                fixed_layout=layout,
                include_noop=args.include_noop,
                step_penalty=args.step_penalty,
                goal_reward=args.goal_reward,
                max_steps=args.max_steps,
                obs_mode=args.obs_mode,
                channels_first=True,
            )
        return _fn

    # default: random env sampling per episode
    def _fn():
        return RicochetRobotsEnv(
            height=args.height,
            width=args.width,
            num_robots=args.num_robots,
            include_noop=args.include_noop,
            step_penalty=args.step_penalty,
            goal_reward=args.goal_reward,
            noop_penalty=args.noop_penalty,
            max_steps=args.max_steps,
            seed=args.seed,
            ensure_solvable=args.ensure_solvable,
            solver_max_depth=args.solver_max_depth,
            solver_max_nodes=args.solver_max_nodes,
            obs_mode=args.obs_mode,
            channels_first=True,
        )
    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Ricochet Robots")
    # Env selection
    parser.add_argument("--env-mode", choices=["random", "v0", "v1"], default="random")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--include-noop", action="store_true")
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--noop-penalty", type=float, default=-0.01)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensure-solvable", action="store_true")
    parser.add_argument("--solver-max-depth", type=int, default=30)
    parser.add_argument("--solver-max-nodes", type=int, default=20000)
    parser.add_argument("--obs-mode", choices=["image", "symbolic"], default="image")

    # Algo
    parser.add_argument("--algo", choices=["ppo"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", default="auto")

    # Logging / checkpoints
    parser.add_argument("--log-dir", default="runs/ppo")
    parser.add_argument("--save-path", default="checkpoints/ppo_model")
    parser.add_argument("--save-freq", type=int, default=50000, help="Timesteps between checkpoints (0=disable)")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Timesteps between evals (0=disable)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    # Model options
    parser.add_argument("--small-cnn", action="store_true", help="Use custom SmallCNN feature extractor for image obs")
    parser.add_argument("--convlstm", action="store_true", help="Use ConvLSTM feature extractor for image obs")
    parser.add_argument("--lstm-channels", type=int, default=64, help="ConvLSTM hidden channels")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--lstm-repeats", type=int, default=1, help="Number of repeats per ConvLSTM timestep")

    args = parser.parse_args()

    # Lazy import SB3 to avoid hard dependency for env users
    import importlib
    sb3 = importlib.import_module("stable_baselines3")
    PPO = sb3.PPO
    vec_env_mod = importlib.import_module("stable_baselines3.common.env_util")
    make_vec_env = vec_env_mod.make_vec_env

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    env_factory = make_env_factory(args)

    vec_env = make_vec_env(env_factory, n_envs=args.n_envs, seed=args.seed)

    # Choose policy: for tiny grids (<8), prefer MlpPolicy to avoid NatureCNN kernel issues
    tiny_grid = False
    if args.env_mode in ("v0", "v1"):
        tiny_grid = True
    else:
        tiny_grid = min(args.height, args.width) < 8
    if args.obs_mode == "image" and not tiny_grid:
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), normalize_images=False)
    if args.obs_mode == "image" and SmallCNN is not None and args.small_cnn:
        policy_kwargs.update(dict(features_extractor_class=SmallCNN, features_extractor_kwargs=dict(features_dim=128)))
    elif args.obs_mode == "image" and ConvLSTMFeaturesExtractor is not None and args.convlstm:
        policy_kwargs.update(dict(
            features_extractor_class=ConvLSTMFeaturesExtractor, 
            features_extractor_kwargs=dict(
                features_dim=128,
                lstm_channels=args.lstm_channels,
                num_lstm_layers=args.lstm_layers,
                num_repeats=args.lstm_repeats,
            )
        ))

    model = PPO(
        policy,
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        device=args.device,
        verbose=1,
    )

    # Callbacks: periodic checkpoint and eval success-rate logging
    callbacks = []
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env as _make
        # Wrap separate eval env
        if args.eval_freq > 0:
            eval_env = _make(make_env_factory(args), n_envs=1, seed=args.seed)
            eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.dirname(args.save_path),
                                   log_path=args.log_dir, eval_freq=args.eval_freq,
                                   n_eval_episodes=args.eval_episodes, deterministic=True, render=False)
            callbacks.append(eval_cb)
        if args.save_freq > 0:
            ckpt_cb = CheckpointCallback(save_freq=args.save_freq, save_path=os.path.dirname(args.save_path),
                                         name_prefix=os.path.basename(args.save_path))
            callbacks.append(ckpt_cb)
    except ImportError:
        pass

    model.learn(total_timesteps=args.timesteps, callback=callbacks if callbacks else None)
    model.save(args.save_path)


if __name__ == "__main__":
    main()


