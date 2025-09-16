from __future__ import annotations

import argparse

from env import RicochetRobotsEnv, fixed_layout_v0_one_move, fixed_layouts_v1_four_targets


def make_single_env(args: argparse.Namespace) -> RicochetRobotsEnv:
    if args.env_mode == "v0":
        layout = fixed_layout_v0_one_move()
        return RicochetRobotsEnv(fixed_layout=layout, obs_mode=args.obs_mode, channels_first=True)
    if args.env_mode == "v1":
        layouts = fixed_layouts_v1_four_targets()
        return RicochetRobotsEnv(fixed_layout=layouts[0], obs_mode=args.obs_mode, channels_first=True)
    return RicochetRobotsEnv(height=args.height, width=args.width, num_robots=args.num_robots, obs_mode=args.obs_mode, channels_first=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-mode", choices=["random", "v0", "v1"], default="random")
    parser.add_argument("--obs-mode", choices=["image", "symbolic"], default="image")
    parser.add_argument("--recurrent", action="store_true", help="Load sb3-contrib RecurrentPPO model or custom recurrent policy")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-robots", type=int, default=2)
    args = parser.parse_args()

    import importlib
    PPO = importlib.import_module("stable_baselines3").PPO  # type: ignore[attr-defined]
    # Try to import RecurrentPPO for recurrent models
    try:
        RecurrentPPO = importlib.import_module("sb3_contrib").RecurrentPPO  # type: ignore[attr-defined]
    except Exception:
        RecurrentPPO = None  # type: ignore
    # Try to import custom recurrent policy for registration
    try:
        from models.recurrent_policy import RecurrentActorCriticPolicy  # type: ignore
        # Registering custom policies for SB3 load if needed happens implicitly by import
        _custom_policy_available = True
    except Exception:
        RecurrentActorCriticPolicy = None  # type: ignore
        _custom_policy_available = False

    env = make_single_env(args)
    policy_type = "CnnPolicy" if args.obs_mode == "image" else "MlpPolicy"
    if args.recurrent and RecurrentPPO is not None:
        model = RecurrentPPO.load(args.model_path)  # type: ignore[assignment]
    else:
        # Attempt to load with SB3; if the checkpoint references a custom policy, the prior import makes it available
        model = PPO.load(args.model_path)

    success = 0
    lengths = []
    # Use unified helper to support both sb3-contrib and custom recurrent policy
    try:
        from monitoring.evaluators import rollout_episode_with_recurrent_support
        make_env_fn = lambda: make_single_env(args)
        for _ in range(args.episodes):
            res = rollout_episode_with_recurrent_support(make_env_fn, model, deterministic=True)
            if res.get("is_success", False):
                success += 1
            lengths.append(int(res.get("length", 0)))
    except ImportError:
        # Fallback to basic loop
        for _ in range(args.episodes):
            obs, info = env.reset()
            done = False
            trunc = False
            steps = 0
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(int(action))
                steps += 1
            if info.get("is_success", False):
                success += 1
            lengths.append(steps)
        if info.get("is_success", False):
            success += 1
        lengths.append(steps)

    rate = success / args.episodes
    avg_len = sum(lengths) / max(1, len(lengths))
    print(f"Success rate: {rate:.3f} ({success}/{args.episodes}), avg length: {avg_len:.2f}")


if __name__ == "__main__":
    main()


