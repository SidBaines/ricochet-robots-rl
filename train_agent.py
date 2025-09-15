from __future__ import annotations

import argparse
import os
from typing import Callable
import json
import math
import copy

from env import RicochetRobotsEnv, fixed_layout_v0_one_move, fixed_layouts_v1_four_targets
from env.curriculum import create_curriculum_wrapper, create_curriculum_manager, create_default_curriculum, CurriculumConfig
try:
    from models.policies import SmallCNN  # type: ignore
    from models.convlstm import ConvLSTMFeaturesExtractor  # type: ignore
    from models.recurrent_policy import RecurrentActorCriticPolicy  # type: ignore
except ImportError:
    SmallCNN = None  # type: ignore
    ConvLSTMFeaturesExtractor = None  # type: ignore
    RecurrentActorCriticPolicy = None  # type: ignore


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


def make_curriculum_env_factory(args: argparse.Namespace) -> tuple[Callable[[], RicochetRobotsEnv], object]:
    """Return a thunk to create a curriculum environment instance based on CLI args."""
    # Create base environment factory for curriculum
    base_env_factory = make_env_factory(args)
    
    # Create curriculum configuration
    if args.curriculum_config is not None:
        # Load custom curriculum config from file
        with open(args.curriculum_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        curriculum_config = CurriculumConfig(**config_data)
    else:
        # Use default curriculum with CLI overrides
        curriculum_config = create_default_curriculum()
        # Override with CLI arguments
        curriculum_config.success_rate_threshold = args.curriculum_success_threshold
        curriculum_config.min_episodes_per_level = args.curriculum_min_episodes
        curriculum_config.success_rate_window_size = args.curriculum_window_size
        curriculum_config.advancement_check_frequency = args.curriculum_check_freq
    
    # Create shared curriculum manager
    curriculum_manager = create_curriculum_manager(
        curriculum_config=curriculum_config,
        initial_level=args.curriculum_initial_level,
        verbose=args.curriculum_verbose
    )
    
    # Return a function that creates a new curriculum wrapper each time
    def _fn():
        return create_curriculum_wrapper(
            base_env_factory=base_env_factory,
            curriculum_manager=curriculum_manager,
            verbose=args.curriculum_verbose
        )
    
    return _fn, curriculum_manager


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
    parser.add_argument("--obs-mode", choices=["image", "symbolic", "rgb_image"], default="image")
    
    # Profiling options
    parser.add_argument("--enable-profiling", action="store_true", help="Enable detailed profiling of training pipeline")
    parser.add_argument("--profiling-report", type=str, default="profiling_report.json", help="Path to save profiling report")
    parser.add_argument("--profiling-summary-freq", type=int, default=10000, help="Frequency of profiling summary prints (timesteps)")
    
    # Curriculum learning options
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--curriculum-config", type=str, help="Path to custom curriculum configuration JSON file")
    parser.add_argument("--curriculum-initial-level", type=int, default=0, help="Initial curriculum level (0-4)")
    parser.add_argument("--curriculum-verbose", action="store_true", default=True, help="Print curriculum progression messages")
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.8, help="Success rate threshold for curriculum advancement")
    parser.add_argument("--curriculum-min-episodes", type=int, default=100, help="Minimum episodes per curriculum level")
    parser.add_argument("--curriculum-window-size", type=int, default=200, help="Success rate window size for curriculum advancement")
    parser.add_argument("--curriculum-check-freq", type=int, default=50, help="Frequency of curriculum advancement checks (episodes)")

    # Algo
    parser.add_argument("--algo", choices=["ppo"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=0.0, help="Final LR for schedule")
    parser.add_argument("--lr-schedule", choices=["none", "linear", "cosine"], default="none")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-end", type=float, default=0.1, help="Final clip range for schedule")
    parser.add_argument("--clip-schedule", choices=["none", "linear", "cosine"], default="none")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    # Normalize advantage toggles
    parser.add_argument("--normalize-advantage", dest="normalize_advantage", action="store_true")
    parser.add_argument("--no-normalize-advantage", dest="normalize_advantage", action="store_false")
    parser.set_defaults(normalize_advantage=True)
    parser.add_argument("--device", default="auto", help="Device to use for training (auto, cpu, mps, cuda)")

    # Config and profiles
    parser.add_argument("--config", type=str, help="Path to hyperparameters JSON config")
    parser.add_argument("--profile", choices=["none", "research_plan", "quick_debug", "baseline_sb3"], default="none")

    # VecNormalize options
    parser.add_argument("--vecnorm", action="store_true", help="Enable VecNormalize wrapper")
    parser.add_argument("--vecnorm-norm-obs", dest="vecnorm_norm_obs", action="store_true")
    parser.add_argument("--no-vecnorm-norm-obs", dest="vecnorm_norm_obs", action="store_false")
    parser.set_defaults(vecnorm_norm_obs=True)
    parser.add_argument("--vecnorm-norm-reward", dest="vecnorm_norm_reward", action="store_true")
    parser.add_argument("--no-vecnorm-norm-reward", dest="vecnorm_norm_reward", action="store_false")
    parser.set_defaults(vecnorm_norm_reward=False)
    parser.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    parser.add_argument("--vecnorm-clip-reward", type=float, default=10.0)

    # Logging / checkpoints
    parser.add_argument("--log-dir", default="runs/ppo")
    parser.add_argument("--save-path", default="checkpoints/ppo_model")
    parser.add_argument("--save-freq", type=int, default=50000, help="Timesteps between checkpoints (0=disable)")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Timesteps between evals (0=disable)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    # Monitoring / logging toggles
    parser.add_argument("--monitor-tensorboard", dest="monitor_tensorboard", action="store_true")
    parser.add_argument("--no-monitor-tensorboard", dest="monitor_tensorboard", action="store_false")
    parser.set_defaults(monitor_tensorboard=True)
    parser.add_argument("--monitor-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--eval-fixedset", type=str, default=None, help="Path to fixed test set JSON")
    parser.add_argument("--solver-cache", type=str, default="eval/solver_cache.json")
    parser.add_argument("--traj-record-freq", type=int, default=50000)
    parser.add_argument("--traj-record-episodes", type=int, default=5)
    # Model options
    parser.add_argument("--small-cnn", action="store_true", help="Use custom SmallCNN feature extractor for image obs")
    parser.add_argument("--convlstm", action="store_true", help="Use ConvLSTM feature extractor for image obs")
    parser.add_argument("--lstm-channels", type=int, default=64, help="ConvLSTM hidden channels")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--lstm-repeats", type=int, default=1, help="Number of repeats per ConvLSTM timestep")

    args = parser.parse_args()

    # Apply profile presets (can be overridden by config/CLI later)
    if args.profile != "none":
        if args.profile == "research_plan":
            # Long-run, plan-aligned defaults
            args.timesteps = 5_000_000 if args.timesteps == 100000 else args.timesteps
            args.clip_range = 0.2
            args.clip_end = 0.05
            args.clip_schedule = "linear"
            args.gae_lambda = 0.95
            args.vf_coef = 0.5
            args.max_grad_norm = 0.5
            args.lr = 3e-4
            args.lr_end = 3e-5
            args.lr_schedule = "linear"
            args.normalize_advantage = True
            # Sparse reward: avoid reward normalization by default
            if not hasattr(args, 'vecnorm') or not args.vecnorm:
                args.vecnorm = False
                args.vecnorm_norm_obs = True
                args.vecnorm_norm_reward = False
        elif args.profile == "quick_debug":
            args.timesteps = 50_000
            args.lr = 1e-3
            args.lr_end = 1e-3
            args.lr_schedule = "none"
            args.clip_range = 0.2
            args.clip_end = 0.2
            args.clip_schedule = "none"
            args.vf_coef = 0.5
            args.max_grad_norm = 0.5
            args.normalize_advantage = True
            args.vecnorm = False
        elif args.profile == "baseline_sb3":
            # Mirror common SB3-ish defaults while keeping our script params
            args.lr = 3e-4
            args.lr_schedule = "none"
            args.clip_range = 0.2
            args.clip_schedule = "none"
            args.gae_lambda = 0.95
            args.vf_coef = 0.5
            args.max_grad_norm = 0.5
            args.normalize_advantage = True
            args.vecnorm = False

    # Load config JSON and overlay into args (CLI wins if explicitly set differently)
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # Flat key overlay: only set attributes present in args
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load config {args.config}: {e}")

    # Initialize profiling if enabled
    if args.enable_profiling:
        try:
            from profiling import get_profiler, print_profiling_summary, save_profiling_report
            profiler = get_profiler()
            profiler.enable()
            print("Profiling enabled - detailed performance tracking active")
        except ImportError:
            print("Warning: Profiling requested but profiling module not available")
            args.enable_profiling = False

    # Device detection for M1/M2 Macs
    if args.device == "auto":
        import torch
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("Using MPS (Metal Performance Shaders) for GPU acceleration on M1/M2 Mac")
        elif torch.cuda.is_available():
            args.device = "cuda"
            print("Using CUDA for GPU acceleration")
        else:
            args.device = "cpu"
            print("Using CPU for training")
    else:
        print(f"Using specified device: {args.device}")

    # Performance knobs per backend
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore
    if torch is not None and args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Prefer TF32/matmul medium precision where beneficial
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    # MPS backend does not expose cudnn knobs; keep defaults

    # Lazy import SB3 to avoid hard dependency for env users
    import importlib
    sb3 = importlib.import_module("stable_baselines3")
    PPO = sb3.PPO
    vec_env_mod = importlib.import_module("stable_baselines3.common.env_util")
    make_vec_env = vec_env_mod.make_vec_env

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Choose environment factory based on curriculum setting
    curriculum_manager = None
    if args.curriculum:
        env_factory, curriculum_manager = make_curriculum_env_factory(args)
        print("Curriculum learning enabled - will progressively increase difficulty")
    else:
        env_factory = make_env_factory(args)
        print("Standard training mode - fixed difficulty")

    vec_env = make_vec_env(env_factory, n_envs=args.n_envs, seed=args.seed)

    # Optional VecNormalize wrapper
    try:
        from stable_baselines3.common.vec_env import VecNormalize as _VecNormalize
    except ImportError:
        _VecNormalize = None  # type: ignore
    vecnorm_used = False
    if args.vecnorm and _VecNormalize is not None:
        vec_env = _VecNormalize(
            vec_env,
            training=True,
            norm_obs=args.vecnorm_norm_obs,
            norm_reward=args.vecnorm_norm_reward,
            clip_obs=args.vecnorm_clip_obs,
            clip_reward=args.vecnorm_clip_reward,
        )
        vecnorm_used = True

    # Choose policy: for tiny grids (<8), prefer MlpPolicy to avoid NatureCNN kernel issues
    tiny_grid = False
    if args.env_mode in ("v0", "v1"):
        tiny_grid = True
    else:
        tiny_grid = min(args.height, args.width) < 8
    
    # Validate observation mode and policy compatibility
    if args.obs_mode == "symbolic" and args.convlstm:
        raise ValueError("ConvLSTM requires image observations, but symbolic mode was selected")
    
    # Policy selection based on observation mode and requirements
    if args.obs_mode in ["image", "rgb_image"]:
        if args.convlstm and ConvLSTMFeaturesExtractor is not None:
            # Use ConvLSTM for image observations (preferred per research plan)
            if RecurrentActorCriticPolicy is None:
                raise ImportError("RecurrentActorCriticPolicy not available. Please check the import.")
            policy = RecurrentActorCriticPolicy
            policy_kwargs = dict(
                features_extractor_class=ConvLSTMFeaturesExtractor,
                features_extractor_kwargs=dict(
                    features_dim=128,
                    lstm_channels=args.lstm_channels,
                    num_lstm_layers=args.lstm_layers,
                    num_repeats=args.lstm_repeats,
                    use_pool_and_inject=True,
                    use_skip_connections=True,
                ),
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                normalize_images=(args.obs_mode == "rgb_image"),  # Normalize RGB images
            )
        elif not tiny_grid:
            # Fallback to CNN for image observations
            policy = "CnnPolicy"
            policy_kwargs = dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]), 
                normalize_images=(args.obs_mode == "rgb_image")  # Normalize RGB images
            )
            if SmallCNN is not None and args.small_cnn:
                policy_kwargs.update(dict(features_extractor_class=SmallCNN, features_extractor_kwargs=dict(features_dim=128)))
            elif not args.convlstm:
                print("Warning: Using CNN instead of ConvLSTM for image observations. Use --convlstm for ConvLSTM.")
        else:
            # Use MLP for very small grids
            policy = "MlpPolicy"
            policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), normalize_images=False)
    elif args.obs_mode == "symbolic":
        # Use MLP for symbolic observations
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), normalize_images=False)
    else:
        raise ValueError(f"Unknown observation mode: {args.obs_mode}")

    # Schedule factories for LR and clip_range
    def _make_schedule(kind: str, start: float, end: float) -> Callable[[float], float] | float:
        if kind == "none":
            return start
        if kind == "linear":
            # progress_remaining: 1 -> 0
            return lambda progress_remaining: end + (start - end) * progress_remaining
        if kind == "cosine":
            # Cosine decay from start to end over progress_remaining
            def _fn(progress_remaining: float) -> float:
                t = 1.0 - progress_remaining  # 0->1
                w = 0.5 * (1 + math.cos(math.pi * t))
                return end + (start - end) * w
            return _fn
        return start

    lr_value: Callable[[float], float] | float = _make_schedule(args.lr_schedule, args.lr, args.lr_end)
    clip_value: Callable[[float], float] | float = _make_schedule(args.clip_schedule, args.clip_range, args.clip_end)

    model = PPO(
        policy,
        vec_env,
        learning_rate=lr_value,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=clip_value,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        normalize_advantage=args.normalize_advantage,
        tensorboard_log=args.log_dir,
        device=args.device,
        verbose=1,
    )

    # Callbacks: periodic checkpoint and eval success-rate logging + monitoring hub
    callbacks = []
    # Monitoring hub setup
    try:
        from monitoring import MonitoringHub, TensorBoardBackend, WandbBackend
        from monitoring.collectors import EpisodeStatsCollector, ActionUsageCollector, CurriculumProgressCollector
        monitoring_backends = []
        if args.monitor_tensorboard:
            try:
                monitoring_backends.append(TensorBoardBackend(args.log_dir))
            except ImportError:
                print("Warning: TensorBoard not available; disabling TB backend")
        if args.monitor_wandb:
            try:
                monitoring_backends.append(WandbBackend(project=(args.wandb_project or "ricochet-robots"), entity=args.wandb_entity, run_name=args.wandb_run_name, tags=args.wandb_tags, config={"args": vars(args)}))
            except ImportError:
                print("Warning: wandb requested but not available; skipping")
        hub = MonitoringHub(backends=monitoring_backends, collectors=[
            EpisodeStatsCollector(window_size=200),
            ActionUsageCollector(track_noop=True),
            CurriculumProgressCollector(),
        ])
    except ImportError:
        hub = None  # type: ignore
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
        from stable_baselines3.common.env_util import make_vec_env as _make
        
        # Custom callback for curriculum learning
        class CurriculumCallback(BaseCallback):
            def __init__(self, curriculum_manager, verbose=0):
                super().__init__(verbose)
                self.curriculum_manager = curriculum_manager
                self.last_logged_level = -1
                
            def _on_step(self) -> bool:
                # Log curriculum stats if available
                if self.curriculum_manager is not None:
                    stats = self.curriculum_manager.get_curriculum_stats()
                    
                    # Log to console
                    if self.verbose > 0 and self.num_timesteps % 1000 == 0:
                        print(f"Curriculum Level {stats['current_level']}: {stats['level_name']} "
                              f"(Success: {stats['success_rate']:.2f}, Episodes: {stats['episodes_at_level']})")
                    
                    # Log to TensorBoard
                    if hasattr(self, 'logger') and self.logger is not None:
                        self.logger.record("curriculum/level", stats['current_level'])
                        self.logger.record("curriculum/success_rate", stats['success_rate'])
                        self.logger.record("curriculum/episodes_at_level", stats['episodes_at_level'])
                        self.logger.record("curriculum/total_episodes", stats['total_episodes'])
                        self.logger.record("curriculum/window_size", stats['window_size'])
                        
                        # Log level advancement
                        if stats['current_level'] != self.last_logged_level:
                            self.logger.record("curriculum/level_advancement", stats['current_level'])
                            self.last_logged_level = stats['current_level']
                
                return True
        
        # Add curriculum callback if curriculum learning is enabled
        if args.curriculum and curriculum_manager is not None:
            curriculum_cb = CurriculumCallback(curriculum_manager=curriculum_manager, verbose=1)
            callbacks.append(curriculum_cb)
            # Also forward level changes to monitoring hub
            if hub is not None:
                class _HubCurriculumForward(BaseCallback):
                    def __init__(self, manager, verbose=0):
                        super().__init__(verbose)
                        self.manager = manager
                        self._last_level = None
                    def _on_step(self) -> bool:
                        lvl = self.manager.get_current_level()
                        if self._last_level != lvl:
                            stats = self.manager.get_curriculum_stats()
                            hub.on_level_change({
                                "global_step": self.num_timesteps,
                                "level_id": stats.get("current_level"),
                                "level_name": stats.get("level_name"),
                                "level_stats": stats,
                            })
                            self._last_level = lvl
                        return True
                callbacks.append(_HubCurriculumForward(curriculum_manager, verbose=0))
        
        # Add profiling callback if profiling is enabled
        if args.enable_profiling:
            class ProfilingCallback(BaseCallback):
                def __init__(self, summary_freq: int, verbose=0):
                    super().__init__(verbose)
                    self.summary_freq = summary_freq
                    self.last_summary = 0
                    
                def _on_step(self) -> bool:
                    # Print profiling summary periodically
                    if self.num_timesteps - self.last_summary >= self.summary_freq:
                        if self.verbose > 0:
                            print(f"\n--- Profiling Summary at {self.num_timesteps} timesteps ---")
                            print_profiling_summary(sort_by='total_time')
                        self.last_summary = self.num_timesteps
                    return True
                
                def _on_training_end(self) -> None:
                    # Save final profiling report
                    if hasattr(self, 'model') and self.model is not None:
                        save_profiling_report(args.profiling_report)
                        print(f"Final profiling report saved to {args.profiling_report}")
            
            profiling_cb = ProfilingCallback(summary_freq=args.profiling_summary_freq, verbose=1)
            callbacks.append(profiling_cb)
        
        # Add recurrent callback for hidden state management
        if args.convlstm and RecurrentActorCriticPolicy is not None:
            class RecurrentCallback(BaseCallback):
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self.last_dones = None
                    
                def _on_step(self) -> bool:
                    # Reset hidden states when episodes end
                    if hasattr(self.model.policy, 'reset_hidden_states'):
                        # Check if any environments have finished episodes
                        if self.last_dones is not None:
                            # Reset hidden states for environments that just finished
                            for done in self.last_dones:
                                if done:
                                    # Reset hidden states for this environment
                                    # Note: This is a simplified approach - in practice, we'd need
                                    # to handle this more carefully with proper batch indexing
                                    pass
                        
                        # Store current dones for next step
                        self.last_dones = self.locals.get('dones', None)
                    return True
            
            recurrent_cb = RecurrentCallback(verbose=1)
            callbacks.append(recurrent_cb)
        
        # Wrap separate eval env
        if args.eval_freq > 0:
            # When curriculum is enabled, mirror the training env at the current level
            if args.curriculum and curriculum_manager is not None:
                # Freeze eval to current level to keep observation space consistent
                current_level = curriculum_manager.get_current_level()
                frozen_cfg = copy.deepcopy(curriculum_manager.config)
                frozen_cfg.max_level = current_level
                # Build a dedicated eval curriculum manager starting at the frozen level
                frozen_manager = create_curriculum_manager(
                    curriculum_config=frozen_cfg,
                    initial_level=current_level,
                    verbose=args.curriculum_verbose,
                )
                # Create an eval env factory using the frozen curriculum
                def _eval_env_factory():
                    return create_curriculum_wrapper(
                        base_env_factory=make_env_factory(args),
                        curriculum_manager=frozen_manager,
                        verbose=args.curriculum_verbose,
                    )
                eval_env = _make(_eval_env_factory, n_envs=1, seed=args.seed)
            else:
                # Standard (non-curriculum) eval env
                eval_env = _make(make_env_factory(args), n_envs=1, seed=args.seed)

            # If using VecNormalize in training, wrap eval env and sync stats
            if vecnorm_used and _VecNormalize is not None:
                eval_env = _VecNormalize(
                    eval_env,
                    training=False,
                    norm_obs=args.vecnorm_norm_obs,
                    norm_reward=args.vecnorm_norm_reward,
                    clip_obs=args.vecnorm_clip_obs,
                    clip_reward=args.vecnorm_clip_reward,
                )
                # Sync running stats
                if hasattr(vec_env, 'obs_rms'):
                    eval_env.obs_rms = vec_env.obs_rms  # type: ignore[attr-defined]
                if hasattr(vec_env, 'ret_rms'):
                    eval_env.ret_rms = vec_env.ret_rms  # type: ignore[attr-defined]
            eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.dirname(args.save_path),
                                   log_path=args.log_dir, eval_freq=args.eval_freq,
                                   n_eval_episodes=args.eval_episodes, deterministic=True, render=False)
            callbacks.append(eval_cb)
            # Add fixed-set evaluation with solver-aware optimality gap via monitoring hub if configured
            if hub is not None and args.eval_fixedset is not None:
                try:
                    import os as _os
                    from monitoring.evaluators import FixedSetEvaluator, OptimalityGapEvaluator, SolverCache, TrajectoryRecorder
                    # Load fixed set
                    with open(args.eval_fixedset, 'r', encoding='utf-8') as f:
                        fixedset = json.load(f)
                    # Factories to build env from fixed layout dict
                    def _make_env_from_layout(layout_dict: dict):
                        from env import RicochetRobotsEnv, FixedLayout
                        import numpy as _np
                        h = int(layout_dict["height"]) ; w = int(layout_dict["width"]) ; tr = int(layout_dict["target_robot"]) ; robots = {int(k): tuple(v) for k, v in layout_dict["robot_positions"].items()}
                        h_walls = _np.array(layout_dict["h_walls"], dtype=bool)
                        v_walls = _np.array(layout_dict["v_walls"], dtype=bool)
                        goal = tuple(layout_dict["goal_position"])  # type: ignore
                        fl = FixedLayout(height=h, width=w, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=tr)
                        return RicochetRobotsEnv(fixed_layout=fl, obs_mode=args.obs_mode, channels_first=True)
                    solver_cache = SolverCache.load(args.solver_cache)
                    fixed_eval = FixedSetEvaluator(fixedset=fixedset, episodes=args.eval_episodes)
                    gap_eval = OptimalityGapEvaluator(fixedset=fixedset, solver_cache=solver_cache, solver_limits={"max_depth": args.solver_max_depth, "max_nodes": args.solver_max_nodes}, skip_unsolved=True)

                    class _HubEvalForward(BaseCallback):
                        def __init__(self, verbose=0):
                            super().__init__(verbose)
                            self._last_traj_dump = 0
                        def _on_step(self) -> bool:
                            if self.num_timesteps % args.eval_freq == 0:
                                fixed_eval.evaluate(hub, _make_env_from_layout, self.model, step=self.num_timesteps)
                                gap_eval.evaluate(hub, _make_env_from_layout, self.model, step=self.num_timesteps)
                            # Trajectory recorder on its own cadence using a standard training env factory
                            if args.traj_record_freq > 0 and (self.num_timesteps - self._last_traj_dump) >= args.traj_record_freq:
                                rec = TrajectoryRecorder(episodes=args.traj_record_episodes)
                                # Use a fresh non-curriculum single-env factory mirroring current args
                                rec_env_factory = make_env_factory(args)
                                trajectories = rec.record(rec_env_factory, self.model)
                                hub.log_text("eval/trajectories/sample", json.dumps({"num": len(trajectories)}), self.num_timesteps)
                                self._last_traj_dump = self.num_timesteps
                            return True
                    callbacks.append(_HubEvalForward(verbose=0))
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Failed to configure fixed-set evaluator: {e}")
        if args.save_freq > 0:
            ckpt_cb = CheckpointCallback(save_freq=args.save_freq, save_path=os.path.dirname(args.save_path),
                                         name_prefix=os.path.basename(args.save_path))
            callbacks.append(ckpt_cb)
    except ImportError:
        pass

    # Wrap training with profiling
    if args.enable_profiling:
        try:
            from profiling import profile
            with profile("training_total", track_memory=True):
                model.learn(total_timesteps=args.timesteps, callback=callbacks if callbacks else None)
        except ImportError:
            model.learn(total_timesteps=args.timesteps, callback=callbacks if callbacks else None)
    else:
        model.learn(total_timesteps=args.timesteps, callback=callbacks if callbacks else None)
    
    model.save(args.save_path)
    # Save VecNormalize statistics if used
    if vecnorm_used and _VecNormalize is not None:
        try:
            vecnorm_path = os.path.join(args.log_dir, "vecnormalize.pkl")
            vec_env.save(vecnorm_path)  # type: ignore[attr-defined]
            print(f"Saved VecNormalize statistics to {vecnorm_path}")
        except (AttributeError, OSError) as e:
            print(f"Warning: Failed to save VecNormalize stats: {e}")
    
    # Print final profiling summary if enabled
    if args.enable_profiling:
        try:
            print("\n" + "="*80)
            print("FINAL PROFILING SUMMARY")
            print("="*80)
            print_profiling_summary(sort_by='total_time')
            save_profiling_report(args.profiling_report)
            print(f"Final profiling report saved to {args.profiling_report}")
        except ImportError:
            pass

    # Close monitoring hub
    try:
        if hub is not None:
            hub.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


