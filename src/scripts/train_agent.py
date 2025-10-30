from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

import argparse
import os
from typing import Callable, Any, Dict
import json
import math
import numbers
import copy
import yaml
from datetime import datetime

from src.env import RicochetRobotsEnv, fixed_layout_v0_one_move, fixed_layouts_v1_four_targets
from src.env.curriculum import (
    create_curriculum_wrapper, create_curriculum_manager, create_default_curriculum, CurriculumConfig,
    create_bank_curriculum_manager, BankCurriculumWrapper
)
from src.env.criteria_env import CriteriaFilteredEnv
from src.env.puzzle_bank import PuzzleBank
try:
    from src.models.policies import SmallCNN  # type: ignore
except ImportError:
    SmallCNN = None  # type: ignore
try:
    from src.models.drc_policy import DRCRecurrentPolicy  # type: ignore
except ImportError:
    DRCRecurrentPolicy = None  # type: ignore
try:
    from src.models.resnet import ResNetFeaturesExtractor, ResNetPolicy  # type: ignore
except ImportError:
    ResNetFeaturesExtractor = None  # type: ignore
    ResNetPolicy = None  # type: ignore


def build_training_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensure-solvable", action="store_true")
    parser.add_argument("--solver-max-depth", type=int, default=30)
    parser.add_argument("--solver-max-nodes", type=int, default=20000)
    parser.add_argument("--obs-mode", choices=["image", "symbolic", "rgb_image", "rgb_cell_image"], default="image")
    parser.add_argument("--cell-obs-pixel-size", type=int, default=128)

    # Profiling options
    parser.add_argument("--enable-profiling", action="store_true", help="Enable detailed profiling of training pipeline")
    parser.add_argument("--profiling-report", type=str, default="profiling_report.json", help="Path to save profiling report")
    parser.add_argument("--profiling-summary-freq", type=int, default=10000, help="Frequency of profiling summary prints (timesteps)")

    # Curriculum learning options
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--use_bank_curriculum", action="store_true", help="Use precomputed bank-based curriculum instead of online filtering")
    parser.add_argument("--bank_dir", type=str, default=str(ARTIFACTS_ROOT / "puzzle_bank"), help="Directory of the precomputed puzzle bank")
    parser.add_argument("--curriculum-config", type=str, help="Path to custom curriculum configuration JSON file")
    parser.add_argument("--curriculum-initial-level", type=int, default=0, help="Initial curriculum level (0-4)")
    parser.add_argument("--curriculum-verbose", action="store_true", default=True, help="Print curriculum progression messages")
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.95, help="Success rate threshold for curriculum advancement")
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
    parser.add_argument("--log-dir", default=str(ARTIFACTS_ROOT / "runs" / "ppo"))
    parser.add_argument("--save-path", default=str(ARTIFACTS_ROOT / "checkpoints" / "ppo_model"))
    parser.add_argument("--save-freq", type=int, default=5000, help="Timesteps between checkpoints (0=disable)")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Timesteps between evals (0=disable)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--init-mode", choices=["fresh", "resume", "warmstart"], default="fresh",
                        help="Initialisation strategy: 'fresh' creates a new model, 'resume' continues from a full checkpoint, 'warmstart' loads weights but resets optimiser state")
    parser.add_argument("--resume-checkpoint-path", type=str, default=None,
                        help="Checkpoint .zip to load when init_mode=resume")
    parser.add_argument("--resume-vecnormalize-path", type=str, default=None,
                        help="VecNormalize statistics file to load when resuming (optional)")
    parser.add_argument("--resume-target-total-timesteps", type=int, default=None,
                        help="Target total timestep count when resuming; remaining steps will be computed against the checkpoint")
    parser.add_argument("--resume-additional-timesteps", action="store_true",
                        help="Interpret --timesteps as additional steps when resuming instead of a total target")
    parser.add_argument("--warmstart-params-path", type=str, default=None,
                        help="Checkpoint .zip providing policy parameters for warmstart initialisation")
    parser.add_argument("--warmstart-vecnormalize-path", type=str, default=None,
                        help="VecNormalize statistics file to load alongside warmstart parameters (optional)")
    parser.add_argument("--vecnorm-load-path", type=str, default=None,
                        help="Explicit VecNormalize statistics file to load (overrides default discovery)")
    parser.add_argument("--load-path", type=str, default=None,
                        help="[Deprecated] Equivalent to setting init_mode=resume and resume_checkpoint_path to this value")
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
    parser.add_argument("--traj-record-freq", type=int, default=2000)
    parser.add_argument("--traj-record-episodes", type=int, default=10)
    # Model options
    parser.add_argument("--small-cnn", action="store_true", help="Use custom SmallCNN feature extractor for image obs")
    parser.add_argument("--convlstm", action="store_true", help="Use sb3-contrib CnnLstmPolicy (baseline recurrent)")
    parser.add_argument("--drc", action="store_true", help="Use sb3-contrib RecurrentPPO with recurrent policy (DRC settings)")
    parser.add_argument("--resnet", action="store_true", help="Use ResNet baseline (feed-forward) as a comparison to DRC")
    parser.add_argument("--features-dim", type=int, default=128, help="DRC projection/features dimension before heads")
    parser.add_argument("--conv-channels", type=int, default=32, help="Encoder conv output channels (E_t channels)")
    parser.add_argument("--lstm-channels", type=int, default=64, help="ConvLSTM hidden channels (C)")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of ConvLSTM layers (L)")
    parser.add_argument("--lstm-repeats", type=int, default=1, help="Repeats (R) per env step inside DRC")

    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in config, got {type(data).__name__}")
    return data


def _normalize_key(name: str) -> str:
    # Normalize keys from YAML (allow hyphenated keys)
    return str(name).replace("-", "_")


def flatten_structured_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Support sectioned YAML by flattening known sections to argparse keys.

    Supported sections:
    - env: maps to env-related CLI flags
    - algo: maps to PPO/RecurrentPPO hyperparameters (and algo name)
    - model: {type: resnet|convlstm|drc|small_cnn, params: {...}}
    - curriculum: curriculum toggles and thresholds (incl. bank_dir)
    - eval: evaluation cadence and options
    - monitoring: logging backends, paths, save freq
    - training/paths: optional aliases for common fields
    Unrecognized sections/keys are left as-is (top-level flat config remains valid).
    """
    flat: Dict[str, Any] = {}

    # Pass through existing top-level simple keys first (will be overridden by sections if present)
    for k, v in data.items():
        if isinstance(v, dict) and k in ("env", "algo", "model", "curriculum", "eval", "monitoring", "training", "paths"):
            continue
        if k == "initialization":
            # handled separately by caller
            continue
        flat[_normalize_key(k)] = v

    # env section
    env = data.get("env") or {}
    if isinstance(env, dict):
        mapping = {
            "mode": "env_mode",
            "height": "height",
            "width": "width",
            "num_robots": "num_robots",
            "include_noop": "include_noop",
            "step_penalty": "step_penalty",
            "goal_reward": "goal_reward",
            "noop_penalty": "noop_penalty",
            "max_steps": "max_steps",
            "seed": "seed",
            "ensure_solvable": "ensure_solvable",
            "solver_max_depth": "solver_max_depth",
            "solver_max_nodes": "solver_max_nodes",
            "obs_mode": "obs_mode",
            "cell_obs_pixel_size": "cell_obs_pixel_size",
        }
        for k, v in env.items():
            dst = mapping.get(_normalize_key(k))
            if dst:
                flat[dst] = v

    # algo section
    algo = data.get("algo") or {}
    if isinstance(algo, dict):
        mapping = {
            "algo": "algo",
            "name": "algo",
            "timesteps": "timesteps",
            "n_envs": "n_envs",
            "n_steps": "n_steps",
            "batch_size": "batch_size",
            "n_epochs": "n_epochs",
            "lr": "lr",
            "lr_end": "lr_end",
            "lr_schedule": "lr_schedule",
            "gamma": "gamma",
            "clip_range": "clip_range",
            "clip_end": "clip_end",
            "clip_schedule": "clip_schedule",
            "gae_lambda": "gae_lambda",
            "vf_coef": "vf_coef",
            "max_grad_norm": "max_grad_norm",
            "ent_coef": "ent_coef",
            "normalize_advantage": "normalize_advantage",
            "device": "device",
        }
        for k, v in algo.items():
            dst = mapping.get(_normalize_key(k))
            if dst:
                flat[dst] = v

    # model section
    model = data.get("model") or {}
    if isinstance(model, dict):
        model_type = model.get("type")
        if isinstance(model_type, str):
            mt = model_type.strip().lower()
            if mt in ("resnet", "convlstm", "drc", "small_cnn"):
                flat[mt] = True
        params = model.get("params") or {}
        if isinstance(params, dict):
            param_mapping = {
                "features_dim": "features_dim",
                "conv_channels": "conv_channels",
                "lstm_channels": "lstm_channels",
                "lstm_layers": "lstm_layers",
                "lstm_repeats": "lstm_repeats",
            }
            for k, v in params.items():
                nk = _normalize_key(k)
                dst = param_mapping.get(nk)
                if dst:
                    flat[dst] = v

    # curriculum section
    cur = data.get("curriculum") or {}
    if isinstance(cur, dict):
        mapping = {
            "enabled": "curriculum",
            "use_bank": "use_bank_curriculum",
            "bank_dir": "bank_dir",
            "config": "curriculum_config",
            "initial_level": "curriculum_initial_level",
            "success_threshold": "curriculum_success_threshold",
            "min_episodes": "curriculum_min_episodes",
            "window_size": "curriculum_window_size",
            "check_freq": "curriculum_check_freq",
            "verbose": "curriculum_verbose",
        }
        for k, v in cur.items():
            dst = mapping.get(_normalize_key(k))
            if dst:
                flat[dst] = v

    # eval section
    ev = data.get("eval") or {}
    if isinstance(ev, dict):
        mapping = {
            "eval_freq": "eval_freq",
            "eval_episodes": "eval_episodes",
            "fixedset": "eval_fixedset",
            "solver_cache": "solver_cache",
            "traj_record_freq": "traj_record_freq",
            "traj_record_episodes": "traj_record_episodes",
        }
        for k, v in ev.items():
            dst = mapping.get(_normalize_key(k))
            if dst:
                flat[dst] = v

    # monitoring section
    mon = data.get("monitoring") or {}
    if isinstance(mon, dict):
        mapping = {
            "log_dir": "log_dir",
            "save_path": "save_path",
            "save_freq": "save_freq",
            "tensorboard": "monitor_tensorboard",
            "wandb": "monitor_wandb",
            "wandb_project": "wandb_project",
            "wandb_entity": "wandb_entity",
            "wandb_run_name": "wandb_run_name",
            "wandb_tags": "wandb_tags",
        }
        for k, v in mon.items():
            dst = mapping.get(_normalize_key(k))
            if dst:
                flat[dst] = v

    # optional training/paths sections as aliases
    for sec_name in ("training", "paths"):
        sec = data.get(sec_name) or {}
        if isinstance(sec, dict):
            for k, v in sec.items():
                nk = _normalize_key(k)
                if nk in ("log_dir", "save_path", "save_freq", "profile"):
                    flat[nk] = v

    return flat
def apply_profile_presets(args_dict: Dict[str, Any], profile: str, defaults: Dict[str, Any]) -> None:
    if profile == "research_plan":
        if args_dict.get("timesteps") == defaults.get("timesteps"):
            args_dict["timesteps"] = 5_000_000
        args_dict["clip_range"] = 0.2
        args_dict["clip_end"] = 0.05
        args_dict["clip_schedule"] = "linear"
        args_dict["gae_lambda"] = 0.95
        args_dict["vf_coef"] = 0.5
        args_dict["max_grad_norm"] = 0.5
        args_dict["lr"] = 3e-4
        args_dict["lr_end"] = 3e-5
        args_dict["lr_schedule"] = "linear"
        args_dict["normalize_advantage"] = True
        if not args_dict.get("vecnorm", False):
            args_dict["vecnorm"] = False
            args_dict["vecnorm_norm_obs"] = True
            args_dict["vecnorm_norm_reward"] = False
    elif profile == "quick_debug":
        args_dict["timesteps"] = 50_000
        args_dict["lr"] = 1e-3
        args_dict["lr_end"] = 1e-3
        args_dict["lr_schedule"] = "none"
        args_dict["clip_range"] = 0.2
        args_dict["clip_end"] = 0.2
        args_dict["clip_schedule"] = "none"
        args_dict["vf_coef"] = 0.5
        args_dict["max_grad_norm"] = 0.5
        args_dict["normalize_advantage"] = True
        args_dict["vecnorm"] = False
    elif profile == "baseline_sb3":
        args_dict["lr"] = 3e-4
        args_dict["lr_schedule"] = "none"
        args_dict["clip_range"] = 0.2
        args_dict["clip_schedule"] = "none"
        args_dict["gae_lambda"] = 0.95
        args_dict["vf_coef"] = 0.5
        args_dict["max_grad_norm"] = 0.5
        args_dict["normalize_advantage"] = True
        args_dict["vecnorm"] = False


def apply_initialization_overrides(args: argparse.Namespace, init_block: Any) -> None:
    """Apply structured initialization configuration overrides onto parsed args."""
    if init_block is None:
        return
    if not isinstance(init_block, dict):
        raise ValueError("'initialization' config must be a mapping")

    valid_modes = {"fresh", "resume", "warmstart"}

    def _clean_optional_path(value: Any, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string path or null")
        stripped = value.strip()
        return stripped or None

    mode_value = init_block.get("mode", args.init_mode)
    if mode_value not in valid_modes:
        raise ValueError(f"initialization.mode must be one of {sorted(valid_modes)}")
    args.init_mode = mode_value

    generalized_vecnorm = init_block.get("vecnormalize_path")
    if generalized_vecnorm is not None:
        args.vecnorm_load_path = _clean_optional_path(generalized_vecnorm, "initialization.vecnormalize_path")

    resume_cfg = init_block.get("resume")
    if resume_cfg is not None:
        if not isinstance(resume_cfg, dict):
            raise ValueError("initialization.resume must be a mapping")
        if "checkpoint_path" in resume_cfg:
            args.resume_checkpoint_path = _clean_optional_path(resume_cfg["checkpoint_path"], "initialization.resume.checkpoint_path")
        if "vecnormalize_path" in resume_cfg:
            args.resume_vecnormalize_path = _clean_optional_path(resume_cfg["vecnormalize_path"], "initialization.resume.vecnormalize_path")
        if "target_total_timesteps" in resume_cfg:
            value = resume_cfg["target_total_timesteps"]
            if value is None:
                args.resume_target_total_timesteps = None
            elif isinstance(value, numbers.Integral):
                if value < 0:
                    raise ValueError("initialization.resume.target_total_timesteps must be non-negative")
                args.resume_target_total_timesteps = int(value)
            else:
                raise ValueError("initialization.resume.target_total_timesteps must be an integer or null")
        if "additional_timesteps" in resume_cfg:
            args.resume_additional_timesteps = bool(resume_cfg["additional_timesteps"])

    warmstart_cfg = init_block.get("warmstart")
    if warmstart_cfg is not None:
        if not isinstance(warmstart_cfg, dict):
            raise ValueError("initialization.warmstart must be a mapping")
        if "params_path" in warmstart_cfg:
            args.warmstart_params_path = _clean_optional_path(warmstart_cfg["params_path"], "initialization.warmstart.params_path")
        if "vecnormalize_path" in warmstart_cfg:
            args.warmstart_vecnormalize_path = _clean_optional_path(warmstart_cfg["vecnormalize_path"], "initialization.warmstart.vecnormalize_path")

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
                cell_obs_pixel_size=getattr(args, "cell_obs_pixel_size", 128),
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
                cell_obs_pixel_size=getattr(args, "cell_obs_pixel_size", 128),
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
            cell_obs_pixel_size=getattr(args, "cell_obs_pixel_size", 128),
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


def make_bank_curriculum_env_factory(args: argparse.Namespace) -> tuple[Callable[[], RicochetRobotsEnv], object]:
    """Return a thunk to create a bank-based curriculum environment instance based on CLI args."""
    # Initialize puzzle bank
    bank = PuzzleBank(args.bank_dir)
    
    # Load curriculum levels from config file or use defaults
    curriculum_levels = None
    if args.curriculum_config is not None:
        # Load custom curriculum config from file
        with open(args.curriculum_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        levels_data = config_data.get("levels", None)
        if levels_data:
            # Convert spec_key dictionaries to SpecKey objects
            from src.env.puzzle_bank import SpecKey
            for level in levels_data:
                if isinstance(level.get("spec_key"), dict):
                    spec_dict = level["spec_key"]
                    level["spec_key"] = SpecKey(
                        height=spec_dict["height"],
                        width=spec_dict["width"],
                        num_robots=spec_dict["num_robots"],
                        edge_t_per_quadrant=spec_dict["edge_t_per_quadrant"],
                        central_l_per_quadrant=spec_dict["central_l_per_quadrant"],
                        generator_rules_version=spec_dict["generator_rules_version"]
                    )
            curriculum_levels = levels_data
        print(f"Loaded custom curriculum config from {args.curriculum_config} with {len(curriculum_levels)} levels")
    else:
        print("Using default curriculum levels")
    
    # Create bank curriculum manager with CLI thresholds
    print(f"Initial level: {args.curriculum_initial_level}")
    print(f"Curriculum levels: {curriculum_levels}")
    if args.curriculum_initial_level is not None:
        curriculum_levels = curriculum_levels[args.curriculum_initial_level:]
    print(f"Curriculum levels: {curriculum_levels}")
    bank_manager = create_bank_curriculum_manager(
        bank=bank,
        curriculum_levels=curriculum_levels,  # use custom or defaults
        success_rate_threshold=args.curriculum_success_threshold,
        min_episodes_per_level=args.curriculum_min_episodes,
        success_rate_window_size=args.curriculum_window_size,
        advancement_check_frequency=args.curriculum_check_freq,
        verbose=args.curriculum_verbose,
    )
    
    def _fn():
        return BankCurriculumWrapper(
            bank=bank,
            curriculum_manager=bank_manager,
            obs_mode=(args.obs_mode if args.obs_mode in ("image", "rgb_image", "rgb_cell_image", "symbolic") else "rgb_image"),
            channels_first=True,
            render_mode=None,
            cell_obs_pixel_size=getattr(args, "cell_obs_pixel_size", 128),
            verbose=args.curriculum_verbose,
        )
    
    return _fn, bank_manager


def main() -> None:
    cli_parser = argparse.ArgumentParser(description="Train PPO on Ricochet Robots")
    cli_parser.add_argument("config_path", help="Path to YAML configuration file")
    cli_args = cli_parser.parse_args()

    config_path = cli_args.config_path
    raw_config = load_config_file(config_path)
    init_config_block = raw_config.get("initialization")
    if init_config_block is not None and not isinstance(init_config_block, dict):
        raise ValueError("'initialization' config must be a mapping")
    # Support structured configs: flatten known sections into argparse keys
    config_data = flatten_structured_config(raw_config)
    config_data.pop("initialization", None)

    training_parser = build_training_parser()
    defaults_namespace = training_parser.parse_args([])
    defaults = vars(defaults_namespace).copy()

    valid_keys = {
        action.dest
        for action in training_parser._actions
        if action.dest not in (argparse.SUPPRESS, None)
    }
    valid_keys.discard("help")

    unknown_keys = set(config_data) - valid_keys
    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown_keys))}")

    profile_value = config_data.get("profile", defaults.get("profile", "none"))
    args_dict: Dict[str, Any] = defaults.copy()
    args_dict["profile"] = profile_value

    apply_profile_presets(args_dict, profile_value, defaults)

    for key, value in config_data.items():
        if key == "profile":
            continue
        args_dict[key] = value

    training_parser.set_defaults(**args_dict)
    args = training_parser.parse_args([])
    args.config_path = config_path
    print(f"Loaded configuration from {config_path}")

    # Apply structured initialization overrides and handle deprecated --load-path usage
    apply_initialization_overrides(args, init_config_block)

    if args.load_path:
        coerced_path = args.load_path.strip()
        if coerced_path:
            if args.init_mode == "fresh" and args.resume_checkpoint_path is None and args.warmstart_params_path is None:
                args.init_mode = "resume"
                args.resume_checkpoint_path = coerced_path
                print("Note: --load-path is deprecated. Using init_mode=resume with the provided checkpoint.")
            elif args.init_mode != "resume":
                print("Warning: --load-path provided but ignored because init_mode is not 'resume'.")
        args.load_path = None

    # Ensure optional fields exist on args Namespace even if not provided via CLI/defaults
    for attr in ("resume_checkpoint_path", "resume_vecnormalize_path", "resume_target_total_timesteps",
                 "warmstart_params_path", "warmstart_vecnormalize_path", "vecnorm_load_path"):
        if not hasattr(args, attr):
            setattr(args, attr, None)

    # Initialize profiling if enabled
    if args.enable_profiling:
        try:
            from src.profiling import get_profiler, print_profiling_summary, save_profiling_report
            profiler = get_profiler()
            profiler.enable()
            print("Profiling enabled - detailed performance tracking active")
        except ImportError:
            print("Warning: Profiling requested but profiling module not available")
            args.enable_profiling = False

    if args.init_mode == "resume" and not args.resume_checkpoint_path:
        raise ValueError("init_mode='resume' requires resume_checkpoint_path to be set")
    if args.init_mode == "warmstart" and not args.warmstart_params_path:
        raise ValueError("init_mode='warmstart' requires warmstart_params_path to be set")
    if args.init_mode == "resume" and args.resume_additional_timesteps and args.resume_target_total_timesteps is not None:
        print("Note: resume_additional_timesteps is True; resume_target_total_timesteps will be ignored.")

    # Enforce mutual exclusivity for recurrent modes
    if sum([int(args.convlstm), int(args.drc), int(args.resnet)]) > 1:
        raise ValueError("Flags --convlstm, --drc, and --resnet are mutually exclusive. Choose one.")

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
    # Try to import sb3-contrib RecurrentPPO
    try:
        sb3c = importlib.import_module("sb3_contrib")
        RecurrentPPO = sb3c.RecurrentPPO  # type: ignore[attr-defined]
    except Exception:
        RecurrentPPO = None  # type: ignore
    vec_env_mod = importlib.import_module("stable_baselines3.common.env_util")
    make_vec_env = vec_env_mod.make_vec_env

    # Timestamp the run and prefix WandB run name and save/checkpoint files
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prefix wandb run name
    try:
        existing_name = getattr(args, "wandb_run_name", None)
        args.wandb_run_name = f"{run_timestamp}_{existing_name}" if existing_name else run_timestamp
    except Exception:
        # Best-effort; keep default WandB naming if something goes wrong
        pass
    # Prefix model save_path basename for final save and checkpoint prefixes
    try:
        if getattr(args, "save_path", None):
            _sp_dir = os.path.dirname(args.save_path) or "."
            _sp_base = os.path.basename(args.save_path)
            args.save_path = os.path.join(_sp_dir, f"{run_timestamp}_{_sp_base}")
    except Exception:
        pass

    os.makedirs(args.log_dir, exist_ok=True)
    _save_dir = os.path.dirname(args.save_path)
    if _save_dir == "":
        _save_dir = "."
    os.makedirs(_save_dir, exist_ok=True)

    # Choose environment factory based on curriculum setting
    curriculum_manager = None
    if args.curriculum:
        if args.use_bank_curriculum:
            env_factory, curriculum_manager = make_bank_curriculum_env_factory(args)
            print(f"Bank curriculum enabled (bank_dir={args.bank_dir}) - no solver during training")
            # Force rgb_image if varying board sizes to keep obs shape fixed
            if args.obs_mode != "rgb_image":
                print("Note: For bank curriculum with varying sizes, obs_mode is recommended as 'rgb_image'.")
        else:
            env_factory, curriculum_manager = make_curriculum_env_factory(args)
            print("Online curriculum enabled - will progressively increase difficulty with solver gating")
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
    vecnorm_loaded_from: str | None = None
    if args.vecnorm and _VecNormalize is not None:
        candidate_paths: list[tuple[str, str]] = []
        seen_paths: set[str] = set()

        def _push_path(label: str, path: str | None) -> None:
            if path is None:
                return
            normalized = path.strip()
            if not normalized or normalized in seen_paths:
                return
            seen_paths.add(normalized)
            candidate_paths.append((label, normalized))

        if args.init_mode == "resume":
            _push_path("resume", args.resume_vecnormalize_path)
        elif args.init_mode == "warmstart":
            _push_path("warmstart", args.warmstart_vecnormalize_path)
        _push_path("override", args.vecnorm_load_path)
        default_vecnorm_path = os.path.join(args.log_dir, "vecnormalize.pkl")
        _push_path("default", default_vecnorm_path)

        mandatory_path = args.resume_vecnormalize_path if args.init_mode == "resume" else None
        if mandatory_path is not None:
            mandatory_path = mandatory_path.strip()
            if mandatory_path and not os.path.exists(mandatory_path):
                raise FileNotFoundError(f"VecNormalize stats required for resume were not found at {mandatory_path}")

        last_error: Exception | None = None
        for label, candidate in candidate_paths:
            if not os.path.exists(candidate):
                continue
            try:
                vec_env = _VecNormalize.load(candidate, venv=vec_env)
                vec_env.training = True  # type: ignore[attr-defined]
                vecnorm_used = True
                vecnorm_loaded_from = candidate
                print(f"Loaded VecNormalize statistics from {candidate} ({label})")
                break
            except Exception as exc:  # pragma: no cover - defensive logging only
                last_error = exc
                continue

        if not vecnorm_used:
            if mandatory_path is not None:
                raise RuntimeError(
                    "VecNormalize statistics could not be loaded from the required resume path."
                    + (f" Last error: {last_error}" if last_error is not None else "")
                )
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
    if args.obs_mode in ["image", "rgb_image", "rgb_cell_image"]:
        if args.drc:
            if RecurrentPPO is None:
                raise ImportError("sb3-contrib is required for recurrent PPO. Please install sb3-contrib.")
            if DRCRecurrentPolicy is None:
                raise ImportError("DRC policy class not importable. Ensure models/drc_policy.py is available.")
            policy = DRCRecurrentPolicy  # type: ignore[assignment]
            policy_kwargs = dict(
                features_extractor_kwargs=dict(
                    # features_dim=args.features_dim,
                    # conv_channels=args.conv_channels,
                    # lstm_channels=args.lstm_channels,
                    # num_lstm_layers=args.lstm_layers,
                    # num_repeats=args.lstm_repeats,
                    # use_pool_and_inject=True,
                ),
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                normalize_images=(args.obs_mode in ("rgb_image", "rgb_cell_image")),
            )
        elif args.convlstm:
            # Until a compliant recurrent policy wired to ConvLSTM features is implemented,
            # train with sb3-contrib RecurrentPPO using its supported LSTM policies.
            # This avoids incorrect training of recurrent models with vanilla PPO.
            if RecurrentPPO is None:
                raise ImportError("sb3-contrib is required for recurrent PPO. Please install sb3-contrib.")
            policy = "CnnLstmPolicy"
            policy_kwargs = dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                normalize_images=(args.obs_mode in ("rgb_image", "rgb_cell_image"))
            )
        elif not tiny_grid:
            # Fallback to CNN for image observations
            if args.resnet:
                # Use ResNet baseline features
                try:
                    from src.models.resnet import ResNetFeaturesExtractor  # type: ignore
                except ImportError as _e:
                    raise ImportError(f"ResNet baseline requested but not available: {_e}")
                policy = ResNetPolicy
                policy_kwargs = dict(
                    # net_arch=dict(pi=[128, 128], vf=[128, 128]),
                    normalize_images=(args.obs_mode in ("rgb_image", "rgb_cell_image")),
                    # features_extractor_class=ResNetFeaturesExtractor,
                    # features_extractor_kwargs=dict(features_dim=args.features_dim),

                )
            else:
                policy = "CnnPolicy"
                policy_kwargs = dict(
                    net_arch=dict(pi=[128, 128], vf=[128, 128]), 
                    normalize_images=(args.obs_mode in ("rgb_image", "rgb_cell_image"))  # Normalize RGB images
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

    # Initialize or load algorithm
    model = None  # type: ignore[assignment]
    resume_mode = args.init_mode == "resume"
    warmstart_mode = args.init_mode == "warmstart"
    print(f"args.init_mode: {args.init_mode}")

    if resume_mode:
        checkpoint_path = args.resume_checkpoint_path or ""
        try:
            if (args.convlstm or args.drc) and RecurrentPPO is not None:
                model = RecurrentPPO.load(checkpoint_path, env=vec_env, device=args.device)  # type: ignore[assignment]
            else:
                model = PPO.load(checkpoint_path, env=vec_env, device=args.device)  # type: ignore[assignment]
            print(f"Loaded model from {checkpoint_path} and attached env for resuming")
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {exc}")
    else:
        if args.convlstm or args.drc:
            # Use sb3-contrib RecurrentPPO for recurrent training
            model = RecurrentPPO(  # type: ignore[operator]
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
        else:
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

        if warmstart_mode:
            params_path = args.warmstart_params_path or ""
            try:
                model.set_parameters(params_path, exact_match=True, device=args.device)
            except Exception as exc:
                try:
                    model.set_parameters(params_path, exact_match=False, device=args.device)
                except Exception as exc2:
                    raise RuntimeError(f"Failed to warmstart from {params_path}: {exc2}") from exc
            print(f"Loaded policy parameters from {params_path} for warmstart initialisation")

    if model is None:
        raise RuntimeError("Model initialisation failed")

    reset_num_timesteps = True
    learn_timesteps = max(0, int(args.timesteps))
    if resume_mode:
        reset_num_timesteps = False
        current_timesteps = int(getattr(model, "num_timesteps", 0))
        if args.resume_additional_timesteps:
            learn_timesteps = max(0, int(args.timesteps))
            print(f"Resuming for an additional {learn_timesteps} timesteps (current: {current_timesteps})")
        else:
            target_total = args.resume_target_total_timesteps
            if target_total is None:
                target_total = int(args.timesteps)
            if target_total < 0:
                raise ValueError("resume_target_total_timesteps must be non-negative")
            remaining = max(0, int(target_total) - current_timesteps)
            learn_timesteps = remaining
            print(
                f"Resuming towards target total {target_total} timesteps. "
                f"Already trained: {current_timesteps}. Remaining: {learn_timesteps}."
            )
    elif warmstart_mode:
        # Warmstart keeps optimiser fresh but uses pre-trained weights
        print(f"Warmstart initialisation complete; training for {learn_timesteps} timesteps from scratch schedule")

    # Callbacks: periodic checkpoint and eval success-rate logging + monitoring hub
    callbacks = []
    # Monitoring hub setup
    try:
        from src.monitoring import MonitoringHub, TensorBoardBackend, WandbBackend
        from src.monitoring.collectors import EpisodeStatsCollector, ActionUsageCollector, CurriculumProgressCollector
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
        if not monitoring_backends:
            print("MonitoringHub disabled: no logging backends enabled (set monitoring.tensorboard or monitoring.wandb).")
            hub = None  # type: ignore
        else:
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

        if hub is not None and hasattr(model, "logger"):
            # Version-agnostic writer: accepts both (kvs, step) and (kvs, key_excluded, step)
            class _HubSB3Writer:
                def __init__(self, hub_ref):
                    self._hub = hub_ref

                def write(self, *args, **kwargs) -> None:
                    # Extract kvs and step across SB3 versions
                    key_values = args[0] if len(args) >= 1 else kwargs.get("key_values")
                    step = kwargs.get("step", 0)
                    if len(args) >= 2 and isinstance(args[1], numbers.Number) and "step" not in kwargs:
                        step = args[1]
                    elif len(args) >= 3 and isinstance(args[2], numbers.Number) and "step" not in kwargs:
                        step = args[2]
                    step_value = None
                    if isinstance(step, numbers.Number):
                        try:
                            step_float = float(step)
                            if math.isfinite(step_float):
                                step_value = int(step_float)
                        except (TypeError, ValueError, OverflowError):
                            step_value = None
                    if not isinstance(key_values, dict):
                        return None
                    for key, value in key_values.items():
                        if not isinstance(key, str):
                            continue
                        if not (key.startswith("train/") or key.startswith("rollout/") or key.startswith("time/") or key.startswith("eval/")):
                            continue
                        try:
                            scalar = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(scalar):
                            continue
                        try:
                            self._hub.log_scalar(key, scalar, step_value)
                        except Exception:
                            pass

                def write_sequence(self, *args, **kwargs) -> None:
                    return None

                def close(self) -> None:
                    return None

            try:
                model.logger.output_formats.append(_HubSB3Writer(hub))
            except Exception:
                pass

        # Custom callback for curriculum learning
        class CurriculumCallback(BaseCallback):
            def __init__(self, curriculum_manager, verbose=0):
                super().__init__(verbose)
                self.curriculum_manager = curriculum_manager
                self.last_logged_level = -1
                self.last_logged_stats = 0
                
            def _on_step(self) -> bool:
                # Log curriculum stats if available
                if self.curriculum_manager is not None:
                    stats_fn = getattr(self.curriculum_manager, "get_stats", None)
                    if stats_fn is None:
                        stats_fn = getattr(self.curriculum_manager, "get_curriculum_stats", None)
                    if stats_fn is None:
                        return True
                    stats = stats_fn()
                    
                    # Log to console every 1000 steps
                    if self.verbose > 0 and self.num_timesteps % 1000 == 0:
                        print(f"Curriculum Level {stats['current_level']}: {stats['level_name']} "
                              f"(Success: {stats['success_rate']:.2f}, Episodes: {stats['episodes_at_level']})")
                    
                    # Log to TensorBoard every 100 steps for more frequent updates
                    if hasattr(self, 'logger') and self.logger is not None and self.num_timesteps - self.last_logged_stats >= 100:
                        self.logger.record("curriculum/level", stats['current_level'])
                        self.logger.record("curriculum/success_rate", stats['success_rate'])
                        self.logger.record("curriculum/episodes_at_level", stats['episodes_at_level'])
                        self.logger.record("curriculum/total_episodes", stats['total_episodes'])
                        self.logger.record("curriculum/window_size", stats['window_size'])
                        self.last_logged_stats = self.num_timesteps
                        
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
                        self._last_logged_stats = 0
                    def _on_step(self) -> bool:
                        lvl = self.manager.get_current_level()
                        stats_fn = getattr(self.manager, "get_stats", None)
                        if stats_fn is None:
                            stats_fn = getattr(self.manager, "get_curriculum_stats", None)
                        if stats_fn is None:
                            return True
                        stats = stats_fn()
                        
                        # Log curriculum stats every 100 steps for more frequent updates
                        if self.num_timesteps - self._last_logged_stats >= 100:
                            hub.log_scalar("curriculum/level", float(stats.get("current_level", 0)), self.num_timesteps)
                            hub.log_scalar("curriculum/success_rate", float(stats.get("success_rate", 0)), self.num_timesteps)
                            hub.log_scalar("curriculum/episodes_at_level", float(stats.get("episodes_at_level", 0)), self.num_timesteps)
                            hub.log_scalar("curriculum/total_episodes", float(stats.get("total_episodes", 0)), self.num_timesteps)
                            self._last_logged_stats = self.num_timesteps
                        
                        # Log level changes immediately
                        if self._last_level != lvl:
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
        
        # Remove ad-hoc recurrent callback; sb3-contrib handles sequence masks internally

        # Forward per-episode optimality gap metrics from src.env infos to monitoring hub
        if hub is not None:
            class _HubEpisodeMetrics(BaseCallback):
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self._rollout_successes: list[int] = []
                    self._rollout_lengths: list[int] = []
                    self._rollout_returns: list[float] = []
                    self._episode_returns: dict[int, float] = {}
                    self._episode_lengths: dict[int, int] = {}
                    self._note_emitted = False
                def _on_step(self) -> bool:
                    try:
                        infos = self.locals.get('infos', [])  # type: ignore[assignment]
                    except Exception:
                        infos = []
                    rewards = self.locals.get('rewards', [])  # type: ignore[assignment]
                    dones = self.locals.get('dones', [])  # type: ignore[assignment]
                    if hasattr(rewards, 'tolist'):
                        rewards_seq = rewards.tolist()
                    else:
                        rewards_seq = list(rewards) if isinstance(rewards, (list, tuple)) else []
                    if hasattr(dones, 'tolist'):
                        dones_seq = dones.tolist()
                    else:
                        dones_seq = list(dones) if isinstance(dones, (list, tuple)) else []

                    # Emit a one-time note explaining success metrics to the monitoring streams.
                    if not self._note_emitted:
                        try:
                            hub.log_text(
                                "train/success_rate_note",
                                "train/success_rate (windowed) reflects the MonitoringHub rolling window; "
                                "SB3's rollout/success_rate remains the stochastic training view, and eval/success_rate "
                                "comes from deterministic eval episodes.",
                                self.num_timesteps,
                            )
                        except Exception:
                            pass
                        self._note_emitted = True
                    # SB3 provides a list of info dicts for each parallel env
                    for idx, info in enumerate(infos or []):
                        if len(rewards_seq) > idx:
                            self._episode_returns[idx] = self._episode_returns.get(idx, 0.0) + float(rewards_seq[idx])
                        self._episode_lengths[idx] = self._episode_lengths.get(idx, 0) + 1
                        if not isinstance(info, dict):
                            continue
                        # Only log on episode end when env attaches the metric
                        if 'optimality_gap' in info and info.get('optimality_gap') is not None:
                            try:
                                hub.log_scalar('train/optimality_gap', float(info['optimality_gap']), self.num_timesteps)
                            except Exception:
                                pass
                        if 'episode_steps' in info and info.get('episode_steps') is not None:
                            try:
                                hub.log_scalar('train/episode_steps', float(info['episode_steps']), self.num_timesteps)
                            except Exception:
                                pass
                        if 'optimal_length' in info and info.get('optimal_length') is not None:
                            try:
                                hub.log_scalar('train/optimal_length', float(info['optimal_length']), self.num_timesteps)
                            except Exception:
                                pass
                        episode_info = info.get('episode') if isinstance(info, dict) else None
                        done_flag = bool(len(dones_seq) > idx and dones_seq[idx])
                        if episode_info is not None or done_flag:
                            length = int((episode_info or {}).get('l', self._episode_lengths.get(idx, 0)))
                            ret = float((episode_info or {}).get('r', self._episode_returns.get(idx, 0.0)))
                            success = 1 if bool(info.get('is_success', False)) else 0
                            self._rollout_successes.append(success)
                            self._rollout_lengths.append(length)
                            self._rollout_returns.append(ret)
                            self._episode_returns[idx] = 0.0
                            self._episode_lengths[idx] = 0
                    return True
                def _on_rollout_end(self) -> None:
                    if self._rollout_successes or self._rollout_lengths or self._rollout_returns:
                        try:
                            hub.on_rollout_end({
                                'global_step': self.num_timesteps,
                                'rollout_successes': self._rollout_successes,
                                'rollout_lengths': self._rollout_lengths,
                                'rollout_returns': self._rollout_returns,
                            })
                        except Exception:
                            pass
                    self._rollout_successes.clear()
                    self._rollout_lengths.clear()
                    self._rollout_returns.clear()
                    self._episode_returns.clear()
                    self._episode_lengths.clear()
            callbacks.append(_HubEpisodeMetrics(verbose=0))
        
        # Always-on hub callback for periodic trajectory logging (independent of eval_fixedset)
        if hub is not None and args.traj_record_freq > 0:
            from src.monitoring.evaluators import TrajectoryRecorder
            class _HubTrajectoryLogger(BaseCallback):
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self._last_traj_dump = 0
                    self._renderer = None
                    try:
                        from src.env.visuals.mpl_renderer import MatplotlibBoardRenderer
                        self._renderer = MatplotlibBoardRenderer(cell_size=28)
                    except Exception:
                        self._renderer = None
                def _on_step(self) -> bool:
                    # Throttle logging by traj_record_freq
                    if (self.num_timesteps - self._last_traj_dump) >= args.traj_record_freq:
                        # Build a renderable env factory; prefer rgb frames for video logging
                        def _make_renderable_env():
                            e = make_env_factory(args)()
                            try:
                                if hasattr(e, 'render_mode'):
                                    e.render_mode = 'rgb'
                            except Exception:
                                pass
                            return e
                        try:
                            rec = TrajectoryRecorder(episodes=args.traj_record_episodes, capture_render=True, renderer=self._renderer)
                            trajectories = rec.record(_make_renderable_env, self.model)
                            # Compact text summary per episode
                            summaries = []
                            for idx, traj in enumerate(trajectories):
                                acts = [str(step.get('action', -1)) for step in traj.get('steps', [])]
                                succ = "✓" if bool(traj.get('success', False)) else "✗"
                                summaries.append(f"ep {idx}: success={succ}, steps={len(acts)}, actions=[" + ",".join(acts[:60]) + (",..." if len(acts) > 60 else "]"))
                            if summaries:
                                hub.log_text("eval/trajectories/text", "\n".join(summaries), self.num_timesteps)
                            # If frames are present, log a short video for the first episode
                            first_frames: list[Any] = []
                            if trajectories and trajectories[0].get('frames'):
                                first_frames = trajectories[0]['frames'] or []
                                try:
                                    import numpy as _np
                                    if first_frames and isinstance(first_frames[0], (_np.ndarray, list)):
                                        video = _np.asarray(first_frames, dtype=_np.uint8)
                                        if video.ndim == 3:
                                            # Single frame -> add time axis
                                            video = video[None, ...]
                                        if video.shape[0] == 1:
                                            # Duplicate lone frame so wandb can encode a short clip
                                            video = _np.repeat(video, 2, axis=0)
                                        if video.ndim == 3:
                                            video = video[:, :, :, None]
                                        elif video.ndim == 4 and video.shape[-1] not in (1, 3, 4):
                                            video = video[..., None]
                                        hub.log_video("eval/trajectories/video0", video, self.num_timesteps, fps=4)
                                    elif first_frames:
                                        hub.log_image("eval/trajectories/frame0", first_frames[0], self.num_timesteps)
                                    else:
                                        print(f"Warning: No frames captured for trajectory video at step {self.num_timesteps}")
                                except Exception as exc:
                                    print(f"Warning: Trajectory video logging failed at step {self.num_timesteps}: {exc}")
                                    try:
                                        if first_frames:
                                            hub.log_image("eval/trajectories/frame0", first_frames[0], self.num_timesteps)
                                    except Exception as img_exc:
                                        print(f"Warning: Trajectory image fallback failed at step {self.num_timesteps}: {img_exc}")
                            # Also log a basic step counter to ensure the run shows activity
                            hub.log_scalar("train/num_timesteps", float(self.num_timesteps), self.num_timesteps)
                        except Exception as exc:
                            # Surface issues instead of failing silently
                            print(f"Warning: Trajectory logging failed at step {self.num_timesteps}: {exc}")
                        self._last_traj_dump = self.num_timesteps
                    return True
            callbacks.append(_HubTrajectoryLogger(verbose=0))

        # Wrap separate eval env
        if args.eval_freq > 0:
            # When curriculum is enabled, mirror the training env at the current level
            if args.curriculum and curriculum_manager is not None:
                # For bank curriculum, use the same environment factory
                # (BankCurriculumManager doesn't have a config attribute)
                if hasattr(curriculum_manager, 'config'):
                    # Old online curriculum system
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
                    # Bank curriculum system - use same factory
                    eval_env = _make(env_factory, n_envs=1, seed=args.seed)
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
            eval_cb = EvalCallback(eval_env, best_model_save_path=(os.path.dirname(args.save_path) or "."),
                                   log_path=args.log_dir, eval_freq=args.eval_freq,
                                   n_eval_episodes=args.eval_episodes, deterministic=True, render=False)
            callbacks.append(eval_cb)
            if hub is not None:
                class _HubEvalMetricsForward(BaseCallback):
                    def __init__(self, eval_callback, verbose=0):
                        super().__init__(verbose)
                        self._eval_callback = eval_callback
                        self._last_logged_eval = len(getattr(eval_callback, "evaluations_results", []) or [])

                    def _on_step(self) -> bool:
                        results = getattr(self._eval_callback, "evaluations_results", None)
                        if results is None:
                            return True
                        current = len(results)
                        if current <= self._last_logged_eval:
                            return True
                        try:
                            import numpy as _np
                        except ImportError:
                            self._last_logged_eval = current
                            return True

                        rewards = results[-1]
                        if isinstance(rewards, (list, tuple, _np.ndarray)) and len(rewards) > 0:
                            try:
                                mean_reward = float(_np.mean(rewards))
                                hub.log_scalar("eval/mean_reward", mean_reward, self.num_timesteps)
                            except Exception:
                                pass
                        lengths_seq = getattr(self._eval_callback, "evaluations_length", None)
                        if isinstance(lengths_seq, list) and len(lengths_seq) >= current:
                            lengths = lengths_seq[-1]
                            if isinstance(lengths, (list, tuple, _np.ndarray)) and len(lengths) > 0:
                                try:
                                    mean_len = float(_np.mean(lengths))
                                except Exception:
                                    mean_len = None
                                if mean_len is not None:
                                    try:
                                        hub.log_scalar("eval/mean_ep_length", mean_len, self.num_timesteps)
                                    except Exception:
                                        pass
                        successes_seq = getattr(self._eval_callback, "evaluations_successes", None)
                        if isinstance(successes_seq, list) and len(successes_seq) >= current:
                            successes = successes_seq[-1]
                            if isinstance(successes, (list, tuple, _np.ndarray)) and len(successes) > 0:
                                try:
                                    success_rate = float(_np.mean(successes))
                                    hub.log_scalar("eval/success_rate", success_rate, self.num_timesteps)
                                except Exception:
                                    pass
                        self._last_logged_eval = current
                        return True

                callbacks.append(_HubEvalMetricsForward(eval_cb, verbose=0))
            # Add fixed-set evaluation with solver-aware optimality gap via monitoring hub if configured
            if hub is not None and args.eval_fixedset is not None:
                try:
                    import os as _os
                    from src.monitoring.evaluators import FixedSetEvaluator, OptimalityGapEvaluator, SolverCache, TrajectoryRecorder
                    # Load fixed set
                    with open(args.eval_fixedset, 'r', encoding='utf-8') as f:
                        fixedset = json.load(f)
                    # Factories to build env from fixed layout dict
                    def _make_env_from_layout(layout_dict: dict):
                        from src.env import RicochetRobotsEnv, FixedLayout
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
                            self._renderer = None
                            try:
                                from src.env.visuals.mpl_renderer import MatplotlibBoardRenderer
                                self._renderer = MatplotlibBoardRenderer(cell_size=28)
                            except Exception:
                                self._renderer = None
                        def _on_step(self) -> bool:
                            if self.num_timesteps % args.eval_freq == 0:
                                fixed_eval.evaluate(hub, _make_env_from_layout, self.model, step=self.num_timesteps)
                                gap_eval.evaluate(hub, _make_env_from_layout, self.model, step=self.num_timesteps)
                            # Trajectory recorder on its own cadence using a standard training env factory
                            if args.traj_record_freq > 0 and (self.num_timesteps - self._last_traj_dump) >= args.traj_record_freq:
                                # Capture RGB frames for video logging
                                def _make_renderable_env():
                                    e = make_env_factory(args)()
                                    # If env supports render_mode, set to rgb for video frames
                                    try:
                                        if hasattr(e, 'render_mode'):
                                            e.render_mode = 'rgb'
                                    except Exception:
                                        pass
                                    return e
                                rec = TrajectoryRecorder(episodes=args.traj_record_episodes, capture_render=True, renderer=self._renderer)
                                # If we have a bank curriculum, record one short trajectory per curriculum level
                                if curriculum_manager is not None and hasattr(curriculum_manager, 'curriculum_levels') and hasattr(curriculum_manager, 'bank'):
                                    try:
                                        from src.env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper
                                        import numpy as _np
                                        levels = getattr(curriculum_manager, 'curriculum_levels')
                                        bank_obj = getattr(curriculum_manager, 'bank')
                                        srt = getattr(curriculum_manager, 'success_rate_threshold', 0.8)
                                        min_eps = getattr(curriculum_manager, 'min_episodes_per_level', 100)
                                        win_sz = getattr(curriculum_manager, 'success_rate_window_size', 200)
                                        chk_freq = getattr(curriculum_manager, 'advancement_check_frequency', 50)
                                        for lvl_idx, lvl_spec in enumerate(levels):
                                            def _make_env_for_level(idx: int = lvl_idx):
                                                mgr = create_bank_curriculum_manager(
                                                    bank=bank_obj,
                                                    curriculum_levels=levels,
                                                    success_rate_threshold=srt,
                                                    min_episodes_per_level=min_eps,
                                                    success_rate_window_size=win_sz,
                                                    advancement_check_frequency=chk_freq,
                                                    verbose=False,
                                                )
                                                mgr.current_level = idx
                                                return BankCurriculumWrapper(
                                                    bank=bank_obj,
                                                    curriculum_manager=mgr,
                                                    obs_mode=(args.obs_mode if args.obs_mode in ("image", "rgb_image", "rgb_cell_image", "symbolic") else "rgb_image"),
                                                    channels_first=True,
                                                    render_mode=None,
                                                    cell_obs_pixel_size=getattr(args, "cell_obs_pixel_size", 128),
                                                    verbose=False,
                                                )
                                            rec_level = TrajectoryRecorder(episodes=1, capture_render=True, renderer=self._renderer)
                                            trajs = rec_level.record(_make_env_for_level, self.model)
                                            if trajs and trajs[0].get('frames'):
                                                frames = trajs[0]['frames'] or []
                                                video_thwc = _np.asarray(frames, dtype=_np.uint8)
                                                if video_thwc.ndim == 3:
                                                    video_thwc = video_thwc[:, :, :, None]
                                                if video_thwc.shape[0] == 1:
                                                    video_thwc = _np.concatenate([video_thwc, video_thwc], axis=0)
                                                hub.log_video(f"eval/trajectories/level{lvl_idx:02d}/video0", video_thwc, self.num_timesteps, fps=4)
                                                # Save local per-level MP4 for inspection
                                                try:
                                                    from pathlib import Path as _Path
                                                    import imageio.v2 as _imageio
                                                    _out_dir = ARTIFACTS_ROOT / "rollout_gifs"
                                                    _out_dir.mkdir(parents=True, exist_ok=True)
                                                    _mp4_path = _Path(_out_dir) / f"eval_traj_step{int(self.num_timesteps)}_L{lvl_idx:02d}.mp4"
                                                    _writer = _imageio.get_writer(_mp4_path, fps=4, codec="libx264", quality=7)
                                                    try:
                                                        for f in video_thwc:
                                                            _writer.append_data(_np.asarray(f, dtype=_np.uint8))
                                                    finally:
                                                        _writer.close()
                                                except Exception as _save_exc:
                                                    print(f"Warning: Local per-level eval video save failed at step {self.num_timesteps}: {_save_exc}")
                                        # After per-level logging, skip the generic single-level path
                                        self._last_traj_dump = self.num_timesteps
                                        return True
                                    except Exception as _per_level_exc:
                                        print(f"Warning: Per-level trajectory logging failed, falling back to default: {_per_level_exc}")
                                # Fallback: Use a fresh non-curriculum single-env factory mirroring current args
                                rec_env_factory = _make_renderable_env
                                trajectories = rec.record(rec_env_factory, self.model)
                                # Build compact text summary of actions per episode
                                try:
                                    summaries = []
                                    for idx, traj in enumerate(trajectories):
                                        acts = [str(step.get('action', -1)) for step in traj.get('steps', [])]
                                        succ = "✓" if bool(traj.get('success', False)) else "✗"
                                        summaries.append(f"ep {idx}: success={succ}, steps={len(acts)}, actions=[" + ",".join(acts[:60]) + (",..." if len(acts) > 60 else "]"))
                                    if summaries:
                                        hub.log_text("eval/trajectories/text", "\n".join(summaries), self.num_timesteps)
                                except Exception as exc:
                                    print(f"Warning: Trajectory text logging failed during eval at step {self.num_timesteps}: {exc}")
                                # If frames are present, log a short video for the first episode and save a local copy
                                first_frames: list[Any] = []
                                try:
                                    if trajectories and trajectories[0].get('frames'):
                                        first_frames = trajectories[0]['frames'] or []
                                        import numpy as _np
                                        video_thwc = _np.asarray(first_frames, dtype=_np.uint8)  # (T,H,W,C) expected
                                        # Ensure shape is (T,H,W,C)
                                        if video_thwc.ndim == 3:  # (T,H,W) -> add channel
                                            video_thwc = video_thwc[:, :, :, None]
                                        # Ensure at least 2 frames for encoders that reject 1-frame videos
                                        if video_thwc.shape[0] == 1:
                                            video_thwc = _np.concatenate([video_thwc, video_thwc], axis=0)
                                        hub.log_video("eval/trajectories/video0", video_thwc, self.num_timesteps, fps=4)

                                        # Also save a local MP4 for quick inspection with a standard codec
                                        try:
                                            from pathlib import Path as _Path
                                            _out_dir = ARTIFACTS_ROOT / "rollout_gifs"
                                            _out_dir.mkdir(parents=True, exist_ok=True)
                                            _mp4_path = _Path(_out_dir) / f"eval_traj_step{int(self.num_timesteps)}_ep0.mp4"
                                            import imageio.v2 as _imageio
                                            _writer = _imageio.get_writer(_mp4_path, fps=4, codec="libx264", quality=7)
                                            try:
                                                for f in video_thwc:
                                                    _writer.append_data(_np.asarray(f, dtype=_np.uint8))
                                            finally:
                                                _writer.close()
                                        except Exception as _save_exc:
                                            print(f"Warning: Local eval video save failed at step {self.num_timesteps}: {_save_exc}")
                                    else:
                                        print(f"Warning: No frames captured for eval trajectory video at step {self.num_timesteps}")
                                except Exception as exc:
                                    print(f"Warning: Trajectory video logging failed during eval at step {self.num_timesteps}: {exc}")
                                    try:
                                        if first_frames:
                                            hub.log_image("eval/trajectories/frame0", first_frames[0], self.num_timesteps)
                                    except Exception as img_exc:
                                        print(f"Warning: Trajectory image fallback failed during eval at step {self.num_timesteps}: {img_exc}")
                                self._last_traj_dump = self.num_timesteps
                            return True
                    callbacks.append(_HubEvalForward(verbose=0))
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Failed to configure fixed-set evaluator: {e}")
            # Always-on trajectory recorder (independent of fixedset) so it runs for default configs
            if hub is not None and (args.eval_fixedset is None):
                try:
                    class _HubTrajectoryRecorder(BaseCallback):
                        def __init__(self, verbose=0):
                            super().__init__(verbose)
                            self._last_traj_dump = 0
                            self._renderer = None
                            try:
                                from src.env.visuals.mpl_renderer import MatplotlibBoardRenderer
                                self._renderer = MatplotlibBoardRenderer(cell_size=28)
                            except Exception:
                                self._renderer = None
                        def _on_step(self) -> bool:
                            if args.traj_record_freq > 0 and (self.num_timesteps - self._last_traj_dump) >= args.traj_record_freq:
                                # Prefer per-level trajectories if using bank curriculum; else fallback to single-env
                                try:
                                    from src.monitoring.evaluators import TrajectoryRecorder
                                    if curriculum_manager is not None and hasattr(curriculum_manager, 'curriculum_levels') and hasattr(curriculum_manager, 'bank'):
                                        from src.env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper
                                        import numpy as _np
                                        levels = getattr(curriculum_manager, 'curriculum_levels')
                                        bank_obj = getattr(curriculum_manager, 'bank')
                                        srt = getattr(curriculum_manager, 'success_rate_threshold', 0.8)
                                        min_eps = getattr(curriculum_manager, 'min_episodes_per_level', 100)
                                        win_sz = getattr(curriculum_manager, 'success_rate_window_size', 200)
                                        chk_freq = getattr(curriculum_manager, 'advancement_check_frequency', 50)
                                        for lvl_idx, _ in enumerate(levels):
                                            def _make_env_for_level(idx: int = lvl_idx):
                                                mgr = create_bank_curriculum_manager(
                                                    bank=bank_obj,
                                                    curriculum_levels=levels,
                                                    success_rate_threshold=srt,
                                                    min_episodes_per_level=min_eps,
                                                    success_rate_window_size=win_sz,
                                                    advancement_check_frequency=chk_freq,
                                                    verbose=False,
                                                )
                                                mgr.current_level = idx
                                                return BankCurriculumWrapper(
                                                    bank=bank_obj,
                                                    curriculum_manager=mgr,
                                                    obs_mode=(args.obs_mode if args.obs_mode in ("image", "rgb_image", "rgb_cell_image", "symbolic") else "rgb_image"),
                                                    channels_first=True,
                                                    render_mode=None,
                                                    cell_obs_pixel_size=getattr(args, "cell_obs_pixel_size", 128),
                                                    verbose=False,
                                                )
                                            rec_level = TrajectoryRecorder(episodes=1, capture_render=True, renderer=self._renderer)
                                            trajs = rec_level.record(_make_env_for_level, self.model)
                                            if trajs and trajs[0].get('frames'):
                                                frames = trajs[0]['frames'] or []
                                                video_thwc = _np.asarray(frames, dtype=_np.uint8)
                                                if video_thwc.ndim == 3:
                                                    video_thwc = video_thwc[:, :, :, None]
                                                if video_thwc.shape[0] == 1:
                                                    video_thwc = _np.concatenate([video_thwc, video_thwc], axis=0)
                                                hub.log_video(f"eval/trajectories/level{lvl_idx:02d}/video0", video_thwc, self.num_timesteps, fps=4)
                                    else:
                                        # Fallback: single non-curriculum env
                                        def _make_renderable_env():
                                            e = make_env_factory(args)()
                                            try:
                                                if hasattr(e, 'render_mode'):
                                                    e.render_mode = 'rgb'
                                            except Exception:
                                                pass
                                            return e
                                        rec = TrajectoryRecorder(episodes=args.traj_record_episodes, capture_render=True, renderer=self._renderer)
                                        rec.record(_make_renderable_env, self.model)
                                except Exception as _exc:
                                    print(f"Warning: Trajectory recorder failed at step {self.num_timesteps}: {_exc}")
                                self._last_traj_dump = self.num_timesteps
                            return True
                    callbacks.append(_HubTrajectoryRecorder(verbose=0))
                except Exception as e:
                    print(f"Warning: Failed to configure trajectory recorder: {e}")
        if args.save_freq > 0:
            ckpt_cb = CheckpointCallback(save_freq=args.save_freq, save_path=(os.path.dirname(args.save_path) or "."),
                                         name_prefix=os.path.basename(args.save_path))
            callbacks.append(ckpt_cb)
    except ImportError:
        pass

    # Wrap training with profiling
    learn_kwargs = dict(total_timesteps=learn_timesteps, callback=callbacks if callbacks else None,
                        reset_num_timesteps=reset_num_timesteps)

    if learn_timesteps > 0:
        if args.enable_profiling:
            try:
                from src.profiling import profile
                with profile("training_total", track_memory=True):
                    model.learn(**learn_kwargs)
            except ImportError:
                model.learn(**learn_kwargs)
        else:
            model.learn(**learn_kwargs)
    else:
        print("No training timesteps requested; skipping model.learn().")
    
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
