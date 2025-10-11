from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

from .hub import MonitoringHub


def _board_signature(layout: Dict[str, Any]) -> str:
    # Serialize fixed layout dict into a deterministic signature
    import hashlib
    payload = json.dumps(layout, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class SolverCache:
    path: str
    data: Dict[str, Dict[str, Any]]

    @classmethod
    def load(cls, path: str) -> "SolverCache":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
        return SolverCache(path=path, data=data)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f)


class FixedSetEvaluator:
    def __init__(self, fixedset: List[Dict[str, Any]], episodes: int = 50) -> None:
        self.fixedset = fixedset
        self.episodes = int(episodes)

    def evaluate(self, hub: MonitoringHub, make_env_fn, model, step: Optional[int] = None) -> Dict[str, Any]:
        import numpy as np
        success = 0
        lengths: List[int] = []
        noop_counts: List[int] = []
        for i in range(min(self.episodes, len(self.fixedset))):
            layout = self.fixedset[i]
            env = make_env_fn(layout)
            obs, info = env.reset()
            done = False
            trunc = False
            steps = 0
            noops = 0
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                is_noop = (getattr(env, "_noop_action", None) is not None and int(action) == env._noop_action)
                if is_noop:
                    noops += 1
                obs, _, done, trunc, info = env.step(int(action))
                steps += 1
            if info.get("is_success", False):
                success += 1
            lengths.append(steps)
            noop_counts.append(noops)
            env.close()
        rate = success / max(1, min(self.episodes, len(self.fixedset)))
        hub.log_scalar("eval/success_rate", float(rate), step)
        if lengths:
            hub.log_scalar("eval/episode_length/mean", float(sum(lengths) / len(lengths)), step)
        if noop_counts:
            hub.log_scalar("eval/actions/noop_rate", float(sum(noop_counts) / max(1, sum(lengths))), step)
        return {"success_rate": rate, "lengths": lengths, "noop_counts": noop_counts}


class OptimalityGapEvaluator:
    def __init__(self, fixedset: List[Dict[str, Any]], solver_cache: SolverCache, solver_limits: Dict[str, int], skip_unsolved: bool = True) -> None:
        self.fixedset = fixedset
        self.cache = solver_cache
        self.limits = solver_limits
        self.skip_unsolved = bool(skip_unsolved)

    def _build_env_from_layout(self, layout_dict: Dict[str, Any]):
        from env import RicochetRobotsEnv, FixedLayout
        import numpy as np
        h = int(layout_dict["height"]) ; w = int(layout_dict["width"]) ; tr = int(layout_dict["target_robot"]) ; robots = {int(k): tuple(v) for k, v in layout_dict["robot_positions"].items()}
        h_walls = np.array(layout_dict["h_walls"], dtype=bool)
        v_walls = np.array(layout_dict["v_walls"], dtype=bool)
        goal = tuple(layout_dict["goal_position"])  # type: ignore
        fl = FixedLayout(height=h, width=w, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=tr)
        return RicochetRobotsEnv(fixed_layout=fl, obs_mode="image", channels_first=True)

    def _ensure_optimal_length(self, layout: Dict[str, Any], sig: str) -> Optional[int]:
        if sig in self.cache.data and "optimal_length" in self.cache.data[sig]:
            return int(self.cache.data[sig]["optimal_length"])
        # Compute
        try:
            from env.solver import solve_bfs
            env = self._build_env_from_layout(layout)
            board = env.get_board()
            actions = solve_bfs(board, max_depth=int(self.limits.get("max_depth", 60)), max_nodes=int(self.limits.get("max_nodes", 50000)))
            env.close()
            if actions is None:
                self.cache.data[sig] = {"optimal_length": None}
                return None
            opt = len(actions)
            self.cache.data[sig] = {"optimal_length": opt}
            self.cache.save()
            return opt
        except Exception:
            return None

    def evaluate(self, hub: MonitoringHub, make_env_fn, model, step: Optional[int] = None) -> Dict[str, Any]:
        gaps: List[int] = []
        considered = 0
        for layout in self.fixedset:
            sig = _board_signature(layout)
            opt = self._ensure_optimal_length(layout, sig)
            if opt is None and self.skip_unsolved:
                continue
            env = make_env_fn(layout)
            obs, info = env.reset()
            done = False
            trunc = False
            steps = 0
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(int(action))
                steps += 1
            env.close()
            if info.get("is_success", False) and opt is not None:
                gaps.append(int(steps - opt))
            considered += 1
        if gaps:
            import numpy as np
            hub.log_scalar("eval/optimality_gap/mean", float(sum(gaps) / len(gaps)), step)
            hub.log_scalar("eval/optimality_gap/median", float(np.median(gaps)), step)
        hub.log_scalar("eval/optimality_gap/num_evaluated", float(considered), step)
        return {"gaps": gaps, "considered": considered}


class TrajectoryRecorder:
    def __init__(self, episodes: int = 5, capture_render: bool = False, renderer: Optional[Any] = None) -> None:
        self.episodes = int(episodes)
        self.capture_render = bool(capture_render)
        self._renderer = renderer
        self._default_renderer: Optional[Any] = None
        self._default_renderer_failed = False

    def _get_renderer(self) -> Optional[Any]:
        if self._renderer is not None:
            return self._renderer
        if self._default_renderer_failed:
            return None
        if self._default_renderer is None:
            try:
                from env.visuals.mpl_renderer import MatplotlibBoardRenderer  # Local import to avoid mandatory dependency
                self._default_renderer = MatplotlibBoardRenderer(cell_size=24)
            except Exception:
                self._default_renderer_failed = True
                return None
        return self._default_renderer

    def _capture_frame(self, env: Any) -> Optional[Any]:
        renderer = self._get_renderer()
        frame: Optional[Any] = None
        if renderer is not None:
            getter = getattr(env, "get_board", None)
            if callable(getter):
                try:
                    board = getter()
                    if hasattr(board, "clone"):
                        board = board.clone()
                    frame = renderer.draw_rgb(board)
                except Exception:
                    frame = None
        if frame is None:
            try:
                frame = env.render()
            except Exception:
                frame = None
        if frame is None:
            return None
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        try:
            import numpy as _np
            arr = _np.asarray(frame)
            if arr.ndim < 2:
                return None
            if arr.dtype != _np.uint8:
                arr = _np.clip(arr, 0, 255).astype(_np.uint8)
            return arr
        except Exception:
            return None

    def record(self, make_env_fn, model) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []
        for _ in range(self.episodes):
            env = make_env_fn()
            obs, info = env.reset()
            done = False
            trunc = False
            steps: List[Dict[str, Any]] = []
            frames: List[Any] = []  # type: ignore
            if self.capture_render:
                initial = self._capture_frame(env)
                if initial is not None:
                    frames.append(initial)
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, trunc, info = env.step(int(action))
                if self.capture_render:
                    captured = self._capture_frame(env)
                    if captured is not None:
                        frames.append(captured)
                steps.append({
                    "action": int(action),
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(trunc),
                    "is_success": bool(info.get("is_success", False)),
                })
                obs = next_obs
            trajectories.append({
                "steps": steps,
                "success": bool(info.get("is_success", False)),
                "frames": frames if self.capture_render else None,
            })
            env.close()
        return trajectories



def rollout_episode_with_recurrent_support(env_factory: Callable[[], Any], model: Any, deterministic: bool = True) -> Dict[str, Any]:
    """Run a single episode handling both sb3-contrib RecurrentPPO and custom recurrent policies.

    Returns a dict with success flag and length.
    """
    env = env_factory()
    obs, info = env.reset()
    done = False
    trunc = False
    steps = 0

    # sb3-contrib RecurrentPPO state tracking
    lstm_state = None
    episode_starts = True

    # Custom policy detection
    policy_obj = getattr(model, 'policy', model)
    custom_recurrent = hasattr(policy_obj, 'set_episode_starts')

    while not (done or trunc):
        if hasattr(model, 'predict'):
            # Try recurrent signature first
            try:
                action_out = model.predict(obs, state=lstm_state, episode_start=episode_starts, deterministic=deterministic)  # type: ignore[arg-type]
                if isinstance(action_out, tuple) and len(action_out) == 2 and not isinstance(action_out[1], (tuple, list)):
                    action, _ = action_out
                else:
                    action, lstm_state = action_out  # type: ignore[assignment]
            except TypeError:
                # Fallback for feedforward models
                action, _ = model.predict(obs, deterministic=deterministic)
        else:
            # Assume custom policy-like API with _predict
            if custom_recurrent:
                policy_obj.set_episode_starts([episode_starts])
            action, _ = policy_obj._predict(obs, deterministic=deterministic)  # type: ignore[attr-defined]

        obs, _, done, trunc, info = env.step(int(action))
        if custom_recurrent:
            policy_obj.set_episode_starts([done or trunc])
        episode_starts = done or trunc
        steps += 1

    result = {"is_success": info.get("is_success", False), "length": steps}
    env.close()
    return result
