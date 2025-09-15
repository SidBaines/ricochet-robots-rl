from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    def __init__(self, episodes: int = 5) -> None:
        self.episodes = int(episodes)

    def record(self, make_env_fn, model) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []
        for _ in range(self.episodes):
            env = make_env_fn()
            obs, info = env.reset()
            done = False
            trunc = False
            steps: List[Dict[str, Any]] = []
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, trunc, info = env.step(int(action))
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
                "success": bool(info.get("is_success", False))
            })
            env.close()
        return trajectories


