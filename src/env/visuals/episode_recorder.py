from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th

from ..ricochet_core import Board
from .renderer_base import BoardRenderer


@dataclass
class EpisodeStep:
    action: int
    reward: float
    is_success: bool


@dataclass
class EpisodeTrace:
    boards: List[Board]
    steps: List[EpisodeStep]
    observations: List[Any]
    values: List[float]
    info: Dict[str, Any]


class EpisodeRecorder:
    def __init__(self, renderer: BoardRenderer) -> None:
        self.renderer = renderer

    def _predict_state_value(
        self,
        policy: Any,
        observation: Any,
        lstm_state: Optional[Any],
        episode_start: Optional[np.ndarray],
    ) -> float:
        """Best-effort value prediction that works for feedforward and recurrent policies."""
        obs_tensor, _ = policy.obs_to_tensor(observation)
        try:
            value_tensor = policy.predict_values(obs_tensor)
        except TypeError:
            # Recurrent policies expect (obs, lstm_state, episode_start)
            if lstm_state is None:
                lstm_state = policy.initial_state(obs_tensor.shape[0])
            if episode_start is None:
                episode_start = np.zeros((obs_tensor.shape[0],), dtype=np.float32)
            episode_start_tensor = th.as_tensor(episode_start).to(obs_tensor.device)
            value_tensor = policy.predict_values(obs_tensor, lstm_state, episode_start_tensor)

        array_like = value_tensor
        if hasattr(array_like, "detach"):
            array_like = array_like.detach()
        if hasattr(array_like, "cpu"):
            array_like = array_like.cpu()
        if hasattr(array_like, "numpy"):
            array_like = array_like.numpy()
        flat = np.asarray(array_like).reshape(-1)
        return float(flat[0]) if flat.size else float("nan")

    def record_single(self, env_factory, model, deterministic: bool = True) -> EpisodeTrace:
        env = env_factory()
        obs, _ = env.reset()
        boards: List[Board] = [env.get_board().clone()]
        observations: List[Any] = [obs]
        steps: List[EpisodeStep] = []
        values: List[float] = []
        done = False
        trunc = False
        policy = getattr(model, "policy", None)
        lstm_state: Optional[Any] = None
        episode_start = np.array([True], dtype=np.float32)
        while not (done or trunc):
            if policy is not None:
                try:
                    values.append(self._predict_state_value(policy, obs, lstm_state, episode_start))
                except Exception:
                    values.append(float("nan"))
            else:
                values.append(float("nan"))
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, trunc, step_info = env.step(int(action))
            boards.append(env.get_board().clone())
            observations.append(obs)
            steps.append(EpisodeStep(action=int(action), reward=float(reward), is_success=bool(step_info.get("is_success", False))))
            episode_start = np.array([float(done or trunc)], dtype=np.float32)
        final_info = step_info if isinstance(step_info, dict) else {}
        if policy is not None:
            try:
                values.append(self._predict_state_value(policy, obs, lstm_state, np.array([True], dtype=np.float32)))
            except Exception:
                values.append(float("nan"))
        else:
            values.append(float("nan"))
        env.close()
        return EpisodeTrace(boards=boards, steps=steps, observations=observations, values=values, info=final_info)

    def to_rgb_frames(self, trace: EpisodeTrace) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for b in trace.boards:
            frames.append(self.renderer.draw_rgb(b))
        return frames
