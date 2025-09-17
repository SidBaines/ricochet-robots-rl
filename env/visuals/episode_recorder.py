from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from env.ricochet_core import Board
from env.visuals.renderer_base import BoardRenderer


@dataclass
class EpisodeStep:
    action: int
    reward: float
    is_success: bool


@dataclass
class EpisodeTrace:
    boards: List[Board]
    steps: List[EpisodeStep]
    info: Dict[str, Any]


class EpisodeRecorder:
    def __init__(self, renderer: BoardRenderer) -> None:
        self.renderer = renderer

    def record_single(self, env_factory, model, deterministic: bool = True) -> EpisodeTrace:
        env = env_factory()
        obs, _ = env.reset()
        boards: List[Board] = [env.get_board().clone()]
        steps: List[EpisodeStep] = []
        done = False
        trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, step_info = env.step(int(action))
            boards.append(env.get_board().clone())
            steps.append(EpisodeStep(action=int(action), reward=float(reward), is_success=bool(step_info.get("is_success", False))))
        final_info = step_info if isinstance(step_info, dict) else {}
        env.close()
        return EpisodeTrace(boards=boards, steps=steps, info=final_info)

    def to_rgb_frames(self, trace: EpisodeTrace) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for b in trace.boards:
            frames.append(self.renderer.draw_rgb(b))
        return frames


