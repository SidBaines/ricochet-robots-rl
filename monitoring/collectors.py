from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from .hub import MonitoringHub


class BaseCollector:
    def on_step(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        return None

    def on_rollout_end(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        return None

    def on_level_change(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        return None

    def on_eval(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class EpisodeStatsCollector(BaseCollector):
    def __init__(self, window_size: int = 200) -> None:
        self.window_size = int(window_size)
        self.success_window: Deque[int] = deque(maxlen=self.window_size)
        self.length_window: Deque[int] = deque(maxlen=self.window_size)
        self.return_window: Deque[float] = deque(maxlen=self.window_size)

    def on_rollout_end(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        # Expect context to provide lists over finished episodes in this rollout
        successes: List[int] = context.get("rollout_successes", [])
        lengths: List[int] = context.get("rollout_lengths", [])
        returns: List[float] = context.get("rollout_returns", [])
        for s in successes:
            self.success_window.append(int(bool(s)))
        for l in lengths:
            self.length_window.append(int(l))
        for r in returns:
            self.return_window.append(float(r))
        step = context.get("global_step")
        if len(self.success_window) > 0:
            hub.log_scalar("train/success_rate", sum(self.success_window) / len(self.success_window), step)
        if len(self.length_window) > 0:
            hub.log_scalar("train/episode_length/mean", sum(self.length_window) / len(self.length_window), step)
        if len(self.return_window) > 0:
            hub.log_scalar("train/return/mean", sum(self.return_window) / len(self.return_window), step)


class ActionUsageCollector(BaseCollector):
    def __init__(self, track_noop: bool = True) -> None:
        self.track_noop = bool(track_noop)
        self.total_actions = 0
        self.noop_actions = 0
        self.current_noop_streak = 0
        self.max_noop_streak = 0

    def on_step(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        action = context.get("last_action")
        is_noop = context.get("last_action_is_noop", False)
        step = context.get("global_step")
        if action is None:
            return None
        self.total_actions += 1
        if self.track_noop:
            if is_noop:
                self.noop_actions += 1
                self.current_noop_streak += 1
                self.max_noop_streak = max(self.max_noop_streak, self.current_noop_streak)
            else:
                self.current_noop_streak = 0
        if self.total_actions % 1000 == 0:
            if self.total_actions > 0 and self.track_noop:
                hub.log_scalar("train/actions/noop_rate", self.noop_actions / max(1, self.total_actions), step)
                hub.log_scalar("train/actions/noop_streak_max", float(self.max_noop_streak), step)


class CurriculumProgressCollector(BaseCollector):
    def __init__(self) -> None:
        self.last_level = None

    def on_level_change(self, hub: MonitoringHub, context: Dict[str, Any]) -> None:
        level_id = context.get("level_id")
        level_name = context.get("level_name")
        step = context.get("global_step")
        if level_id is not None:
            hub.log_scalar("curriculum/level", float(level_id), step)
            if level_name is not None:
                hub.log_text("curriculum/level_name", str(level_name), step)
        stats = context.get("level_stats")
        if isinstance(stats, dict):
            if "success_rate" in stats:
                hub.log_scalar("curriculum/success_rate", float(stats["success_rate"]), step)
            if "episodes_at_level" in stats:
                hub.log_scalar("curriculum/episodes_at_level", float(stats["episodes_at_level"]), step)
            if "total_episodes" in stats:
                hub.log_scalar("curriculum/total_episodes", float(stats["total_episodes"]), step)


