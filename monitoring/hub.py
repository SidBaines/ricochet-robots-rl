from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class LoggerBackend(Protocol):
    def record_scalar(self, key: str, value: float, step: Optional[int] = None) -> None: ...
    def record_histogram(self, key: str, values: List[float], step: Optional[int] = None) -> None: ...
    def record_text(self, key: str, text: str, step: Optional[int] = None) -> None: ...
    def record_image(self, key: str, image: Any, step: Optional[int] = None) -> None: ...
    def flush(self) -> None: ...


class Collector(Protocol):
    def on_step(self, hub: "MonitoringHub", context: Dict[str, Any]) -> None: ...
    def on_rollout_end(self, hub: "MonitoringHub", context: Dict[str, Any]) -> None: ...
    def on_level_change(self, hub: "MonitoringHub", context: Dict[str, Any]) -> None: ...
    def on_eval(self, hub: "MonitoringHub", context: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...


@dataclass
class MonitoringConfig:
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    tensorboard_log_dir: str = "runs/ppo"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    collectors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluators: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MonitoringHub:
    def __init__(self, *, backends: List[LoggerBackend], collectors: List[Collector]) -> None:
        self._backends = backends
        self._collectors = collectors

    # Logging API
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        for b in self._backends:
            b.record_scalar(key, value, step)

    def log_histogram(self, key: str, values: List[float], step: Optional[int] = None) -> None:
        for b in self._backends:
            b.record_histogram(key, values, step)

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        for b in self._backends:
            b.record_text(key, text, step)

    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        for b in self._backends:
            b.record_image(key, image, step)

    def flush(self) -> None:
        for b in self._backends:
            b.flush()

    # Event routing
    def on_step(self, context: Dict[str, Any]) -> None:
        for c in self._collectors:
            c.on_step(self, context)

    def on_rollout_end(self, context: Dict[str, Any]) -> None:
        for c in self._collectors:
            c.on_rollout_end(self, context)

    def on_level_change(self, context: Dict[str, Any]) -> None:
        for c in self._collectors:
            c.on_level_change(self, context)

    def on_eval(self, context: Dict[str, Any]) -> None:
        for c in self._collectors:
            c.on_eval(self, context)

    def close(self) -> None:
        for c in self._collectors:
            c.close()
        for b in self._backends:
            b.flush()


