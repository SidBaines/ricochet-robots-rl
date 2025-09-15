from __future__ import annotations

from typing import Any, List, Optional


class TensorBoardBackend:
    def __init__(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("TensorBoard not available") from e
        self._writer = SummaryWriter(log_dir=log_dir)

    def record_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        self._writer.add_scalar(key, value, global_step=step)

    def record_histogram(self, key: str, values: List[float], step: Optional[int] = None) -> None:
        import numpy as np
        self._writer.add_histogram(key, np.asarray(values), global_step=step)

    def record_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        self._writer.add_text(key, text, global_step=step)

    def record_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        import numpy as np
        arr = image
        if hasattr(image, "numpy"):
            arr = image.numpy()
        if isinstance(arr, list):
            import numpy as np
            arr = np.asarray(arr)
        # Expect HxWxC uint8 or CxHxW
        self._writer.add_image(key, arr, global_step=step, dataformats='HWC' if arr.ndim == 3 and arr.shape[-1] in (1,3,4) else 'CHW')

    def flush(self) -> None:
        self._writer.flush()


class WandbBackend:
    def __init__(self, project: str, entity: Optional[str] = None, run_name: Optional[str] = None, tags: Optional[List[str]] = None, config: Optional[dict] = None) -> None:
        try:
            import wandb  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("wandb not available") from e
        self._wandb = wandb
        self._wandb.init(project=project, entity=entity, name=run_name, tags=tags, config=config, reinit=True)

    def record_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        self._wandb.log({key: value}, step=step)

    def record_histogram(self, key: str, values: List[float], step: Optional[int] = None) -> None:
        self._wandb.log({key: self._wandb.Histogram(values)}, step=step)

    def record_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        self._wandb.log({key: text}, step=step)

    def record_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        self._wandb.log({key: self._wandb.Image(image)}, step=step)

    def flush(self) -> None:
        # wandb flushes on log
        return None


