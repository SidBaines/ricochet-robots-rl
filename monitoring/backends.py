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

    def record_video(self, key: str, video_frames: Any, step: Optional[int] = None, fps: int = 4) -> None:
        """Log a video given frames.

        Accepts (T, H, W, C) uint8 and converts to TensorBoard required shape (N, C, T, H, W).
        """
        import numpy as np
        arr = video_frames
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        if isinstance(arr, list):
            arr = np.asarray(arr)
        
        # Ensure dtype uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
            
        # Convert (T, H, W, C) to (N, C, T, H, W) for TensorBoard
        if arr.ndim == 4:  # (T, H, W, C)
            T, H, W, C = arr.shape
            # Transpose to (C, T, H, W) then add batch dimension
            arr = np.transpose(arr, (3, 0, 1, 2))  # (C, T, H, W)
            arr = arr[np.newaxis, ...]  # (1, C, T, H, W)
        elif arr.ndim == 5:
            # Already has batch dimension, check if we need to transpose
            if arr.shape[1] in (1, 3, 4):  # (N, C, T, H, W) - already correct
                pass
            elif arr.shape[-1] in (1, 3, 4):  # (N, T, H, W, C) -> (N, C, T, H, W)
                arr = np.transpose(arr, (0, 4, 1, 2, 3))
            else:
                return  # Unknown format
        else:
            return  # Wrong number of dimensions
            
        try:
            self._writer.add_video(key, arr, global_step=step, fps=fps)
        except Exception as e:
            # Best-effort: skip if TB can't encode
            print(f"TensorBoard video logging failed: {e}")
            pass

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

    def record_video(self, key: str, video_frames: Any, step: Optional[int] = None, fps: int = 4) -> None:
        import numpy as np
        arr = video_frames
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        if isinstance(arr, list):
            arr = np.asarray(arr)
        
        # Handle different input formats
        if arr.ndim == 5:
            # (N, C, T, H, W) or (N, T, H, W, C) -> pick first batch
            if arr.shape[1] in (1, 3, 4):
                # (N, C, T, H, W) -> (T, H, W, C)
                arr = np.transpose(arr[0], (1, 2, 3, 0))
            else:
                # (N, T, H, W, C)
                arr = arr[0]
        elif arr.ndim == 3:
            # (T, H, W) -> (T, H, W, 1) for grayscale
            arr = arr[:, :, :, np.newaxis]
        
        if arr.ndim != 4:
            print(f"Warning: Expected 4D video array (T, H, W, C), got {arr.ndim}D with shape {arr.shape}")
            return
            
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
            
        # Ensure we have at least 2 frames for a video
        if arr.shape[0] < 2:
            print(f"Warning: Video has only {arr.shape[0]} frames, logging as image instead")
            try:
                self._wandb.log({key+"/frame0": self._wandb.Image(arr[0])}, step=step)
            except Exception:
                pass
            return
            
        try:
            self._wandb.log({key: self._wandb.Video(arr, fps=fps, format="mp4")}, step=step)
        except Exception as e:
            print(f"Warning: wandb video logging failed: {e}")
            # Fallback: log first frame as image if video fails
            try:
                self._wandb.log({key+"/frame0": self._wandb.Image(arr[0])}, step=step)
            except Exception:
                pass

    def flush(self) -> None:
        # wandb flushes on log
        return None


