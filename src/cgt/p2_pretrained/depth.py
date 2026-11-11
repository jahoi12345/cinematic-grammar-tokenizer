"""Depth Anything V2 (small) depth maps and a depth-of-field estimate from keyframes."""
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _pipe = pipeline("depth-estimation", model=_MODEL, device=dev)
    return _pipe


def depth_map(bgr: np.ndarray) -> np.ndarray:
    """Relative depth in [0, 1] at frame resolution; 1 = nearest (model predicts inverse depth)."""
    out = _get_pipe()(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    d = out["predicted_depth"].squeeze().float().cpu().numpy()
    d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(d.min()), float(d.max())
    return ((d - lo) / (hi - lo + 1e-9)).astype(np.float32)


def dof_features(bgr: np.ndarray, depth: np.ndarray) -> dict:
    """Sharpness (Laplacian variance) of the nearest 30% of pixels vs the farthest 30%.
    High near/far ratio => blurred background => shallow DoF."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    near = depth >= np.quantile(depth, 0.7)
    far = depth <= np.quantile(depth, 0.3)
    sharp_near = float((lap[near] ** 2).mean())
    sharp_far = float((lap[far] ** 2).mean())
    log_ratio = float(np.log((sharp_near + 1e-3) / (sharp_far + 1e-3)))
    return {"dof_log_ratio": log_ratio, "dof": dof_label(log_ratio)}


def dof_label(log_ratio: float, shallow: float = 1.4, deep: float = 0.4) -> str:
    """Provisional cutoffs (ratio ~4x / ~1.5x); tune on the dev set (never the gold set)."""
    if log_ratio >= shallow:
        return "shallow"
    if log_ratio <= deep:
        return "deep"
    return "medium"
