"""Editing rhythm from shot boundaries."""
import numpy as np


def shot_lengths(shots) -> np.ndarray:
    return np.array([e - s for s, e in shots], dtype=float)


def asl(shots) -> float:
    """Average shot length in seconds."""
    return float(shot_lengths(shots).mean())


def rolling_asl(shots, window_s: float = 300.0, step_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Cutting speed over time: (window centres, ASL of shots starting inside each window)."""
    starts = np.array([s for s, _ in shots])
    lens = shot_lengths(shots)
    end = shots[-1][1]
    centres, vals = [], []
    t = 0.0
    while t < end:
        m = (starts >= t) & (starts < t + window_s)
        if m.any():
            centres.append(t + window_s / 2)
            vals.append(lens[m].mean())
        t += step_s
    return np.array(centres), np.array(vals)
