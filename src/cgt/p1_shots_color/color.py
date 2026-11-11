"""Per-frame color features. Frames are BGR uint8 arrays (OpenCV convention)."""
import warnings

import cv2
import numpy as np
from sklearn.cluster import KMeans

_WORK_WIDTH = 160


def _small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= _WORK_WIDTH:
        return frame
    return cv2.resize(frame, (_WORK_WIDTH, max(1, round(h * _WORK_WIDTH / w))), interpolation=cv2.INTER_AREA)


def _lab(frame: np.ndarray) -> np.ndarray:
    """Float LAB: L in 0-100, a*/b* signed."""
    return cv2.cvtColor(_small(frame).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _cluster(frame: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pix = _lab(frame).reshape(-1, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=1, random_state=0).fit(pix)
    shares = np.bincount(km.labels_, minlength=k) / len(pix)
    order = np.argsort(-shares, kind="stable")
    return km.cluster_centers_[order], shares[order]


def palette_lab(frame: np.ndarray, k: int = 5) -> np.ndarray:
    """(k, 3) LAB cluster centres, most common first."""
    return _cluster(frame, k)[0]


def palette_shares(frame: np.ndarray, k: int = 5) -> np.ndarray:
    """(k,) pixel share of each palette colour, descending, sums to 1."""
    return _cluster(frame, k)[1]


def color_stats(frame: np.ndarray) -> dict:
    small = _small(frame)
    hsv = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    lab = _lab(frame)
    return {
        "sat_mean": float(hsv[..., 1].mean()),
        "val_mean": float(hsv[..., 2].mean()),
        "warmth": float(lab[..., 2].mean()),  # mean b*: >0 warm/yellow, <0 cool/blue
        "contrast": float(lab[..., 0].std()),  # std of luminance, 0-100 scale
    }


def lighting_key(frame: np.ndarray) -> str:
    """Rough key from the luminance distribution (thresholds provisional; tune on the dev set, never the gold set)."""
    med = float(np.median(_lab(frame)[..., 0]))
    if med < 35:
        return "low-key"
    if med > 65:
        return "high-key"
    return "mid-key"
