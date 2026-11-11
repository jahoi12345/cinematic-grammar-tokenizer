"""helpers for the descriptive GeoCalib field-of-view probe of the ShotQA lens classes (docs/plans/geocalib_lens_probe.md).
Pure functions only (no GeoCalib import here)."""
from collections.abc import Sequence

import numpy as np

CLASSES = ["Ultra Wide / Fisheye", "Wide", "Medium", "Long Lens"]   # widest first
SHORT = {"Ultra Wide / Fisheye": "ultra_wide", "Wide": "wide", "Medium": "medium", "Long Lens": "long"}
SEED, PER_CLASS = 28, 150
# step 4 still-photo convention on horizontal FoV (degrees), NOT a ShotQA definition; never tuned
BANDS = [("ultra_wide", 84.0, None), ("wide", 54.0, 84.0), ("medium", 29.0, 54.0), ("long", None, 29.0)]
BAR_MEAN, BAR_P99, BAR_MAX_SIDE, BAR_MIN_KEEP = 0.06, 0.12, 0.30, 0.40


def luma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float64) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _bar_run(lines: np.ndarray, limit: int) -> int:
    """Number of leading lines (rows of `lines`, each a 1-D luma line) that are black bars, at most `limit`."""
    n = 0
    for line in lines[:limit]:
        if line.mean() <= BAR_MEAN and np.percentile(line, 99) <= BAR_P99:
            n += 1
        else:
            break
    return n


def crop_black_bars(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Strip contiguous black letterbox rows / pillarbox columns from each edge (plan section 3). Returns (cropped image, info);
    info = {"top", "bottom", "left", "right", "cropped", "refused", "capped"}. With the 30% per-side cap at least 40% always
    remains, so `refused` is a safety net; `capped` flags frames where a side hit the cap. If less than BAR_MIN_KEEP of the height or width would
    remain, nothing is cropped and refused=True."""
    y = luma(rgb)
    h, w = y.shape
    lh, lw = int(BAR_MAX_SIDE * h), int(BAR_MAX_SIDE * w)
    t, b = _bar_run(y, lh), _bar_run(y[::-1], lh)
    left, right = _bar_run(y.T, lw), _bar_run(y.T[::-1], lw)
    info = {"top": t, "bottom": b, "left": left, "right": right, "refused": False,
            "capped": bool(t == lh or b == lh or left == lw or right == lw)}   # a side hit the 30% cap (e.g. a black frame)
    if (h - t - b) < BAR_MIN_KEEP * h or (w - left - right) < BAR_MIN_KEEP * w:
        info.update(top=0, bottom=0, left=0, right=0, refused=True)
        info["cropped"] = False
        return rgb, info
    info["cropped"] = bool(t or b or left or right)
    return rgb[t:h - b, left:w - right], info


def fov_band(hfov_deg: float) -> str:
    """Still-photo band of a horizontal FoV in degrees (ultra_wide > 84, wide [54, 84], medium [29, 54), long < 29)."""
    if not np.isfinite(hfov_deg):
        raise ValueError("non-finite FoV")
    if hfov_deg > 84.0:
        return "ultra_wide"
    if hfov_deg >= 54.0:
        return "wide"
    if hfov_deg >= 29.0:
        return "medium"
    return "long"


def balanced_sample(by_class: dict[str, Sequence[str]], per_class: int = PER_CLASS, seed: int = SEED) -> dict[str, list[str]]:
    """For each class in CLASSES order: sorted ids permuted with ONE np.random.default_rng(seed); first per_class taken."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in CLASSES:
        ids = sorted(by_class[c])
        if len(ids) < per_class:
            raise ValueError(f"{c}: only {len(ids)} ids")
        perm = rng.permutation(len(ids))
        out[c] = [ids[i] for i in perm[:per_class]]
    return out


def auc(x: Sequence[float], pos: Sequence[bool]) -> float | None:
    """Mann-Whitney AUC (ties count 1/2): P(score of a positive > score of a negative). None if a group is empty."""
    x = np.asarray(x, float)
    pos = np.asarray(pos, bool)
    a, b = x[pos], x[~pos]
    if len(a) == 0 or len(b) == 0:
        return None
    gt = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(gt / (len(a) * len(b)))


def cluster_bootstrap_auc(x: Sequence[float], pos: Sequence[bool], clusters: Sequence[str], n: int = 2000,
                          seed: int = SEED) -> tuple[float, float] | None:
    """95% percentile interval of the AUC, resampling clusters (films) with replacement."""
    x, pos, clusters = np.asarray(x, float), np.asarray(pos, bool), np.asarray(clusters)
    keys = np.unique(clusters)
    idx = {k: np.flatnonzero(clusters == k) for k in keys}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        v = auc(x[pick], pos[pick])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def quantiles(v: Sequence[float]) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            **{f"q{p}": float(x) for p, x in zip((10, 25, 50, 75, 90), q, strict=True)}}


def reading(pairs: dict) -> dict:
    """Pre-registered reading (plan section 5). `pairs[name]` has hfov_auc, hfov_ci (lo, hi) and cv_gain for the names
    'wide_vs_medium' and 'medium_vs_long'."""
    near = [pairs["wide_vs_medium"], pairs["medium_vs_long"]]
    yes = [p["hfov_auc"] >= 0.65 and p["hfov_ci"][0] > 0.55 and p["cv_gain"] >= 0.03 for p in near]
    no = all((p["hfov_auc"] < 0.58 or p["hfov_ci"][0] <= 0.50) and p["cv_gain"] < 0.01 for p in near)
    verdict = "yes" if any(yes) else ("no" if no else "unclear")
    return {"verdict": verdict, "yes_by_pair": dict(zip(("wide_vs_medium", "medium_vs_long"), yes, strict=True))}
