"""Depth-parallax features (separate module, own cache key; docs/phase4_interface.md section 3b). CPU only, Depth-Anything-V2-Small (inverse relative depth, 1 = nearest,
per frame min-max normalised, centred per frame). Idea: rotation (pan/tilt) and zoom produce flow that does NOT depend on depth, while translation (truck, boom, dolly)
produces flow that scales with inverse depth. Per frame pair the dense flow on a grid is fitted by
    u = a0 + a1 x + a2 y  +  d_c (c0 + c1 x + c2 y),    (same form for v)     with d_c = inverse depth - its frame mean,
so [a] is the depth-independent (rotation/zoom-like) part and [c] the depth-dependent (translation-parallax) part: ptx, pty = c0 of u and v, pdiv = (c1_u + c2_v) / 2.
Depth is computed on every second sampled frame (about 6 per second) and reused for the next frame's pair (one frame of camera motion of misalignment, small on a stride-8 grid).
Units as flow.py (normalised, per second after scaling). Never reads ShotBench or CameraBench."""
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import torch

from cgt.p4_camera_move import flow as F

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_WIDTH, DEPTH_EVERY, MAP_W = 392, 2, 96
PARAMS = {"model": MODEL, "depth_width": DEPTH_WIDTH, "depth_every": DEPTH_EVERY, "ridge": 1e-4, "huber_k": 1.345, "iters": 5, "flow_key": None}
KEYS = ["ptx", "pty", "pdiv", "corr", "pshare", "r2_gain", "tx", "ty", "div", "rot"]
PER_SECOND = {"ptx", "pty", "pdiv", "tx", "ty", "div", "rot"}
_MEAN, _STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
_model = None


def cache_key() -> str:
    """sha256 of this module's source + parameters + the flow module key (the flow frames and fits are reused)."""
    return hashlib.sha256(Path(__file__).read_bytes() + json.dumps(PARAMS | {"flow_key": F.cache_key()}, sort_keys=True).encode()).hexdigest()[:16]


def clip_key(path: str | Path) -> str:
    """Per-clip cache key = code key + first 12 hex of the clip's content SHA-256."""
    return f"{cache_key()}.{F.file_digest(path)[:12]}"


def _get_model():
    global _model
    if _model is None:
        from transformers import AutoModelForDepthEstimation
        _model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    return _model


def inverse_depth(bgr: np.ndarray) -> np.ndarray:
    """Relative inverse depth in [0, 1] (1 = nearest) at MAP_W px wide (aspect kept), float32."""
    h0, w0 = bgr.shape[:2]
    w = round(DEPTH_WIDTH / 14) * 14
    h = max(14, round(w * h0 / w0 / 14) * 14)
    x = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255 - _MEAN) / _STD
    with torch.no_grad():
        d = _get_model()(torch.from_numpy(x.transpose(2, 0, 1))[None]).predicted_depth[0].float().numpy()
    mh = max(2, round(MAP_W * h0 / w0))
    d = cv2.resize(d, (MAP_W, mh), interpolation=cv2.INTER_AREA)
    return ((d - d.min()) / (d.max() - d.min() + 1e-9)).astype(np.float32)


def _frames(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """(colour frames at WIDTH px wide, grey frames, fps_used), sampled exactly like flow.sample_gray."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(src / F.FPS))
    col, gray, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = round(fr.shape[0] * F.WIDTH / fr.shape[1] / 2) * 2
            c = cv2.resize(fr, (F.WIDTH, h), interpolation=cv2.INTER_AREA)
            col.append(c)
            gray.append(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    if len(col) < 3:
        raise ValueError(f"{path}: only {len(col)} frames")
    return col, gray, src / step


def fit_depth_flow(flow: np.ndarray, depth_map: np.ndarray, step: int = F.STEP, iters: int = 5) -> dict:
    """Robust fit of the depth-independent + depth-dependent affine model described in the module docstring on a stride-`step` grid of the dense flow."""
    h, w = flow.shape[:2]
    half = w / 2.0
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    xn, yn = ((xs - w / 2.0) / half).ravel(), ((ys - h / 2.0) / half).ravel()
    uv = flow[ys.ravel(), xs.ravel()] / half
    d = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)[ys.ravel(), xs.ravel()]
    d = d - d.mean()
    A = np.stack([np.ones_like(xn), xn, yn, d, d * xn, d * yn], axis=1)
    lam = 1e-4 * len(xn)

    def solve(cols, wt):
        Aw = A[:, cols] * np.sqrt(wt)[:, None]
        reg = np.sqrt(lam) * np.eye(len(cols))
        c, *_ = np.linalg.lstsq(np.vstack([Aw, reg]), np.vstack([uv * np.sqrt(wt)[:, None], np.zeros((len(cols), 2))]), rcond=None)
        return c

    out = {}
    for name, cols in (("aff", [0, 1, 2]), ("full", [0, 1, 2, 3, 4, 5])):
        wt = np.ones(len(xn))
        for _ in range(iters):
            coef = solve(cols, wt)
            mag = np.linalg.norm(uv - A[:, cols] @ coef, axis=1)
            wt = np.minimum(1.0, 1.345 * 1.4826 * (np.median(mag) + 1e-9) / (mag + 1e-12))
        res = uv - A[:, cols] @ coef
        out[name] = (coef, 1.0 - float((res ** 2).sum()) / (float((uv ** 2).sum()) + 1e-12))
    coef, r2_full = out["full"]
    dep_part = A[:, 3:] @ coef[3:]
    tot_fit = A @ coef
    mag = np.linalg.norm(uv, axis=1)
    corr = float(np.corrcoef(mag, d)[0, 1]) if mag.std() > 1e-9 and d.std() > 1e-9 else 0.0
    return {"ptx": float(coef[3, 0]), "pty": float(coef[3, 1]), "pdiv": float((coef[4, 0] + coef[5, 1]) / 2), "corr": corr,
            "pshare": float((dep_part ** 2).sum() / ((tot_fit ** 2).sum() + 1e-12)), "r2_gain": float(r2_full - out["aff"][1])}


def clip_series(path: str | Path) -> dict[str, np.ndarray]:
    """Per frame-pair series of KEYS (per-second units for PER_SECOND) + `depth` (the stack of small depth maps, float16) + fps_used."""
    col, gray, fps_used = _frames(Path(path))
    dmaps = {i: inverse_depth(col[i]) for i in range(0, len(col), DEPTH_EVERY)}
    rows = []
    for i, (a, b) in enumerate(pairwise(gray)):
        flow = F.dis_flow(a, b)
        f0 = F.fit_affine(flow)
        r = fit_depth_flow(flow, dmaps[i - i % DEPTH_EVERY])
        rows.append(r | {k: f0[k] for k in ("tx", "ty", "div", "rot")})
    out = {k: np.array([r[k] for r in rows], dtype=np.float64) * (fps_used if k in PER_SECOND else 1.0) for k in KEYS}
    out["depth"] = np.stack([dmaps[i] for i in sorted(dmaps)]).astype(np.float16)
    out["fps_used"] = np.array(fps_used)
    return out


def clip_features(series: dict[str, np.ndarray]) -> dict[str, float]:
    """Clip summaries of the parallax series (mean, median, std, p10, p90, signed sum, max abs, first-half minus second-half) with prefix `px_`, three-segment means
    (`px_seg{0,1,2}_<key>`) for translation-vs-rotation evidence per segment, and the two translation-share ratios `px_vshare` (vertical) and `px_hshare` (horizontal)."""
    f: dict[str, float] = {}
    fps = float(series["fps_used"])
    for k in KEYS:
        x = series[k]
        half = len(x) // 2
        f.update({f"px_{k}_mean": float(x.mean()), f"px_{k}_median": float(np.median(x)), f"px_{k}_std": float(x.std()), f"px_{k}_p10": float(np.percentile(x, 10)),
                  f"px_{k}_p90": float(np.percentile(x, 90)), f"px_{k}_sum": float(x.sum() / fps), f"px_{k}_maxabs": float(np.abs(x).max()),
                  f"px_{k}_h1_minus_h2": float(x[:half].mean() - x[half:].mean()) if half >= 1 else 0.0})
    for s in range(3):
        for k in KEYS:
            x = series[k]
            seg = x[s * len(x) // 3:(s + 1) * len(x) // 3]
            f[f"px_seg{s}_{k}"] = float(seg.mean()) if len(seg) else 0.0
    f["px_vshare"] = float(abs(f["px_pty_sum"]) / (abs(f["px_pty_sum"]) + abs(f["px_ty_sum"]) + 1e-9))
    f["px_hshare"] = float(abs(f["px_ptx_sum"]) / (abs(f["px_ptx_sum"]) + abs(f["px_tx_sum"]) + 1e-9))
    return f


def extract_clip(path: str | Path, cache_dir: Path, key: str | None = None) -> Path:
    """Cached `clip_series` as `<cache_dir>/<clip name>.<key>.npz` (idempotent: an existing file for the current key is reused)."""
    key = key or clip_key(path)
    dst = Path(cache_dir) / f"{Path(path).name}.{key}.npz"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".tmp_" + dst.name)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **clip_series(path))
    tmp.replace(dst)
    return dst
