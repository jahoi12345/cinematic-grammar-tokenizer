"""Classical flow geometry for camera movement (docs/phase4_interface.md section 3). CPU only (OpenCV DIS flow, numpy). Never reads ShotBench or CameraBench.
Units: flow in half-image-widths per frame pair ("normalised"), coordinates x, y in [-1, 1] (y uses the same scale as x, centred), per-second values are
per-pair values times the sampling fps. A camera pan to the RIGHT moves the scene to the LEFT in the image, so tx < 0 for pan-right / move-right."""
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

FPS, WIDTH, STEP = 12.0, 384, 8
SERIES_KEYS = ["tx", "ty", "div", "rot", "shear1", "shear2", "r2", "resid_rms", "flow_rms", "res_top_bottom", "res_left_right", "div_spread", "tx_spread"]
PER_SECOND = {"tx", "ty", "div", "rot", "shear1", "shear2", "resid_rms", "flow_rms", "res_top_bottom", "res_left_right", "div_spread", "tx_spread"}
PARAMS = {"fps": FPS, "width": WIDTH, "step": STEP, "dis_preset": "MEDIUM", "huber_k": 1.345, "irls_iters": 5}


def cache_key() -> str:
    """First 16 hex of sha256(source of this module + parameters): any code or parameter change invalidates every cached clip."""
    return hashlib.sha256(Path(__file__).read_bytes() + json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:16]


def file_digest(path: str | Path) -> str:
    """SHA-256 of the video file bytes (advisory of the Phase 4 build review: a replaced same-name file must invalidate its cache entry)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_key(path: str | Path) -> str:
    """Per-clip cache key = code key (module source + parameters) + first 12 hex of the clip's content SHA-256."""
    return f"{cache_key()}.{file_digest(path)[:12]}"


def sample_gray(path: str | Path, fps: float = FPS, width: int = WIDTH) -> tuple[list[np.ndarray], float]:
    """Grey frames every round(src_fps / fps)-th frame, resized to `width` px wide with the aspect ratio kept (even height). Returns (frames, fps_used)."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(src / fps))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            h = round(g.shape[0] * width / g.shape[1] / 2) * 2
            frames.append(cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    if len(frames) < 3:
        raise ValueError(f"{path}: only {len(frames)} frames decoded")
    return frames, src / step


def dis_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """DIS dense flow (preset MEDIUM) from a to b in pixels, float32 (h, w, 2)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(a, b, None).astype(np.float32)


def fit_affine(flow: np.ndarray, step: int = STEP, iters: int = 5) -> dict:
    """Robust (Huber IRLS) affine fit u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y of a pixel flow field; returns the normalised parameters and the
    residual/parallax statistics documented in docs/phase4_interface.md section 3."""
    h, w = flow.shape[:2]
    half = w / 2.0
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    xn, yn = ((xs - w / 2.0) / half).ravel(), ((ys - h / 2.0) / half).ravel()
    uv = flow[ys.ravel(), xs.ravel()] / half  # (n, 2) normalised flow
    A = np.stack([np.ones_like(xn), xn, yn], axis=1)
    wt = np.ones(len(xn))
    coef = np.zeros((3, 2))
    for _ in range(iters):
        sw = np.sqrt(wt)[:, None]
        coef, *_ = np.linalg.lstsq(A * sw, uv * sw, rcond=None)
        res = uv - A @ coef
        mag = np.linalg.norm(res, axis=1)
        delta = 1.345 * 1.4826 * (np.median(mag) + 1e-9)
        wt = np.minimum(1.0, delta / (mag + 1e-12))
    res = uv - A @ coef
    (a0, b0), (a1, b1), (a2, b2) = coef
    tot = float((uv ** 2).sum()) + 1e-12
    rmag = np.linalg.norm(res, axis=1).reshape(ys.shape)
    gh, gw = rmag.shape
    third_h, third_w = max(1, gh // 3), max(1, gw // 3)
    # local divergence / translation on a 3 x 5 block grid of the dense flow (parallax proxies: depth-dependent motion varies over the frame)
    fn = flow / half
    du_dx, dv_dy = np.gradient(fn[..., 0], axis=1), np.gradient(fn[..., 1], axis=0)
    ldiv = (du_dx + dv_dy) / 2.0
    bl_div, bl_tx = [], []
    for bi in range(3):
        for bj in range(5):
            sl = (slice(bi * h // 3, (bi + 1) * h // 3), slice(bj * w // 5, (bj + 1) * w // 5))
            bl_div.append(float(ldiv[sl].mean()))
            bl_tx.append(float(fn[..., 0][sl].mean()))
    return {"tx": float(a0), "ty": float(b0), "div": float((a1 + b2) / 2), "rot": float((b1 - a2) / 2), "shear1": float((a1 - b2) / 2), "shear2": float((a2 + b1) / 2),
            "r2": float(1.0 - (res ** 2).sum() / tot), "resid_rms": float(np.sqrt((res ** 2).sum(1).mean())), "flow_rms": float(np.sqrt((uv ** 2).sum(1).mean())),
            "res_top_bottom": float(rmag[:third_h].mean() - rmag[-third_h:].mean()), "res_left_right": float(rmag[:, :third_w].mean() - rmag[:, -third_w:].mean()),
            "div_spread": float(np.std(bl_div)), "tx_spread": float(np.std(bl_tx))}


def clip_series(path: str | Path) -> dict[str, np.ndarray]:
    """Per-frame-pair series of every key in SERIES_KEYS (per-second units for the keys in PER_SECOND) plus scalar entries fps_used and duration_s."""
    frames, fps_used = sample_gray(path)
    rows = [fit_affine(dis_flow(a, b)) for a, b in pairwise(frames)]
    out = {k: np.array([r[k] for r in rows], dtype=np.float64) * (fps_used if k in PER_SECOND else 1.0) for k in SERIES_KEYS}
    out["fps_used"] = np.array(fps_used)
    out["duration_s"] = np.array(len(frames) / fps_used)
    return out


def _hp(x: np.ndarray) -> np.ndarray:
    return x - uniform_filter1d(x, size=5, mode="nearest")


def clip_features(series: dict[str, np.ndarray]) -> dict[str, float]:
    """One feature dict per clip: summaries of every series (mean, median, std, p10, p90, signed sum, max abs, first-half minus second-half mean) and
    jitter stats (handheld cues) of tx, ty and rot, plus n_pairs, fps_used, duration_s."""
    f: dict[str, float] = {}
    fps = float(series["fps_used"])
    for k in SERIES_KEYS:
        x = series[k]
        half = len(x) // 2
        f.update({f"{k}_mean": float(x.mean()), f"{k}_median": float(np.median(x)), f"{k}_std": float(x.std()), f"{k}_p10": float(np.percentile(x, 10)),
                  f"{k}_p90": float(np.percentile(x, 90)), f"{k}_sum": float(x.sum() / fps), f"{k}_maxabs": float(np.abs(x).max()),
                  f"{k}_h1_minus_h2": float(x[:half].mean() - x[half:].mean()) if half >= 1 else 0.0})
    for k in ("tx", "ty", "rot"):
        x = series[k]
        hp = _hp(x)
        if len(x) >= 8:
            p = np.abs(np.fft.rfft(x - x.mean())) ** 2
            fr = np.fft.rfftfreq(len(x), d=1.0 / fps)
            hi = float(p[fr > 2.0].sum() / (p[1:].sum() + 1e-12))
        else:
            hi = float("nan")
        f.update({f"{k}_jitter_std": float(hp.std()), f"{k}_jitter_hf_share": hi,
                  f"{k}_jitter_zcr": float((np.sign(hp[1:]) != np.sign(hp[:-1])).mean()) if len(hp) > 1 else 0.0,
                  f"{k}_diff_std": float(np.diff(x).std()) if len(x) > 1 else 0.0})
    f.update({"n_pairs": float(len(series["tx"])), "fps_used": fps, "duration_s": float(series["duration_s"])})
    return f


def extract_clip(path: str | Path, cache_dir: Path, key: str | None = None) -> Path:
    """Cached `clip_series` as `<cache_dir>/<clip name>.<code key>.<content hash>.npz`; a clip whose file for the current key exists is not recomputed (idempotent);
    a changed code or a replaced video file gives another key."""
    key = key or clip_key(path)
    dst = Path(cache_dir) / f"{Path(path).name}.{key}.npz"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".tmp_" + dst.name)
    with open(tmp, "wb") as fh:
        np.savez(fh, **clip_series(path))
    tmp.replace(dst)
    return dst
