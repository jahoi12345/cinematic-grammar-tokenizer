"""GIF input for the flow-geometry features (CameraBench feasibility sample, docs/phase4_interface.md section 3). Same DIS + affine pipeline as `flow.clip_series`;
GIF frame timing comes from the per-frame durations (ms) stored in the file. CPU only."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from cgt.p4_camera_move import flow as F


def gif_props(path: str | Path) -> dict:
    """Container facts: size, frame count, per-frame durations (ms) summary, effective fps (1000 / median duration), colour mode and palette size of frame 0."""
    im = Image.open(path)
    durs = []
    n = 0
    for fr in ImageSequence.Iterator(im):
        durs.append(float(fr.info.get("duration", 0)))
        n += 1
    im.seek(0)
    pal = im.getpalette()
    med = float(np.median(durs)) if durs else 0.0
    return {"w": im.size[0], "h": im.size[1], "n_frames": n, "duration_s": float(sum(durs) / 1000.0), "frame_ms_median": med, "frame_ms_min": float(min(durs)),
            "frame_ms_max": float(max(durs)), "fps": 1000.0 / med if med else float("nan"), "mode": im.mode, "palette_colors": (len(pal) // 3 if pal else None),
            "bytes": Path(path).stat().st_size}


def gif_gray(path: str | Path, fps: float = F.FPS, width: int = F.WIDTH) -> tuple[list[np.ndarray], float]:
    """Grey frames sampled to about `fps` (every round(src_fps / fps)-th frame), resized to `width` px wide with the aspect kept. Returns (frames, fps_used)."""
    p = gif_props(path)
    src = p["fps"] if np.isfinite(p["fps"]) and p["fps"] > 0 else 12.0
    step = max(1, round(src / fps))
    frames = []
    for i, fr in enumerate(ImageSequence.Iterator(Image.open(path))):
        if i % step == 0:
            g = np.asarray(fr.convert("L"))
            h = round(g.shape[0] * width / g.shape[1] / 2) * 2
            frames.append(cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA))
    if len(frames) < 3:
        raise ValueError(f"{path}: only {len(frames)} frames")
    return frames, src / step


def gif_series(path: str | Path) -> dict[str, np.ndarray]:
    """Same output as `flow.clip_series` for a GIF."""
    from itertools import pairwise
    frames, fps_used = gif_gray(path)
    rows = [F.fit_affine(F.dis_flow(a, b)) for a, b in pairwise(frames)]
    out = {k: np.array([r[k] for r in rows], dtype=np.float64) * (fps_used if k in F.PER_SECOND else 1.0) for k in F.SERIES_KEYS}
    out["fps_used"] = np.array(fps_used)
    out["duration_s"] = np.array(len(frames) / fps_used)
    return out


# ---- inputs of the sealed final evaluation (docs/phase4_interface.md section 10): label-free GIF readers for the video model and for the depth features ----
def cache_key() -> str:
    """First 16 hex of sha256(source of this module + the flow and depth-parallax keys): any code change invalidates every cached GIF series."""
    import hashlib

    from cgt.p4_camera_move import depth_parallax as D
    return hashlib.sha256(Path(__file__).read_bytes() + F.cache_key().encode() + D.cache_key().encode()).hexdigest()[:16]


def gif_colour_gray(path: str | Path, fps: float = F.FPS, width: int = F.WIDTH) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """(BGR frames at `width` px wide, grey frames, fps_used) sampled like `gif_gray` (every round(src_fps / fps)-th frame, per-frame durations from the file) and prepared like the
    video reader of `depth_parallax._frames` (grey = grey of the resized colour frame), so that the depth-parallax code runs on a GIF unchanged."""
    p = gif_props(path)
    src = p["fps"] if np.isfinite(p["fps"]) and p["fps"] > 0 else 12.0
    step = max(1, round(src / fps))
    col, gray = [], []
    for i, fr in enumerate(ImageSequence.Iterator(Image.open(path))):
        if i % step == 0:
            bgr = cv2.cvtColor(np.asarray(fr.convert("RGB")), cv2.COLOR_RGB2BGR)
            h = round(bgr.shape[0] * width / bgr.shape[1] / 2) * 2
            c = cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA)
            col.append(c)
            gray.append(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
    if len(col) < 3:
        raise ValueError(f"{path}: only {len(col)} frames")
    return col, gray, src / step


def gif_depth_series(path: str | Path) -> dict[str, np.ndarray]:
    """`depth_parallax.clip_series` for a GIF: the SAME code (depth maps, DIS flow, parallax fits) runs on the GIF frames of `gif_colour_gray` (its video reader is replaced for the call)."""
    from unittest import mock

    from cgt.p4_camera_move import depth_parallax as D
    col, gray, fps_used = gif_colour_gray(path)
    with mock.patch.object(D, "_frames", lambda _p: (col, gray, fps_used)):
        return D.clip_series(path)


def _cached(path: str | Path, cache_dir: Path, compute) -> Path:
    key = f"{cache_key()}.{F.file_digest(path)[:12]}"
    dst = Path(cache_dir) / f"{Path(path).name}.{key}.npz"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".tmp_" + dst.name)
    with open(tmp, "wb") as fh:
        np.savez(fh, **compute(path))
    tmp.replace(dst)
    return dst


def extract_gif_flow(path: str | Path, cache_dir: Path) -> Path:
    """Cached `gif_series` (flow features' series) as `<cache_dir>/<gif name>.<gif key>.<content hash>.npz` (idempotent)."""
    return _cached(path, cache_dir, gif_series)


def extract_gif_depth(path: str | Path, cache_dir: Path) -> Path:
    """Cached `gif_depth_series` as `<cache_dir>/<gif name>.<gif key>.<content hash>.npz` (idempotent)."""
    return _cached(path, cache_dir, gif_depth_series)


def gif_frames_k(path: str | Path, k: int, tokens_per_frame: int, cache_dir: Path) -> list[Path]:
    """K frames of a GIF for the video model: frame indices at the centres of k equal time bins (`video_vlm.frame_indices` over the GIF's frame indices), RGB, resized to the per-frame token
    budget exactly like `video_vlm.extract_frames`, saved as JPEG q92 under `<cache_dir>/<tokens>t/<gif name>.<content hash 12>.<k>f.<i>.jpg` (idempotent). A GIF with fewer than k frames repeats frames by the same rule (Amendment 15); ValueError only for a GIF with no frame."""
    from cgt.p3b_vlm.images import MULTIPLE, budget_size
    from cgt.p4_camera_move import video_vlm as V
    path = Path(path)
    key = F.file_digest(path)[:12]
    out = [Path(cache_dir) / f"{tokens_per_frame}t" / f"{path.name}.{key}.{k}f.{i}.jpg" for i in range(k)]
    if all(p.exists() for p in out):
        return out
    frames = [fr.convert("RGB") for fr in ImageSequence.Iterator(Image.open(path))]
    if not frames:
        raise ValueError(f"{path}: no decodable frame")
    # fewer frames than K (Amendment 15): the same bin-centre rule over the true count repeats frames (nearest in time); the clip is kept, never dropped
    for i, fi in enumerate(V.frame_indices(len(frames), k)):
        im = frames[fi].resize(budget_size(*frames[fi].size, V.budget_px(tokens_per_frame), MULTIPLE), Image.BICUBIC)
        out[i].parent.mkdir(parents=True, exist_ok=True)
        im.save(out[i], quality=92)
    return out
