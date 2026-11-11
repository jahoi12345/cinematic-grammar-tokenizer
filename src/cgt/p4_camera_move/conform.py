"""Label-free INPUT CONFORMITY of the test clips, version `fps-conform-v1` (docs/phase4_video_plan.md Amendment 12 item 4, born from Amendment 11). The training features carry `fps_used` = 11.988 with a
standard deviation of 0.003 (every training clip is a 23.976 fps clip sampled every second frame), so a test clip with another frame rate would get a standardized `fps_used` of hundreds and saturate the
per-atom logistic regressions. Before ANY flow or depth feature extraction a test video or GIF is resampled IN TIME to 23.976 fps (output frame j = the source frame nearest to j / 23.976 s for videos; the
frame displayed at that instant, from the cumulative frame durations, for GIFs) and written as a lossless temporary clip; the UNCHANGED flow and depth code then samples every second frame (11.988 fps).
No label is involved. The version and the hash of this file are recorded in the preparation manifest and refused when absent or different."""
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import flow as F

CONFORM_VERSION = "fps-conform-v1"
TARGET_FPS = 24000 / 1001  # 23.976
GIF_DEFAULT_MS = 100  # a GIF frame with duration 0 is displayed for 100 ms (browser convention)


def code_hash() -> str:
    """First 16 hex of the SHA-256 of this file: any change of the conformity code changes the provenance that the evaluator checks."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def provenance() -> dict:
    """What a preparation manifest must carry and the evaluator re-derives: conformity version and code hash, target rate, and the cache keys of the unchanged flow and depth extractors."""
    return {"conform_version": CONFORM_VERSION, "conform_code_sha256": code_hash(), "target_fps": TARGET_FPS, "flow_key": F.cache_key(), "depth_key": D.cache_key()}


def _writer(dst: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"FFV1"), TARGET_FPS, size)
    if not wr.isOpened():
        raise RuntimeError("OpenCV cannot write FFV1")
    return wr


def video_source_indices(n_frames: int, fps: float) -> list[int]:
    """Source frame index of every output frame: nearest to j / TARGET_FPS seconds, for j < floor(duration x TARGET_FPS)."""
    n_out = int(n_frames / fps * TARGET_FPS)
    return [min(n_frames - 1, round(j / TARGET_FPS * fps)) for j in range(n_out)]


def conform_video(src: Path, dst: Path) -> int:
    """Write the conformed lossless clip of the video `src`; returns the number of output frames (ValueError if the video cannot be decoded or has fewer than 3 output frames)."""
    cap = cv2.VideoCapture(str(src))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 24.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = video_source_indices(n, fps)
    if len(idx) < 3:
        raise ValueError(f"{src}: fewer than 3 conformed frames")
    wr, pos, last, written = None, 0, None, 0
    for i in idx:
        while pos <= i:
            ok, fr = cap.read()
            if not ok:
                break
            last, pos = fr, pos + 1
        if last is None:
            raise ValueError(f"{src}: undecodable")
        if wr is None:
            wr = _writer(dst, (last.shape[1], last.shape[0]))
        wr.write(last)
        written += 1
    wr.release()
    cap.release()
    return written


def gif_source_indices(durations_ms: list[float]) -> list[int]:
    """Source frame displayed at each output instant j / TARGET_FPS, from the cumulative durations (variable durations are handled by timestamp)."""
    d = np.array([x if x > 0 else GIF_DEFAULT_MS for x in durations_ms], dtype=np.float64) / 1000.0
    starts, total = np.concatenate([[0.0], np.cumsum(d)[:-1]]), float(d.sum())
    n_out = int(total * TARGET_FPS)
    return [int(np.searchsorted(starts, j / TARGET_FPS, side="right") - 1) for j in range(n_out)]


def conform_gif(src: Path, dst: Path) -> int:
    """Write the conformed lossless clip of the GIF `src`; returns the number of output frames."""
    frames, durs = [], []
    for fr in ImageSequence.Iterator(Image.open(src)):
        durs.append(float(fr.info.get("duration", 0)))
        frames.append(cv2.cvtColor(np.asarray(fr.convert("RGB")), cv2.COLOR_RGB2BGR))
    idx = gif_source_indices(durs)
    if len(idx) < 3:
        raise ValueError(f"{src}: fewer than 3 conformed frames")
    wr = _writer(dst, (frames[0].shape[1], frames[0].shape[0]))
    for i in idx:
        wr.write(frames[i])
    wr.release()
    return len(idx)


def conformed_series(path: Path, kind: str) -> tuple[dict, dict]:
    """(flow series, depth-parallax series) of the conformed clip of `path` (kind 'video' or 'gif'): the unchanged `flow.clip_series` and `depth_parallax.clip_series` on the lossless temporary clip."""
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "conformed.avi"
        {"video": conform_video, "gif": conform_gif}[kind](Path(path), clip)
        return F.clip_series(clip), D.clip_series(clip)
