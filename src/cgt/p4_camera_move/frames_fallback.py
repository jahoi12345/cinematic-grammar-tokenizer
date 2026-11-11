"""K frames of a test video for the video model with a SEQUENTIAL fallback (Amendment 14). `video_vlm.extract_frames` seeks to frame indices computed from the container's frame count; a few ShotBench
movement videos report a frame count that is larger than the number of decodable frames ("frame 58 unreadable"). The training clips never triggered this. The fallback decodes the video sequentially, counts the
frames that really decode and applies the SAME rule (`video_vlm.frame_indices` centres of K equal time bins over the true count), the same resize, JPEG quality and cache file names, so that a healthy video gives
byte-identical frames through either path. No label is involved."""
from pathlib import Path

import cv2
from PIL import Image

from cgt.p3b_vlm.images import MULTIPLE, budget_size
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.flow import file_digest


def extract_frames_robust(clip_path: str | Path, k: int, tokens_per_frame: int, cache_dir: Path) -> list[Path]:
    try:
        return V.extract_frames(clip_path, k, tokens_per_frame, cache_dir)
    except ValueError:
        pass
    clip_path = Path(clip_path)
    key = file_digest(clip_path)[:12]
    out = [Path(cache_dir) / f"{tokens_per_frame}t" / f"{clip_path.name}.{key}.{k}f.{i}.jpg" for i in range(k)]
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < k:
        raise ValueError(f"{clip_path}: only {len(frames)} decodable frames < {k}")
    for i, fi in enumerate(V.frame_indices(len(frames), k)):
        im = Image.fromarray(cv2.cvtColor(frames[fi], cv2.COLOR_BGR2RGB))
        im = im.resize(budget_size(*im.size, V.budget_px(tokens_per_frame), MULTIPLE), Image.BICUBIC)
        out[i].parent.mkdir(parents=True, exist_ok=True)
        im.save(out[i], quality=92)
    return out
