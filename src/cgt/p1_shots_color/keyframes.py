"""Keyframe times and extraction."""
import cv2
import numpy as np


def pick_keyframe_times(start_s: float, end_s: float) -> tuple[float, float, float]:
    """Start, middle, end times strictly inside [start_s, end_s), avoiding the boundary frames."""
    d = end_s - start_s
    margin = min(0.1 * d, 0.25)
    return (start_s + margin, start_s + 0.5 * d, end_s - max(margin, 0.05 * d))


def read_frames(video_path: str, times: list[float]) -> list[np.ndarray | None]:
    """Decode the frame at each time (seconds); None when a frame cannot be read."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            out.append(frame if ok else None)
    finally:
        cap.release()
    return out
