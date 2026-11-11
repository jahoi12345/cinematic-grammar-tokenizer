"""Shot boundary detection with PySceneDetect's ContentDetector."""
import cv2
from scenedetect import ContentDetector, detect


def video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    return n / fps if fps else 0.0


def detect_shots(video_path: str, threshold: float = 27.0) -> list[tuple[float, float]]:
    """Return contiguous [(start_s, end_s), ...]; first start 0.0, last end = duration."""
    scenes = detect(str(video_path), ContentDetector(threshold=threshold), start_in_scene=True)
    if not scenes:
        return [(0.0, video_duration(video_path))]
    shots = [(s.seconds, e.seconds) for s, e in scenes]
    shots[0] = (0.0, shots[0][1])
    return shots
