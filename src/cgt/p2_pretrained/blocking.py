"""People, faces, framing on the thirds grid, facing and light direction (YOLO + OpenCV YuNet).
MediaPipe face mesh was tried first but crashes at startup here (Metal service unavailable)."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
_yolo = None
_face = None


def _get_yolo():
    global _yolo
    if _yolo is None:
        w = ROOT / "models/yolo11n.pt"
        _yolo = YOLO(str(w) if w.exists() else "yolo11n.pt")
    return _yolo


def _get_face(w: int, h: int):
    global _face
    if _face is None:
        _face = cv2.FaceDetectorYN.create(str(ROOT / "models/face_detection_yunet_2023mar.onnx"), "", (w, h),
                                          score_threshold=0.6)
    _face.setInputSize((w, h))
    return _face


def detect_people(bgr: np.ndarray, conf: float = 0.35) -> list[tuple[float, float, float, float]]:
    """Person boxes (x1, y1, x2, y2) in pixels, largest first."""
    r = _get_yolo()(bgr, classes=[0], conf=conf, verbose=False)[0]
    boxes = [tuple(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def detect_faces(bgr: np.ndarray) -> list[dict]:
    """Faces as dicts: box (x1,y1,x2,y2) and landmark x's: eye_l, eye_r (image left/right), nose."""
    h, w = bgr.shape[:2]
    _, det = _get_face(w, h).detect(bgr)
    faces = []
    for r in (det if det is not None else []):
        x, y, bw, bh = r[:4]
        faces.append({"box": (float(x), float(y), float(x + bw), float(y + bh)),
                      "eye_l": float(r[4]), "eye_r": float(r[6]), "nose": float(r[8])})
    return faces


def facing(face: dict, profile_offset: float = 0.25) -> str:
    """Yaw proxy: nose position between the two eyes (0.5 = frontal)."""
    span = face["eye_r"] - face["eye_l"]
    if span < 1:
        return "profile"
    r = (face["nose"] - face["eye_l"]) / span
    return "profile" if abs(r - 0.5) > profile_offset else "camera"


def _area(f: dict) -> float:
    x1, y1, x2, y2 = f["box"]
    return (x2 - x1) * (y2 - y1)


def light_direction(gray: np.ndarray, face: dict, asym_thr: float = 0.25, back_thr: float = 0.6) -> dict:
    """front / side / back from left-vs-right face brightness and face-vs-background brightness."""
    x1, y1, x2, y2 = (max(0, int(v)) for v in face["box"])
    crop = gray[y1:y2, x1:x2].astype(np.float32)
    if crop.size < 16 or crop.shape[1] < 4:
        return {"light_dir": None, "light_asym": None, "face_bg_ratio": None}
    mid = crop.shape[1] // 2
    lum_l, lum_r = crop[:, :mid].mean(), crop[:, mid:].mean()
    asym = float((lum_r - lum_l) / (crop.mean() + 1e-6))
    mask = np.ones(gray.shape, bool)
    mask[y1:y2, x1:x2] = False
    ratio = float(crop.mean() / (gray[mask].mean() + 1e-6)) if mask.any() else 1.0
    if ratio < back_thr:
        d = "back"
    elif abs(asym) > asym_thr:
        d = "side"
    else:
        d = "front"
    return {"light_dir": d, "light_asym": asym, "face_bg_ratio": ratio}


def blocking_features(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    people = detect_people(bgr)
    faces = detect_faces(bgr)
    out = {"people": len(people), "faces": len(faces), "subject_col": None, "subject_row": None,
           "headroom": None, "facing": None, "light_dir": None, "light_asym": None, "face_bg_ratio": None}
    if not people and not faces:
        return out
    if people:
        x1, y1, x2, y2 = people[0]
        out["headroom"] = float(y1 / h)
    else:  # face only (tight close-up): use the largest face box as the subject
        x1, y1, x2, y2 = max(faces, key=_area)["box"]
        out["headroom"] = float(y1 / h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    out["subject_col"], out["subject_row"] = int(min(2, cx / w * 3)), int(min(2, cy / h * 3))
    inside = [f for f in faces if x1 <= (f["box"][0] + f["box"][2]) / 2 <= x2 and y1 <= (f["box"][1] + f["box"][3]) / 2 <= y2]
    if inside:
        main = max(inside, key=_area)
        out["facing"] = facing(main)
        out.update(light_direction(gray, main))
    elif people:
        out["facing"] = "away"
    return out
