"""Fetch small pretrained weights used by Phase 2 into models/ and the HF cache.
  - OpenCV YuNet face detector (~230 KB) from github.com/opencv/opencv_zoo (MediaPipe crashes on this Mac)
  - YOLO11n (~5 MB) via ultralytics auto-download
  - Depth-Anything-V2-Small-hf (~100 MB) and AST AudioSet classifier (~350 MB) from Hugging Face"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "models").mkdir(exist_ok=True)

face = ROOT / "models/face_detection_yunet_2023mar.onnx"
if not face.exists():
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", face)
print("face:", face.stat().st_size)

import shutil

from ultralytics import YOLO

y = ROOT / "models/yolo11n.pt"
if not y.exists():
    YOLO("yolo11n.pt")  # downloads into cwd
    if Path("yolo11n.pt").exists():
        shutil.move("yolo11n.pt", y)
print("yolo:", y.exists())

from huggingface_hub import snapshot_download

for repo in ("depth-anything/Depth-Anything-V2-Small-hf", "MIT/ast-finetuned-audioset-10-10-0.4593"):
    print(repo, snapshot_download(repo_id=repo))
