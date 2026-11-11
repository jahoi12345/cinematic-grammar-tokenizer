"""CPU timing of Depth-Anything-V2-Small (HF cache from Phase 2) on a 50-clip subset of the ShotQA movement clips (training-side clips only; no ShotBench or CameraBench
data). K frames per clip, two input sizes. Writes results/phase4/depth_timing.json. Usage: uv run python scripts/p4_depth_timing.py [--clips 50] [--k 6] [--threads 4]"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def frames_at(path: Path, k: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, n - 1, k + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def prep(bgr: np.ndarray, width: int) -> torch.Tensor:
    h_in, w_in = bgr.shape[:2]
    w = round(width / 14) * 14
    h = max(14, round(w * h_in / w_in / 14) * 14)
    x = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=50)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()
    clips = sorted(QA.glob("*.mp4"))[:: max(1, 1058 // a.clips)][: a.clips]
    res = {"clips": len(clips), "k_frames_per_clip": a.k, "threads": a.threads, "device": "cpu", "torch": torch.__version__, "sizes": {}}
    allf = [frames_at(c, a.k) for c in clips]
    for width in (392, 518):
        with torch.no_grad():
            model(prep(allf[0][0], width))  # warm-up
            t0 = time.perf_counter()
            n = 0
            for fl in allf:
                for fr in fl:
                    d = model(prep(fr, width)).predicted_depth
                    n += 1
            tot = time.perf_counter() - t0
        res["sizes"][str(width)] = {"input_hw": list(prep(allf[0][0], width).shape[2:]), "frames": n, "seconds_per_frame": round(tot / n, 3), "seconds_per_clip": round(tot / len(clips), 2),
                                     "projected_hours_all_1058_clips": round(1058 * tot / len(clips) / 3600, 2), "output_shape": list(d.shape)}
        print(width, res["sizes"][str(width)], flush=True)
    (ROOT / "results/phase4/depth_timing.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
