"""CPU-only cost test for the Phase 4 flow stage (GPU-free, 2 threads): decode + classical dense flow (Farneback, DIS) on ShotQA movement clips (training
side only; no ShotBench video is read). RAFT is NOT measured here (needs weights and the GPU). Writes results/phase4/flow_cost.json.
Usage: uv run python scripts/p4_flow_cost.py"""
import json
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
cv2.setNumThreads(2)


def main() -> None:
    clips = sorted(QA.glob("*.mp4"))[::106][:10]  # 10 clips spread over the sorted list
    out = {"n_clips": len(clips), "threads": 2, "sample_fps": 12, "per_clip": []}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for c in clips:
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(str(c))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 24.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / 12))
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(g, (384, 160), interpolation=cv2.INTER_AREA))
            i += 1
        cap.release()
        t_dec = time.perf_counter() - t0
        t0 = time.perf_counter()
        for a, b in pairwise(frames):
            cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        t_fb = time.perf_counter() - t0
        t0 = time.perf_counter()
        for a, b in pairwise(frames):
            dis.calc(a, b, None)
        t_dis = time.perf_counter() - t0
        out["per_clip"].append({"clip": c.name, "frames_decoded": n, "pairs": len(frames) - 1, "decode_s": round(t_dec, 2), "farneback_s": round(t_fb, 2), "dis_s": round(t_dis, 2)})
    pc = out["per_clip"]
    pairs = sum(p["pairs"] for p in pc)
    out["mean_decode_s_per_clip"] = round(float(np.mean([p["decode_s"] for p in pc])), 2)
    out["farneback_ms_per_pair_384x160"] = round(1000 * sum(p["farneback_s"] for p in pc) / pairs, 1)
    out["dis_ms_per_pair_384x160"] = round(1000 * sum(p["dis_s"] for p in pc) / pairs, 1)
    out["mean_pairs_per_clip"] = round(pairs / len(pc), 1)
    out["projected_1058_clips_hours_decode_plus_dis"] = round(1058 * (out["mean_decode_s_per_clip"] + out["mean_pairs_per_clip"] * out["dis_ms_per_pair_384x160"] / 1000) / 3600, 2)
    out["projected_1058_clips_hours_decode_plus_farneback"] = round(1058 * (out["mean_decode_s_per_clip"] + out["mean_pairs_per_clip"] * out["farneback_ms_per_pair_384x160"] / 1000) / 3600, 2)
    (ROOT / "results/phase4/flow_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()
