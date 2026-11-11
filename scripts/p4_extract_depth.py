"""Depth-parallax features for all ShotQA movement clips (CPU only, cached, idempotent, separate cache key from the flow features):
`data/cache/p4/depth/<clip>.<key>.npz` and `data/cache/p4/features_depth_shotqa.parquet` (gitignored). Never reads ShotBench or CameraBench.
Usage: uv run python scripts/p4_extract_depth.py [--an earlier change] [--threads 2] [--limit N]"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move.labels import load_shotqa_movement_meta

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/p4"
QA = ROOT / "data/external/shotqa"
THREADS = 2


def init() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(THREADS)


def one(clip: str) -> tuple[str, str | None]:
    try:
        D.extract_clip(QA / clip, CACHE / "depth")
        return clip, None
    except Exception as e:  # noqa: BLE001 - reported, never silently dropped
        return clip, f"{type(e).__name__}: {e}"


def main() -> None:
    global THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    THREADS = a.threads
    clips = [r["clip"] for r in load_shotqa_movement_meta()][: a.limit]
    key = D.cache_key()
    ckeys = {c: D.clip_key(QA / c) for c in clips}
    todo = [c for c in clips if not (CACHE / "depth" / f"{c}.{ckeys[c]}.npz").exists()]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} cached for code key {key} (+ content hash), {len(todo)} to compute", flush=True)
    t0, errs = time.time(), []
    with Pool(a.workers, initializer=init) as pool:
        for i, (c, e) in enumerate(pool.imap_unordered(one, todo, chunksize=1), 1):
            if e:
                errs.append((c, e))
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    if errs:
        raise SystemExit(f"{len(errs)} clips failed, first: {errs[:3]}")
    feats = []
    for c in sorted(clips):
        z = np.load(CACHE / "depth" / f"{c}.{ckeys[c]}.npz")
        feats.append({"clip": c} | D.clip_features({k: z[k] for k in z.files if k != "depth"}))
    df = pd.DataFrame(feats)
    df.to_parquet(CACHE / "features_depth_shotqa.parquet", index=False)
    (CACHE / "features_depth_shotqa.meta.json").write_text(json.dumps({"cache_key": key, "n_clips": len(df), "n_features": df.shape[1] - 1, "params": D.PARAMS,
                                                                        "seconds_this_run": round(time.time() - t0, 1)}, indent=1))
    print(f"wrote {df.shape}, nan share {df.isna().mean().mean():.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
