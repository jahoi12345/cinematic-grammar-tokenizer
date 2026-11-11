"""Classical flow-geometry features for all ShotQA movement clips (CPU only, cached, idempotent). Per clip: DIS flow at 12 fps / 384 px wide, global affine fit per
frame pair, summaries -> `data/cache/p4/features_shotqa.parquet` (gitignored). Cache files `data/cache/p4/flow/<clip>.<key>.npz` are keyed by clip name + the code
key of cgt.p4_camera_move.flow (source hash + parameters); a clip whose file for the current key exists is not recomputed. Never reads ShotBench or CameraBench.
Usage: uv run python scripts/p4_extract_flow.py [--an earlier change] [--limit N]"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move.labels import load_shotqa_movement_meta

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/p4"
QA = ROOT / "data/external/shotqa"


def one(clip: str) -> tuple[str, str | None]:
    cv2.setNumThreads(1)
    try:
        F.extract_clip(QA / clip, CACHE / "flow")
        return clip, None
    except Exception as e:  # noqa: BLE001 - reported, never silently dropped
        return clip, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = load_shotqa_movement_meta()
    clips = [r["clip"] for r in rows][: a.limit]
    key = F.cache_key()
    ckeys = {c: F.clip_key(QA / c) for c in clips}
    todo = [c for c in clips if not (CACHE / "flow" / f"{c}.{ckeys[c]}.npz").exists()]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} cached for code key {key} (+ content hash), {len(todo)} to compute", flush=True)
    t0, errs = time.time(), []
    with Pool(a.workers) as pool:
        for i, (c, e) in enumerate(pool.imap_unordered(one, todo, chunksize=4), 1):
            if e:
                errs.append((c, e))
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    if errs:
        raise SystemExit(f"{len(errs)} clips failed, first: {errs[:3]}")
    feats = []
    for c in sorted(clips):
        z = np.load(CACHE / "flow" / f"{c}.{ckeys[c]}.npz")
        feats.append({"clip": c} | F.clip_features({k: z[k] for k in z.files}))
    df = pd.DataFrame(feats)
    df.to_parquet(CACHE / "features_shotqa.parquet", index=False)
    (CACHE / "features_shotqa.meta.json").write_text(json.dumps({"cache_key": key, "n_clips": len(df), "n_features": df.shape[1] - 1, "params": F.PARAMS,
                                                                  "seconds_this_run": round(time.time() - t0, 1)}, indent=1))
    print(f"wrote {CACHE / 'features_shotqa.parquet'}: {df.shape}, nan share {df.isna().mean().mean():.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
