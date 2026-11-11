"""Frozen SigLIP-base embeddings of 8 uniformly sampled frames for every ShotQA movement clip (cached, idempotent), then pooled features
`data/cache/p4/features_siglip_shotqa.parquet` (gitignored). Device is chosen by --device (default cpu: another worker may hold the GPU). Never reads ShotBench or CameraBench.
Usage: uv run python scripts/p4_extract_siglip.py [--device cpu] [--threads 2] [--limit N]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cgt.p4_camera_move import frame_embed as E
from cgt.p4_camera_move.labels import load_shotqa_movement_meta

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/p4"
QA = ROOT / "data/external/shotqa"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    clips = [r["clip"] for r in load_shotqa_movement_meta()][: a.limit]
    key = E.cache_key()
    todo = [c for c in clips if not (CACHE / "siglip" / f"{c}.{key}.npy").exists()]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} cached for key {key}, {len(todo)} to compute on {a.device}", flush=True)
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        E.extract_clip(QA / c, CACHE / "siglip", key, a.device)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    feats = [{"clip": c} | {f"sg_{j}": float(v) for j, v in enumerate(E.pooled(np.load(CACHE / "siglip" / f"{c}.{key}.npy")))} for c in sorted(clips)]
    df = pd.DataFrame(feats)
    df.to_parquet(CACHE / "features_siglip_shotqa.parquet", index=False)
    (CACHE / "features_siglip_shotqa.meta.json").write_text(json.dumps({"cache_key": key, "n_clips": len(df), "n_features": df.shape[1] - 1, "params": E.PARAMS,
                                                                         "device": a.device, "seconds_this_run": round(time.time() - t0, 1)}, indent=1))
    print(f"wrote {df.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
