"""LABEL-FREE preparation of the inputs of the sealed final evaluations (docs/phase4_interface.md section 10.6): for ShotBench movement the 464 videos are extracted from `videos.tar` (only the
members named by the question file), for CameraBench the GIFs are listed; for both, the flow and depth-parallax features (the SAME code and column order as the training features), the table
`data/cache/p4_eval/features_<dataset>.parquet`, the K=8 frames of the video model, and a manifest with the SHA-256 of every input file and of the feature table
(`results/phase4/final_eval/prepare_<dataset>.json`). It reads NO label and NO answer (the ShotBench question file is read with the answer column excluded; test.jsonl is never opened), writes no
ledger entry and needs no go file; it is separate from the evaluator so that hours of CPU work never sit inside a sealed run. CPU only; idempotent (cached by clip name + content hash).
Usage: uv run python scripts/p4_final_prepare.py --dataset shotbench_movement [--an earlier change] [--limit N]
Library use (tests): `prepare_paths(paths, kind, out_parquet, ...)` on any list of video or GIF files."""
import argparse
import hashlib
import json
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from cgt.p4_camera_move import conform
from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_inputs as FI
from cgt.p4_camera_move import final_stores as FS
from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move import gif as G
from cgt.p4_camera_move.camerabench import CB_ROOT
from cgt.p4_camera_move.frames_fallback import extract_frames_robust

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/p4"
EVAL_CACHE = ROOT / "data/cache/p4_eval"
FRAMES = CACHE / "video_frames"
THREADS = 2


def _init() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(THREADS)


def _series_cached(path: Path, kind: str, cache: Path) -> tuple[dict, dict]:
    """The conformed flow and depth series of `path` (fps-conform-v1: resampled in time to the training rate BEFORE the unchanged extractors), cached by source bytes + conformity code + extractor keys."""
    key = f"{FI.sha256_file(Path(path))[:12]}.{conform.code_hash()}.{F.cache_key()}.{D.cache_key()}"
    fp, dp = Path(cache) / "flow_conf" / f"{Path(path).name}.{key}.npz", Path(cache) / "depth_conf" / f"{Path(path).name}.{key}.npz"
    if not (fp.exists() and dp.exists()):
        sf, sd = conform.conformed_series(Path(path), kind)
        for dst, ser in ((fp, sf), (dp, sd)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(".tmp_" + dst.name)
            with open(tmp, "wb") as fh:
                np.savez(fh, **ser)
            tmp.replace(dst)
    zf, zd = np.load(fp), np.load(dp)
    return {k: zf[k] for k in zf.files}, {k: zd[k] for k in zd.files}


def _one(args) -> tuple[str, dict | None, str | None]:
    path, kind, cache = args
    try:
        sf, sd = _series_cached(Path(path), kind, Path(cache))
        frames = (extract_frames_robust if kind == "video" else G.gif_frames_k)(path, 8, 64, FRAMES)  # the video model's K = 8 frames are unchanged by the conformity
        row = {"clip": Path(path).name} | F.clip_features(sf) | D.clip_features({k: v for k, v in sd.items() if k != "depth"})
        return str(path), row | {"_frames": len(frames)}, None
    except Exception as e:  # noqa: BLE001 - reported, never silently dropped
        return str(path), None, f"{type(e).__name__}: {e}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_paths(paths: list[Path], kind: str, out_parquet: Path, *, dataset: str = "test", cache: Path = CACHE, workers: int = 3) -> dict:
    """Features of every file in `paths` (kind 'video' or 'gif') -> `out_parquet` (index column `clip` = file name, columns = the training feature columns in the training order)."""
    t0, rows, errs = time.time(), [], []
    with Pool(workers, initializer=_init) as pool:
        for i, (p, row, err) in enumerate(pool.imap_unordered(_one, [(str(p), kind, str(cache)) for p in paths], chunksize=1), 1):
            if err:
                errs.append((p, err))
            else:
                rows.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(paths)} done, {time.time() - t0:.0f}s", flush=True)
    if errs:
        raise SystemExit(f"{len(errs)} files failed, first: {errs[:3]}")
    df = pd.DataFrame(rows).drop(columns="_frames").sort_values("clip").reset_index(drop=True)
    train_cols = [c for f in ("features_shotqa.parquet", "features_depth_shotqa.parquet") for c in pd.read_parquet(CACHE / f).columns if c != "clip"]
    if list(df.columns[1:]) != train_cols:
        raise SystemExit("the test feature columns differ from the training feature columns")
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    short = sorted(Path(p).name for p in paths if kind == "gif" and G.gif_props(p)["n_frames"] < 8)  # GIFs with fewer frames than K (their frames repeat; Amendment 15)
    return {"manifest": FI.build_manifest(dataset, paths, Path(out_parquet), train_cols, gifs_with_fewer_than_8_frames=short, nan_share=round(float(df.iloc[:, 1:].isna().mean().mean()), 5), seconds=round(time.time() - t0, 1)), "n_features": len(train_cols)}


def extract_shotbench_videos(names: list[str], dest: Path) -> list[Path]:
    """Extract exactly the named members `video/<name>` of videos.tar into `dest` (existing files are kept)."""
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with tarfile.open(FS.SB / "videos.tar") as t:
        for n in names:
            target = dest / n
            if not target.exists():
                member = t.getmember(f"video/{n}")
                target.write_bytes(t.extractfile(member).read())
            out.append(target)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=FE.DATASETS, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    THREADS = a.threads
    if a.dataset == "shotbench_movement":
        names = [q["clip"] for q in FS.ShotBenchMovementStore().questions()][: a.limit]  # label-free: the answer column is excluded at the parser
        paths, kind = extract_shotbench_videos(names, EVAL_CACHE / "shotbench_movement_video"), "video"
    else:
        paths, kind = [CB_ROOT / "videos_gif" / n for n in FS.CameraBenchFinalStore().clips()][: a.limit], "gif"  # names only
    info = prepare_paths(paths, kind, EVAL_CACHE / f"features_{a.dataset}.parquet", dataset=a.dataset, workers=a.workers)
    out = ROOT / "results/phase4/final_eval" / f"prepare_{a.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**info["manifest"], "limit": a.limit}, indent=1))
    print(f"prepared {len(paths)} {kind} files: {info['n_features']} features, nan share {info['manifest']['nan_share']}, {info['manifest']['seconds']} s -> {out}")
