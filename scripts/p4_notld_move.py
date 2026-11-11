"""NOTLD camera-movement slot, INFERENCE ONLY (docs/phase4_video_plan.md Amendment 10): the frozen GEOMETRY model (flow + depth-parallax features, per-atom logistic regression fitted on the 903 ShotQA
train clips, `move_geom_flowdepth_v1`, C = 0.01, train statistics) applied to the 1,077 NOTLD shot clips. Our film is an inference target: no label exists, nothing is tuned or selected on it. Rule fixed in
the amendment: atoms with probability >= 0.5, else the single top-1 atom (the set is never empty); no other threshold. Shots are cut from film.mp4 by start_s/end_s into a lossless temporary clip and
the UNCHANGED flow and depth code runs on it; per-shot series are cached (film hash + frame range + code keys), idempotent. CPU only, `nice -n 19`. No shot is dropped: flags `short_lt_1s`,
`few_pairs_lt_8`, `no_features`. Writes data/processed/night_of_the_living_dead_move.parquet (gitignored) and results/phase4/notld_move_report.md. All numbers are predictions of UNMEASURED accuracy.
Usage: uv run python scripts/p4_notld_move.py [--an earlier change] [--limit N]"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move.labels import schema_moves

ROOT = Path(__file__).resolve().parents[1]
FILM_ID = "night_of_the_living_dead"
SHOTS = ROOT / f"data/processed/{FILM_ID}_shots.parquet"
OUT = ROOT / f"data/processed/{FILM_ID}_move.parquet"
CACHE = ROOT / "data/cache/p4/notld"
REPORT = ROOT / "results/phase4/notld_move_report.md"
THREADS = 2


def _init() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(THREADS)


TARGET_FPS = 24000 / 1001  # 23.976: the frame rate of the training clips, so that the unchanged flow code samples every second frame (11.988 fps_used), Amendment 11


def cut_clip(film: Path, start_s: float, end_s: float, dst: Path) -> int:
    """The shot written LOSSLESSLY (FFV1) at TARGET_FPS: output frame j is the film frame nearest to start_s + j / TARGET_FPS, j < floor((end_s - start_s) x TARGET_FPS); returns the frame count."""
    cap = cv2.VideoCapture(str(film))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_out = int((end_s - start_s) * TARGET_FPS)
    src = [round((start_s + j / TARGET_FPS) * fps) for j in range(n_out)]
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"FFV1"), TARGET_FPS, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, src[0] if src else 0)
    pos, last, n = src[0] if src else 0, None, 0
    for i in src:
        while pos <= i:
            ok, fr = cap.read()
            if not ok:
                break
            last, pos = fr, pos + 1
        if last is None:
            break
        wr.write(last)
        n += 1
    wr.release()
    cap.release()
    return n


def one(args) -> tuple[int, dict | None, str | None]:
    sid, film, start_s, end_s, film_key = args
    try:
        key = f"{F.cache_key()}.{D.cache_key()}.{film_key}"
        f_path, d_path = CACHE / "flow" / f"shot{sid}.{key}.npz", CACHE / "depth" / f"shot{sid}.{key}.npz"
        if not (f_path.exists() and d_path.exists()):
            with tempfile.TemporaryDirectory() as td:
                clip = Path(td) / "shot.avi"
                if cut_clip(Path(film), start_s, end_s, clip) < 3:
                    return sid, None, "fewer than 3 frames"
                for dst, fn in ((f_path, F.clip_series), (d_path, D.clip_series)):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(".tmp_" + dst.name)
                    with open(tmp, "wb") as fh:
                        np.savez(fh, **fn(clip))
                    tmp.replace(dst)
        zf, zd = np.load(f_path), np.load(d_path)
        row = F.clip_features({k: zf[k] for k in zf.files}) | D.clip_features({k: zd[k] for k in zd.files if k != "depth"})
        return sid, row, None
    except Exception as e:  # noqa: BLE001 - reported and flagged, the shot is kept
        return sid, None, f"{type(e).__name__}: {e}"


def predict(P: np.ndarray, atoms: list[str]) -> tuple[list[list[str]], list[bool]]:
    """Amendment 10 item 2: atoms with p >= 0.5, else the single top-1 atom. Returns (sets in schema order, fallback flags)."""
    sets, fb = [], []
    for p in P:
        idx = [j for j in range(len(atoms)) if p[j] >= 0.5]
        fb.append(not idx)
        sets.append([atoms[j] for j in (idx or [int(np.argmax(p))])])
    return sets, fb


def main() -> None:
    global THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    THREADS = a.threads
    from p4_final_eval import TrainContext
    t0 = time.time()
    film = ROOT / next(f for f in yaml.safe_load((ROOT / "configs/films.yaml").read_text())["films"] if f["id"] == FILM_ID)["path"]
    shots = pd.read_parquet(SHOTS)[["shot_id", "start_s", "end_s", "len_s"]].sort_values("shot_id").reset_index(drop=True)
    if a.limit:
        shots = shots.head(a.limit)
    h = hashlib.sha256()
    with open(film, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    film_sha = h.hexdigest()
    geo = FR.geometry_identity()
    ctx = TrainContext(geo["C"], geo["split"])
    jobs = [(int(r.shot_id), str(film), float(r.start_s), float(r.end_s), f"{film_sha[:12]}.{round(r.start_s * 1000)}-{round(r.end_s * 1000)}.r24") for r in shots.itertuples()]
    rows, errs = {}, {}
    with Pool(a.workers, initializer=_init) as pool:
        for i, (sid, row, err) in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            (errs.__setitem__(sid, err) if err else rows.__setitem__(sid, row))
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} shots, {time.time() - t0:.0f}s", flush=True)
    cols = ctx.columns
    X = np.full((len(shots), len(cols)), np.nan)
    for i, sid in enumerate(shots.shot_id):
        if sid in rows:
            X[i] = [rows[sid][c] for c in cols]
    ok = np.array([int(s) in rows for s in shots.shot_id])
    P = np.tile(ctx.prior, (len(shots), 1))
    if ok.any():
        P[ok] = FE.fit_atom_probs(ctx.Xtr, ctx.Ytr, X[ok], ctx.C)
    sets, fb = predict(P, ctx.atoms)
    order = {a_: i for i, a_ in enumerate(schema_moves())}
    cfg = json.loads((ROOT / "results/phase4/frozen_config_flow_depth.json").read_text())["config_hash"]
    df = pd.DataFrame({"shot_id": shots.shot_id.astype(int)})
    for j, at in enumerate(ctx.atoms):
        df[f"p_{at}"] = P[:, j]
    df["move_atoms"] = [" + ".join(sorted(s, key=order.get)) for s in sets]
    df["n_atoms"] = [len(s) for s in sets]
    df["fallback_top1"] = fb
    df["short_lt_1s"] = (shots.len_s < 1.0).to_numpy()
    df["n_pairs"] = [rows[s]["n_pairs"] if s in rows else np.nan for s in shots.shot_id]
    df["few_pairs_lt_8"] = df.n_pairs < 8
    df["no_features"] = ~ok
    df["feature_code_hash"] = f"{F.cache_key()}.{D.cache_key()}"
    df["model"], df["config_hash"] = "move_geom_flowdepth_v1", cfg
    if not a.limit:
        df.to_parquet(OUT, index=False)
    write_report(df, shots, ctx, film_sha, errs, partial=bool(a.limit))
    print(f"{len(df)} shots, {int(df.no_features.sum())} without features, {int(df.fallback_top1.sum())} top-1 fallbacks, {time.time() - t0:.0f}s")


def write_report(df: pd.DataFrame, shots: pd.DataFrame, ctx, film_sha: str, errs: dict, partial: bool) -> None:
    n = len(df)
    cnt = Counter(a for s in df.move_atoms for a in s.split(" + "))
    sizes = df.n_atoms.value_counts().sort_index()
    sec = shots.set_index("shot_id").start_s.reindex(df.shot_id).to_numpy()
    edges = np.arange(0, np.ceil(sec.max() / 600) * 600 + 1, 600)
    L = ["# NOTLD camera movement: predictions of the frozen geometry model (INFERENCE ONLY; accuracy on NOTLD is UNMEASURED)", "",
         f"Model `move_geom_flowdepth_v1` (flow + depth-parallax features, per-atom logistic regression fitted on the 903 ShotQA train clips, C = {ctx.C}), applied to {n} NOTLD shot clips cut from `film.mp4` (sha256 `{film_sha[:16]}`) by the Phase 1 "
         "table. Rule of Amendment 10: atoms with probability >= 0.5, else the single top-1 atom. No NOTLD label exists, nothing was tuned on NOTLD; every number below is a PREDICTION, not a measurement. "
         "The geometry model has no example of zoom-in/zoom-out beyond 3 train clips each and none of dolly-zoom; on ShotQA validation its option accuracy was 72.9% (a four-option task on other sources), which says nothing about free atom sets on this film." + (" PARTIAL RUN (--limit)." if partial else ""), "",
         f"- shots {n}; top-1 fallback (no atom reached 0.5): **{int(df.fallback_top1.sum())}** ({df.fallback_top1.mean():.1%}); flagged: under 1 s {int(df.short_lt_1s.sum())}, fewer than 8 frame pairs {int(df.few_pairs_lt_8.sum())}, no features {int(df.no_features.sum())} (kept; errors: {errs or 'none'})",
         f"- set sizes: {', '.join(f'{k} atoms: {v}' for k, v in sizes.items())}; static share (a set containing `static`): **{(df.move_atoms.str.contains('static')).mean():.1%}**; exactly `static` alone: {(df.move_atoms == 'static').mean():.1%}", "",
         "## Shots per atom (a shot counts once per atom in its predicted set)", "", "| atom | shots | share |", "|---|---:|---:|"]
    L += [f"| {a} | {cnt.get(a, 0)} | {cnt.get(a, 0) / n:.1%} |" for a in ctx.atoms]
    L += ["", "## Series over the film (10-minute bins by shot start; share of shots whose predicted set contains the atom)", "",
          "| minutes | shots | static | pan-left/right | tilt | push/pull | move l/r | boom | mean n_atoms | fallback |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lo, hi in pairwise(edges):
        m = (sec >= lo) & (sec < hi)
        if not m.any():
            continue
        s = df.move_atoms[m]
        def sh(pat, s=s):
            return f"{s.str.contains(pat).mean():.0%}"

        L.append(f"| {int(lo // 60)}-{int(hi // 60)} | {int(m.sum())} | {sh('static')} | {sh('pan-')} | {sh('tilt-')} | {sh('push-in|pull-out')} | {sh('move-')} | {sh('boom-')} | {df.n_atoms[m].mean():.2f} | {df.fallback_top1[m].mean():.0%} |")
    fl = df[df.short_lt_1s | df.few_pairs_lt_8 | df.no_features]
    L += ["", f"## Flagged shots ({len(fl)}; kept, less reliable)", "", f"under 1 s: {int(df.short_lt_1s.sum())}; fewer than 8 frame pairs: {int(df.few_pairs_lt_8.sum())} (shot ids {df.shot_id[df.few_pairs_lt_8].tolist()}); no features: {int(df.no_features.sum())}. "  # noqa: ISC004 - adjacent string literals are intended
          f"Among the flagged shots the top-1 fallback rate is {fl.fallback_top1.mean() if len(fl) else float('nan'):.1%} (unflagged {df[~df.index.isin(fl.index)].fallback_top1.mean():.1%}); static in {fl.move_atoms.str.contains('static').mean() if len(fl) else float('nan'):.1%} (unflagged {df[~df.index.isin(fl.index)].move_atoms.str.contains('static').mean():.1%}).", "",
          "Feature code hash `" + str(df.feature_code_hash.iloc[0]) + "`; table `data/processed/night_of_the_living_dead_move.parquet` (gitignored)."]
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
