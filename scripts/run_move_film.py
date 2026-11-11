"""Film-level camera-movement slot by INFERENCE with the frozen geometry model (docs/phase4_film_move.md): generalizes scripts/p4_notld_move.py (Amendments 10 and 11 of docs/phase4_video_plan.md). The rule is
unchanged: atoms with probability >= 0.5, else the single top-1 atom. Refuses an unregistered film, an owned (not public_domain) film without --allow-owned, a missing film file or shot table, an existing output
(without --overwrite) and a machine with less than 25% free memory. CPU only, lowest priority, at most 3 workers; per-shot series are cached (film hash + frame range + code keys). Every output is a PREDICTION of
UNMEASURED accuracy; owned films write only under gitignored data/.
Usage: uv run python scripts/run_move_film.py --film night_of_the_living_dead [--cache-dir D] [--out T] [--shots N] [--an earlier change]
       uv run python scripts/run_move_film.py --film <owned_film_id> --allow-owned"""
import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import film_move as FM
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_run as FR
from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move.labels import schema_moves
from cgt.pipeline import registry

ROOT = Path(__file__).resolve().parents[1]
THREADS = 2


def _init() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(THREADS)


def one(args) -> tuple[int, dict | None, str | None]:
    sid, film, start_s, end_s, fkey, cache = args
    try:
        f_path, d_path = FM.cache_names(Path(cache), sid, fkey)
        if not (f_path.exists() and d_path.exists()):
            with tempfile.TemporaryDirectory() as td:
                clip = Path(td) / "shot.avi"
                if FM.cut_clip(Path(film), start_s, end_s, clip) < 3:
                    return sid, None, "fewer than 3 frames"
                for dst, fn in ((f_path, F.clip_series), (d_path, D.clip_series)):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(".tmp_" + dst.name)
                    with open(tmp, "wb") as fh:
                        np.savez(fh, **fn(clip))
                    tmp.replace(dst)
        zf, zd = np.load(f_path), np.load(d_path)
        return sid, F.clip_features({k: zf[k] for k in zf.files}) | D.clip_features({k: zd[k] for k in zd.files if k != "depth"}), None
    except Exception as e:  # noqa: BLE001 - reported and flagged, the shot is kept
        return sid, None, f"{type(e).__name__}: {e}"


def build_table(shots: pd.DataFrame, rows: dict, ctx, geo: dict) -> pd.DataFrame:
    cols = ctx.columns
    X = np.full((len(shots), len(cols)), np.nan)
    for i, sid in enumerate(shots.shot_id):
        if int(sid) in rows:
            X[i] = [rows[int(sid)][c] for c in cols]
    ok = np.array([int(s) in rows for s in shots.shot_id])
    P = np.tile(ctx.prior, (len(shots), 1))
    if ok.any():
        P[ok] = FE.fit_atom_probs(ctx.Xtr, ctx.Ytr, X[ok], ctx.C)
    sets, fb = FM.predict(P, ctx.atoms)
    order = {a: i for i, a in enumerate(schema_moves())}
    df = pd.DataFrame({"shot_id": shots.shot_id.astype(int)})
    for j, at in enumerate(ctx.atoms):
        df[f"p_{at}"] = P[:, j]
    df["move_atoms"] = [" + ".join(sorted(s, key=order.get)) for s in sets]
    df["n_atoms"] = [len(s) for s in sets]
    df["fallback_top1"] = fb
    df["short_lt_1s"] = (shots.len_s < 1.0).to_numpy()
    df["n_pairs"] = [rows[int(s)]["n_pairs"] if int(s) in rows else np.nan for s in shots.shot_id]
    df["few_pairs_lt_8"] = df.n_pairs < 8
    df["no_features"] = ~ok
    df["feature_code_hash"] = f"{F.cache_key()}.{D.cache_key()}"
    df["model"], df["config_hash"] = "move_geom_flowdepth_v1", geo["config_hash"]
    return df


def write_report(path: Path, film: dict, df: pd.DataFrame, shots: pd.DataFrame, ctx, film_sha: str, errs: dict, partial: bool) -> dict:
    sets = [s.split(" + ") for s in df.move_atoms]
    st = FM.describe(sets, df.fallback_top1.tolist())
    flagged = int((df.short_lt_1s | df.few_pairs_lt_8 | df.no_features).sum())
    L = [f"# Movement slot of `{film['id']}`: predictions of the frozen geometry model (INFERENCE ONLY; accuracy on this film is UNMEASURED)", "",
         f"Model `move_geom_flowdepth_v1` (C = {ctx.C}); {st['n_shots']} shots cut from the film file (sha256 `{film_sha[:16]}`) by the shot table, covering 0-{shots.end_s.max():.0f} s. Rule of Amendment 10 (unchanged): atoms with probability >= 0.5, else the single "
         "top-1 atom. No label of this film exists and nothing was tuned on it: every number is a PREDICTION of unmeasured accuracy." + (" PARTIAL RUN (--shots)." if partial else ""), "",
         f"- set sizes: {', '.join(f'{k} atoms: {v}' for k, v in st['set_sizes'].items())}; median {st['median_set_size']:.0f}",
         f"- exactly `static`: {st['static_alone']} ({st['static_alone_share']:.1%}); top-1 fallback: {st['fallback_top1']} ({st['fallback_top1'] / st['n_shots']:.1%})",
         f"- **contradictory sets** (an opposite pair, or `static` with another atom; definition in docs/phase4_film_move.md): {st['contradictory']} ({st['contradictory_share']:.1%}); with an opposite pair {st['opposite_pair']}, `static` with another atom {st['static_with_other']}",
         f"- flags (shots kept): under 1 s {int(df.short_lt_1s.sum())}, fewer than 8 frame pairs {int(df.few_pairs_lt_8.sum())}, no features {int(df.no_features.sum())} (errors: {errs or 'none'}); flagged in total {flagged}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return st


def resolve(film_id: str, allow_owned: bool, root: Path) -> dict:
    """The film through the merged registry. An overlay film is
    local only: refused without --allow-owned whatever its public_domain flag."""
    film = FM.resolve_film(film_id, allow_owned=allow_owned, root=root, films_yaml=None)
    if registry.is_overlay(film) and not allow_owned:
        raise FM.Refused(2, f"film '{film_id}' is an ad-hoc film of the local overlay (local only): pass --allow-owned for local, unpublished use")
    return film


def default_report_path(root: Path, film: dict) -> Path:
    """results/phase4/film_move/ only for a public-domain films.yaml film; an owned film and every overlay film report under gitignored data/."""
    if film.get("public_domain") and not registry.is_overlay(film):
        return root / f"results/phase4/film_move/{film['id']}_report.md"
    return root / f"data/processed/{film['id']}_move_report.md"


def main() -> None:
    global THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--allow-owned", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache-dir")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shots", type=int, help="evenly spaced subset of N shots (partial run; used for cold-cache checks)")
    a = ap.parse_args()
    THREADS = min(a.threads, 2)
    try:
        try:
            os.setpriority(os.PRIO_PROCESS, 0, 19)
        except OSError:
            pass  # already at the lowest priority through `nice -n 19` (the OS refuses a repeated change)
        film = resolve(a.film, a.allow_owned, ROOT)
        out = Path(a.out) if a.out else ROOT / f"data/processed/{a.film}_move.parquet"
        FM.refuse_existing(out, a.overwrite)
        FM.memory_guard()
    except FM.Refused as e:
        print(f"REFUSED (exit {e.code}): {e}")
        sys.exit(e.code)
    report = Path(a.report) if a.report else default_report_path(ROOT, film)
    cache = Path(a.cache_dir) if a.cache_dir else ROOT / f"data/cache/p4/film_move/{a.film}"
    from p4_final_eval import TrainContext
    t0 = time.time()
    geo = FR.geometry_identity()
    ctx = TrainContext(geo["C"], geo["split"])
    shots = pd.read_parquet(film["shots_path"])[["shot_id", "start_s", "end_s", "len_s"]].sort_values("shot_id").reset_index(drop=True)
    if a.shots:
        shots = shots.iloc[np.linspace(0, len(shots) - 1, a.shots).astype(int)].reset_index(drop=True)
    sha = FM.film_sha256(film["film_path"])
    jobs = [(int(r.shot_id), str(film["film_path"]), float(r.start_s), float(r.end_s), FM.film_key(sha, r.start_s, r.end_s), str(cache)) for r in shots.itertuples()]
    rows, errs = {}, {}
    with Pool(a.workers, initializer=_init) as pool:
        for i, (sid, row, err) in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            (errs.__setitem__(sid, err) if err else rows.__setitem__(sid, row))
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} shots, {time.time() - t0:.0f}s", flush=True)
    df = build_table(shots, rows, ctx, geo)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    st = write_report(report, film, df, shots, ctx, sha, errs, partial=bool(a.shots))
    print(json.dumps({"film": a.film, "table": str(out), "report": str(report), **st}, indent=1))


if __name__ == "__main__":
    main()
