"""FEATURES-ONLY extremes of the test clips against the training distribution (docs/phase4_video_plan.md Amendment 12 item 5, Amendment 13 item 3): how many test clips sit more than 10 TRAINING standard
deviations from the TRAINING mean on any feature, in particular `fps_used`. It reads test videos and GIFs for features (frame rates, flow and depth features) and NEVER a label or an answer.
Modes: `--fps-metadata` (exact `fps_used` of every clip under the UNCHANGED sampling rule, from file metadata, before conformity); `--table T` (all features of a prepared table, after conformity);
`--sample N` (N clips per dataset, evenly spaced in sorted order: features with and without the conformity step side by side). Writes results/phase4/final_eval_extremes.md (modes append sections).
Usage: uv run python scripts/p4_final_extremes.py --fps-metadata | --table data/cache/p4_eval/features_shotbench_movement.parquet | --sample 40 [--an earlier change]"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_final_prepare as P

from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import final_stores as FS
from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move import gif as G
from cgt.p4_camera_move.camerabench import CB_ROOT

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/phase4/final_eval_extremes.md"
LIMIT_SD = 10.0


def train_stats(usable_train: bool = False) -> tuple[pd.Series, pd.Series, list[str]]:
    """Training mean and standard deviation. Default: the POOLED reference, all 1,058 ShotQA movement clips (train + validation features, no label). `usable_train=True`: the 903 atom-bearing TRAIN
    clips that the geometry model is actually fitted on (their train labels decide which clips have an atom; validation labels are never read)."""
    tr = pd.concat([pd.read_parquet(ROOT / "data/cache/p4" / f).set_index("clip") for f in ("features_shotqa.parquet", "features_depth_shotqa.parquet")], axis=1)
    cols = list(tr.columns)
    if usable_train:
        from cgt.p4_camera_move.labels import MovementLabelStore
        man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
        rows = [r for r in MovementLabelStore(man["clips"]["val"]).rows(man["clips"]["train"]) if r["atoms"]]
        tr = tr.loc[[r["clip"] for r in rows]]
    return tr.mean(), tr.std().replace(0, np.nan), cols


def extremes(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> dict:
    """Clips with |z| > 10 on any feature (features with a training standard deviation of 0 are reported separately), the worst features and `fps_used` specifically."""
    z = ((df - mu[df.columns]) / sd[df.columns]).abs()
    anyx = (z > LIMIT_SD).any(axis=1)
    worst = (z > LIMIT_SD).sum().sort_values(ascending=False)
    return {"n_clips": len(df), "clips_beyond_10sd_on_any_feature": int(anyx.sum()), "clips_beyond_10sd_on_fps_used": int((z["fps_used"] > LIMIT_SD).sum()) if "fps_used" in z else None,
            "median_abs_z_fps_used": round(float(z["fps_used"].median()), 2) if "fps_used" in z else None, "worst_features": {k: int(v) for k, v in worst.head(6).items() if v > 0},
            "max_abs_z": round(float(np.nanmax(z.to_numpy())), 1), "zero_variance_train_features": [c for c in df.columns if np.isnan(sd.get(c, np.nan))]}


def fps_metadata() -> dict:
    mu, sd, _ = train_stats()
    out = {"train_fps_used_mean": round(float(mu["fps_used"]), 4), "train_fps_used_sd": round(float(sd["fps_used"]), 5)}
    names = [q["clip"] for q in FS.ShotBenchMovementStore().questions()]
    paths = P.extract_shotbench_videos(names, P.EVAL_CACHE / "shotbench_movement_video")  # label-free extraction of the 464 videos named by the question file
    fp = {}
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        src = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        fp[p.name] = src / max(1, round(src / F.FPS))
    cb = {}
    for n in FS.CameraBenchFinalStore().clips():
        pr = G.gif_props(CB_ROOT / "videos_gif" / n)
        src = pr["fps"] if np.isfinite(pr["fps"]) and pr["fps"] > 0 else 12.0
        cb[n] = src / max(1, round(src / F.FPS))
    for name, d in (("shotbench_movement", fp), ("camerabench_public_test", cb)):
        a = np.array(list(d.values()))
        z = np.abs(a - mu["fps_used"]) / sd["fps_used"]
        out[name] = {"n": len(a), "fps_used_min": round(float(a.min()), 3), "median": round(float(np.median(a)), 3), "max": round(float(a.max()), 3), "n_beyond_10sd": int((z > LIMIT_SD).sum()),
                     "values": {str(k): int(v) for k, v in pd.Series(np.round(a, 2)).value_counts().head(8).items()}}
    return out


def sample_compare(n: int, workers: int) -> dict:
    mu, sd, cols = train_stats()
    res = {}
    for dataset in ("shotbench_movement", "camerabench_public_test"):
        if dataset == "shotbench_movement":
            names = [q["clip"] for q in FS.ShotBenchMovementStore().questions()]
            paths, kind = P.extract_shotbench_videos(names, P.EVAL_CACHE / "shotbench_movement_video"), "video"
        else:
            paths, kind = [CB_ROOT / "videos_gif" / x for x in FS.CameraBenchFinalStore().clips()], "gif"
        pick = [paths[i] for i in np.linspace(0, len(paths) - 1, n).astype(int)]
        with tempfile.TemporaryDirectory() as td:
            info = P.prepare_paths(pick, kind, Path(td) / "t.parquet", dataset=dataset, cache=P.CACHE, workers=workers)
            conformed = pd.read_parquet(Path(td) / "t.parquet").set_index("clip")
        rows = []
        for p in pick:  # the UNCHANGED sampling (no conformity step), for the comparison only
            if kind == "video":
                sf, sd_ = F.clip_series(p), D.clip_series(p)
            else:
                sf, sd_ = G.gif_series(p), G.gif_depth_series(p)
            rows.append({"clip": p.name} | F.clip_features(sf) | D.clip_features({k: v for k, v in sd_.items() if k != "depth"}))
        old = pd.DataFrame(rows).set_index("clip")[cols]
        res[dataset] = {"n": len(pick), "before_conformity": extremes(old, mu, sd), "after_conformity": extremes(conformed[cols], mu, sd), "nan_share_after": info["manifest"]["nan_share"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps-metadata", action="store_true")
    ap.add_argument("--table")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    head = "# Features-only extremes of the test clips against the training distribution (no label read; more than 10 TRAINING standard deviations from the TRAINING mean)\n\n"
    text = REPORT.read_text() if REPORT.exists() else head
    if a.fps_metadata:
        r = fps_metadata()
        text += f"\n## `fps_used` before conformity, exact, from file metadata (unchanged sampling rule)\n\n```\n{json.dumps(r, indent=1)}\n```\n"
    if a.table:
        df = pd.read_parquet(a.table).set_index("clip")
        r = {"reference_pooled_1058_clips (train+validation features)": extremes(df, *train_stats()[:2]), "reference_usable_903_TRAIN_clips (the geometry model's training rows)": extremes(df, *train_stats(True)[:2])}
        text += f"\n## Prepared table `{Path(a.table).name}` (after conformity; all features; two references)\n\n```\n{json.dumps(r, indent=1)}\n```\n"
    if a.sample:
        r = sample_compare(a.sample, a.workers)
        text += f"\n## Sample of {a.sample} clips per dataset, evenly spaced: before vs after the conformity step (all features)\n\n```\n{json.dumps(r, indent=1)}\n```\n"
    REPORT.write_text(text)
    print(text[-3000:])


if __name__ == "__main__":
    main()
