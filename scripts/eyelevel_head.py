"""Track E: eye-level head on frozen SigLIP-base features. Plan: docs/plans/eyelevel_head.md (binding).

CPU only, no GPU, no downloads. Never reads data/gold, data/dev, ShotBench or CameraBench files.
Subcommands (run in this order):
  prep   verify the feature pool (Phase 3A embedder, covers split_v3 angle train + val), rebuild an earlier change's near-duplicate groups
         from the cached hashes and check them against split_v3. Writes data/cache/eyelevel_head/prep.json (ids, groups)
         and results/eyelevel_head/prep.json (counts only).
  cv     REFUSES unless the plan is committed and unchanged. Grouped 5-fold CV over the pre-registered grid on split_v3
         angle TRAIN only, selection, threshold on out-of-fold train scores, refit on all train. Writes
         results/eyelevel_head/cv.json; head weights to data/cache/eyelevel_head/heads.pkl (gitignored).
  final  REFUSES unless the plan and cv.json are committed and unchanged, and refuses to run twice. Scores CineScale2 val
         ONCE (with group bootstrap CIs), the ShotQA validation guard (aggregates only), applies the adoption rule.
         Writes results/eyelevel_head/summary.json.
  diag   POST HOC (added after final, descriptive only): image-size source confound check -> diag_source.json.
Run: PYTHONPATH=src nice -n 19 <venv>/bin/python scripts/eyelevel_head.py {prep,cv,final}"""
import argparse
import datetime as dt
import hashlib
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.eyelevel import head as H
from cgt.worktrees import sibling

CS2 = sibling("cinescale2")
POOL = CS2 / "data/cache/cinescale2/g1_cs2_pool.npz"
HASHES = CS2 / "data/cache/cinescale2/hashes_cs2_e974f456323fa9f4.npz"
SPLIT = CS2 / "results/cinescale2/split_v3.json"
SPLIT_SHA = "01e279f89cca408b00243d35dd444fc22313e6b19219a1fae43e5ab5eb149b8c"  # as recorded in results/stepb/data_summary.json
SHOTQA_VAL = CS2 / "data/cache/stepb/shotqa_val_dedup.jsonl"
SHOTQA_FEATS = (sibling("phase2") / "data/cache/p3/"
                    "google__siglip-base-patch16-224")
PLAN = "docs/plans/eyelevel_head.md"
OUT = ROOT / "results/eyelevel_head"
CACHE = ROOT / "data/cache/eyelevel_head"
FORBIDDEN = ("data/gold", "data/dev", "shotbench", "camerabench", "g1_shotbench")


def guard_path(p: Path) -> Path:
    r = str(Path(p).resolve()).lower()
    if any(f in r for f in FORBIDDEN):
        raise SystemExit(f"refusing to read {p}")
    return Path(p)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def require_committed(rel: str) -> str:
    """The file must be tracked, committed and unchanged in the work tree; returns the last commit touching it."""
    git = ["git", "-C", str(ROOT)]
    if subprocess.run(git + ["ls-files", "--error-unmatch", rel], capture_output=True, check=False).returncode:
        raise SystemExit(f"{rel} is not committed")
    if subprocess.run(git + ["diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        raise SystemExit(f"{rel} has uncommitted changes")
    return subprocess.run(git + ["log", "-1", "--format=%H %cI", "--", rel], capture_output=True, text=True, check=False).stdout.strip()


def load_split() -> dict:
    if sha(guard_path(SPLIT)) != SPLIT_SHA:
        raise SystemExit("split_v3.json sha256 differs from the Step B record")
    return json.loads(SPLIT.read_text())


def load_pool() -> tuple[list[str], np.ndarray, dict]:
    z = np.load(guard_path(POOL), allow_pickle=False)
    st = H.check_embedder(str(z["stats"]))
    return [str(i) for i in z["ids"]], H.l2norm(z["feats"]), st


def load_train_val():
    ids, feats, st = load_pool()
    split = load_split()
    train, val = H.angle_split(split, ids)
    want = split["train_counts"]["angle"]
    got = Counter(i.split("/")[-2] for i in train)
    if dict(got) != want:
        raise SystemExit(f"angle train counts {dict(got)} != split_v3 {want}")
    pos = {i: k for k, i in enumerate(ids)}
    return (train, feats[[pos[i] for i in train]], np.array([H.label_of(i) for i in train]),
            val, feats[[pos[i] for i in val]], st)


# ---------------------------------------------------------------- prep
def cmd_prep(_a) -> None:
    ids, _feats, st = load_pool()
    split = load_split()
    train, val = H.angle_split(split, ids)
    want = split["train_counts"]["angle"]
    got = Counter(i.split("/")[-2] for i in train)
    if dict(got) != want:
        raise SystemExit(f"angle train counts {dict(got)} != split_v3 {want}")
    if len(val) != sum(split["val_counts"]["angle"].values()):
        raise SystemExit("angle val count differs from split_v3")
    z = np.load(guard_path(HASHES), allow_pickle=False)
    group_of = H.build_groups([str(i) for i in z["ids"]], z["full"], z["trim"], z["thumbs"])
    n_groups_all = len(set(group_of.values()))
    if n_groups_all != split["groups"]["n_groups"]:
        raise SystemExit(f"rebuilt {n_groups_all} groups, split_v3 records {split['groups']['n_groups']}")
    H.check_groups(group_of, train, val)
    # all split_v3 train/val (both tasks) must also be group-consistent
    all_val = set(split["val"])
    excluded = set().union(*map(set, split["excluded"].values()))
    all_train = [i for i in group_of if i not in all_val and i not in excluded]
    H.check_groups(group_of, all_train, all_val)
    norms = np.linalg.norm(np.load(POOL)["feats"][:256].astype(np.float32), axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (CACHE / "prep.json").write_text(json.dumps({"train": train, "val": val,
                                                 "groups": {i: group_of[i] for i in train + val}}))
    info = {"time": now(), "embedder": {k: st[k] for k in ("model", "revision", "weights_sha256", "device")},
            "pool": {"file": str(POOL), "sha256": sha(POOL), "n": len(ids), "raw_norm_range_first256":
                     [float(norms.min()), float(norms.max())]},
            "split_v3_sha256": SPLIT_SHA, "angle_train": len(train), "angle_val": len(val),
            "angle_train_counts": dict(got), "groups_rebuilt": n_groups_all,
            "angle_train_groups": len({group_of[i] for i in train}), "angle_val_groups": len({group_of[i] for i in val}),
            "groups_spanning_train_and_val": 0}
    (OUT / "prep.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------- cv
def cmd_cv(_a) -> None:
    plan = require_committed(PLAN)
    train, xtr, ytr, val, _xva, _st = load_train_val()
    del _xva  # never used in this step
    prep = json.loads((CACHE / "prep.json").read_text())
    if prep["train"] != train or prep["val"] != val:
        raise SystemExit("prep.json does not match the current split")
    groups = [prep["groups"][i] for i in train]
    fold = H.folds(ytr, groups)
    is_eye = ytr == H.EYE
    results, oof_eye, oof_probs = {}, {}, {}
    for cfg in H.grid():
        eye, probs = H.oof(cfg, xtr, ytr, fold)
        r = {"family": cfg.family, "C": cfg.C, "cw": cfg.class_weight,
             "eye_ap": H.eye_ap(is_eye, eye)}
        if probs is not None:
            pred = probs.argmax(1)
            r["balanced_acc"] = H.balanced_accuracy(ytr, pred)
            r["acc"] = float(np.mean(pred == ytr))
        results[cfg.key()] = r
        oof_eye[cfg.key()], oof_probs[cfg.key()] = eye, probs
        print(cfg.key(), {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and k != "C"}, flush=True)
    head_key, det_key = H.select_configs(results, train, val)
    thr = H.select_threshold(train, oof_eye[det_key], is_eye, val)
    cfgs = {c.key(): c for c in H.grid()}
    head = H.fit(cfgs[head_key], xtr, ytr)
    det = H.fit(cfgs[det_key], xtr, ytr)
    CACHE.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps({"head_key": head_key, "det_key": det_key, "head": head, "det": det, "threshold": thr})
    (CACHE / "heads.pkl").write_bytes(blob)
    out = {"time": now(), "plan_commit": plan, "n_train": len(train), "n_train_groups": len(set(groups)),
           "fold_sizes": Counter(int(f) for f in fold), "grid": results, "selected_5class_head": head_key,
           "selected_eye_detector": det_key, "threshold": thr,
           "heads_pkl_sha256": hashlib.sha256(blob).hexdigest(), "heads_pkl": str(CACHE / "heads.pkl")}
    out["fold_sizes"] = {str(k): v for k, v in sorted(out["fold_sizes"].items())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cv.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("selected_5class_head", "selected_eye_detector", "threshold")}, indent=1))


# ---------------------------------------------------------------- final (val once + guard)
def shotqa_guard_set() -> tuple[list[str], list[list[str]]]:
    """Unique ShotQA validation images with an angle question (deduplicated Step B file) and their angle labels."""
    labels: dict[str, set[str]] = {}
    for line in guard_path(SHOTQA_VAL).read_text().splitlines():
        s = json.loads(line)
        if s["attribute"] == "angle":
            labels.setdefault(s["img_id"], set()).update(x.strip() for x in s["options"][s["answer"]].split(","))
    ids = sorted(labels)
    return ids, [sorted(labels[i]) for i in ids]


def shotqa_feats(ids: list[str]) -> np.ndarray:
    root = guard_path(SHOTQA_FEATS)
    prov = json.loads((root / "provenance.json").read_text())
    if prov["model_name"] != H.EMBEDDER["model"] or prov["model_revision"] != H.EMBEDDER["revision"]:
        raise SystemExit("ShotQA cache is not the Phase 3A embedder")
    want, found = set(ids), {}
    for f in sorted(root.glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        for i, v in zip(z["ids"], z["feats"], strict=True):
            if str(i) in want:
                found[str(i)] = v
    missing = want - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} ShotQA val images missing from the feature cache")
    return H.l2norm(np.stack([found[i] for i in ids]))


def cmd_final(_a) -> None:
    plan = require_committed(PLAN)
    cvc = require_committed("results/eyelevel_head/cv.json")
    stamp = CACHE / "final.stamp"
    if stamp.exists() or (OUT / "summary.json").exists():
        raise SystemExit("CineScale2 val was already evaluated once; refusing to run again")
    cv = json.loads((OUT / "cv.json").read_text())
    blob = (CACHE / "heads.pkl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != cv["heads_pkl_sha256"]:
        raise SystemExit("heads.pkl differs from the one recorded in cv.json")
    m = pickle.loads(blob)  # our own file, sha-checked above
    cfgs = {c.key(): c for c in H.grid()}
    dcfg, thr = cfgs[m["det_key"]], m["threshold"]
    t_eval = now()
    stamp.write_text(t_eval + "\n")
    _train, _xtr, _ytr, val, xva, _st = load_train_val()
    prep = json.loads((CACHE / "prep.json").read_text())
    groups = [prep["groups"][i] for i in val]
    yva = np.array([H.label_of(i) for i in val])
    pred = H.class_probs(m["head"], xva).argmax(1)
    score = H.eye_score(dcfg, m["det"], xva)
    flag = score >= thr["threshold"]
    is_eye = yva == H.EYE

    def boot(fn):
        return H.ci(H.group_bootstrap(groups, fn))
    five = H.five_class(yva, pred)
    five_ci = {"accuracy": boot(lambda ix: np.mean(pred[ix] == yva[ix])),
               "balanced_accuracy": boot(lambda ix: H.balanced_accuracy(yva[ix], pred[ix])),
               "recall": {c: boot(lambda ix, i=i: np.mean(pred[ix][yva[ix] == i] == i)) for i, c in enumerate(H.CLASSES)}}
    eye = H.eye_prf(is_eye, flag)

    def prf(ix, key):
        return H.eye_prf(is_eye[ix], flag[ix])[key]
    eye_ci = {k: boot(lambda ix, k=k: prf(ix, k)) for k in ("precision", "recall", "f1")}
    eye_argmax = H.eye_prf(is_eye, pred == H.EYE)
    ap = H.eye_ap(is_eye, score)
    # guard: ShotQA validation angle images (no eye-level label exists there)
    qids, qlabels = shotqa_guard_set()
    qscore = H.eye_score(dcfg, m["det"], shotqa_feats(qids))
    qflag = qscore >= thr["threshold"]
    rate = H.override_rate(qflag)
    per_label = {}
    for lab in sorted({x for ls in qlabels for x in ls}):
        mask = np.array([lab in ls for ls in qlabels])
        k, n = int(qflag[mask].sum()), int(mask.sum())
        per_label[lab] = {"n": n, "overrides": k, "rate": k / n, "wilson95": H.wilson(k, n)}
    single = {}
    for combo, n in Counter(", ".join(ls) for ls in qlabels).most_common():
        mask = np.array([", ".join(ls) == combo for ls in qlabels])
        single[combo] = {"n": n, "overrides": int(qflag[mask].sum())}
    adopt, reasons = H.adoption(eye["precision"], eye["recall"], rate, thr["reached_target"])
    summary = {
        "time_val_evaluation": t_eval, "plan_commit": plan, "cv_commit": cvc,
        "selected_5class_head": m["head_key"], "selected_eye_detector": m["det_key"], "threshold": thr,
        "cinescale2_val": {"n": len(val), "groups": len(set(groups)), "five_class": five, "five_class_ci95": five_ci,
                           "eye_level_detector": eye, "eye_level_detector_ci95": eye_ci, "eye_level_ap": ap,
                           "eye_level_from_5class_argmax": eye_argmax,
                           "bootstrap": {"unit": "near-duplicate group", "n_boot": H.N_BOOT, "seed": H.BOOT_SEED}},
        "shotqa_guard": {"n_images": len(qids), "overrides": int(qflag.sum()), "override_rate": rate,
                         "wilson95": H.wilson(int(qflag.sum()), len(qids)), "cap": H.ADOPT_GUARD_CAP,
                         "per_label": per_label, "per_label_combination": single,
                         "score_percentiles": {str(p): float(np.percentile(qscore, p)) for p in (50, 90, 95, 99)}},
        "adoption": {"adopt": adopt, "reasons_failed": reasons,
                     "rule": {"train_cv_target_precision": H.TARGET_PRECISION, "val_precision_min": H.ADOPT_PRECISION,
                              "val_recall_min": H.ADOPT_RECALL, "shotqa_override_cap": H.ADOPT_GUARD_CAP}},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("cinescale2_val", "shotqa_guard", "adoption")}, indent=1)[:6000])


# ---------------------------------------------------------------- diag (POST HOC, not pre-registered; added after final)
def cmd_diag(_a) -> None:
    """Source/format confound check, descriptive only (nothing is selected or refit): (1) image-size mix per class on
    train; (2) OOF eye-level AP of a classifier on image SIZE only (width, height, aspect), same grouped folds, train only;
    (3) flag counts of the frozen detector split by class x (size == 720x576), on train (in-sample) and val."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    if not (OUT / "summary.json").exists():
        raise SystemExit("diag runs only after final")
    prep = json.loads((CACHE / "prep.json").read_text())
    z = np.load(guard_path(HASHES), allow_pickle=False)
    size = {str(i).split("/unpacked/", 1)[1]: (int(w), int(h)) for i, (w, h) in zip(z["ids"], z["size"], strict=True)}
    m = pickle.loads((CACHE / "heads.pkl").read_bytes())
    cfg, thr = {c.key(): c for c in H.grid()}[m["det_key"]], m["threshold"]["threshold"]
    train, xtr, ytr, val, xva, _st = load_train_val()
    mix = {}
    for c, name in enumerate(H.CLASSES):
        cnt = Counter(size[i] for i, y in zip(train, ytr, strict=True) if y == c)
        n = int((ytr == c).sum())
        mix[name] = {"n": n, "top_sizes": [[f"{w}x{h}", k, round(k / n, 3)] for (w, h), k in cnt.most_common(3)]}
    xs = np.array([[size[i][0], size[i][1], size[i][0] / size[i][1]] for i in train], float)
    fold = H.folds(ytr, [prep["groups"][i] for i in train])
    s = np.zeros(len(train))
    for k in np.unique(fold):
        clf = HistGradientBoostingClassifier(random_state=0).fit(xs[fold != k], ytr[fold != k] == H.EYE)
        s[fold == k] = clf.predict_proba(xs[fold == k])[:, 1]
    cells = {}
    for name, ids, x in (("train_in_sample", train, xtr), ("val", val, xva)):
        y = np.array([H.label_of(i) for i in ids])
        f = H.eye_score(cfg, m["det"], x) >= thr
        pal = np.array([size[i] == (720, 576) for i in ids])
        cells[name] = {nm: {"n": int(mk.sum()), "flagged": int(f[mk].sum())} for nm, mk in (
            ("neutral_720x576", (y == H.EYE) & pal), ("neutral_other_size", (y == H.EYE) & ~pal),
            ("other_class_720x576", (y != H.EYE) & pal), ("other_class_other_size", (y != H.EYE) & ~pal))}
    out = {"time": now(), "post_hoc": True, "note": "descriptive only; not part of the pre-registered rule",
           "train_size_mix": mix, "size_only_oof_eye_ap_train": H.eye_ap(ytr == H.EYE, s),
           "train_eye_prevalence": float(np.mean(ytr == H.EYE)), "detector_flags_by_size": cells}
    (OUT / "diag_source.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "cv", "final", "diag"))
    a = ap.parse_args()
    {"prep": cmd_prep, "cv": cmd_cv, "final": cmd_final, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
