"""Gate G1: semantic same-scene check, CineScale2 vs ShotBench test images. CPU only, never MPS,
never the GPU lock. Image embeddings only: no ShotBench questions, labels, answers or accuracy.

Phases (run in this order; `apply` refuses to run unless the calibration file is committed and unchanged):
  calibrate  ShotQA only. Threshold = 99th percentile of the different-film max-cosine null (cgt.cinescale2.semantic),
             reference size = number of ShotBench test images (a count, not their content); same-film sensitivity;
             CPU-vs-cache consistency of the ShotQA embeddings. Writes results/cinescale2/g1_calibration.json.
  embed      CPU embeddings of the post-hash-exclusion CineScale2 pool (split v1 train + val), the 268 frames attributed to
             ShotBench films (diagnostic) and the ShotBench test images -> data/cache/cinescale2/g1_*.npz (gitignored).
  apply      max cosine CineScale2 -> ShotBench; flagged = above the committed threshold; writes results/cinescale2/g1_result.json
             and results/cinescale2/g1_flags.json (consumed by scripts/cs2_prepare.py --g1-flags to write split_v2.json).

  PYTHONPATH= nice -n 19 <venv>/bin/python -I scripts/cs2_g1.py <phase> --cs2 data/external/cinescale2/unpacked \
     --shotbench-images <main>/data/external/shotbench/image --shotqa-features <a sibling working copy>/data/cache/p3/google__siglip-base-patch16-224"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cgt.cinescale2 import semantic as M

OUT = ROOT / "results/cinescale2"
CACHE = ROOT / "data/cache/cinescale2"
CAL = OUT / "g1_calibration.json"
N_SHOTBENCH_IMAGES = 3049  # size of the reference set (count only; from results/cinescale2/contamination.json)
SAME_FILM_K = 12           # ShotBench has ~3,049 images / 242 films = 12.6 frames per film
CONSISTENCY_N = 256
REJECT = ("data/gold", "data/dev")


def guard(p) -> Path:
    if any(r in str(Path(p).resolve()) for r in REJECT):
        raise SystemExit(f"refusing {p}")
    return Path(p)


def log(*a):
    print(*a, flush=True)


def shotqa_features(cache_dir: Path):
    ids, feats = [], []
    for f in sorted(Path(cache_dir).glob("shard_*.npz")):
        with np.load(f) as z:
            ids += [str(x) for x in z["ids"]]
            feats.append(z["feats"].astype(np.float32))
    prov = json.loads((Path(cache_dir) / "provenance.json").read_text())
    return ids, np.concatenate(feats), prov


def calibrate(a) -> None:
    from cgt.p3_static_attrs.data import SHOTQA_ROOT, assert_shotqa_path, image_films, image_path
    from cgt.p3b_vlm.data import film_split_ids
    ids, feats, prov = shotqa_features(a.shotqa_features)
    if prov.get("model_revision") != M.REVISION or prov.get("model_name") != M.MODEL:
        raise SystemExit(f"ShotQA feature cache is from another model: {prov}")
    train, val, _ = film_split_ids()
    film, _ = image_films(assert_shotqa_path(SHOTQA_ROOT / "meta.jsonl"), train | val)
    pos = {i: k for k, i in enumerate(ids)}
    tr = [pos[i] for i in sorted(train) if i in pos]
    va_ids = [i for i in sorted(val) if i in pos]
    va = [pos[i] for i in va_ids]
    cal = M.calibrate(feats[va], feats[tr], N_SHOTBENCH_IMAGES, seed=0)
    thr = cal["threshold"]
    # sensitivity: same-film frames (hash near-duplicates of the same frame are not drawn: dHash <= 8 within val)
    from cgt.cinescale2 import contamination as C
    zf = sorted(CACHE.glob("hashes_shotqa_val_*.npz"))
    ex = set()
    if zf:
        z = np.load(zf[-1])
        hid = [Path(str(x)).name for x in z["ids"]]
        hp = C.candidate_pairs(z["full"], z["trim"], z["full"], z["trim"], 8, same_set=True)
        hpos = {i: k for k, i in enumerate(va_ids)}
        for i, j, _ in hp:
            a_, b_ = hpos.get(hid[i]), hpos.get(hid[j])
            if a_ is not None and b_ is not None:
                ex.add((min(a_, b_), max(a_, b_)))
    sens = {f"k{k}": M.same_film_sensitivity(feats[va], [film[i] for i in va_ids], thr, k, seed=0, exclude_pairs=ex)
            for k in (1, 4, SAME_FILM_K)}
    # train films: a second, larger same-film sample (k = 12)
    tr_ids = [i for i in sorted(train) if i in pos]
    rng = np.random.default_rng(1)
    sub = sorted(rng.choice(len(tr_ids), size=8000, replace=False))
    sens["train_k12_subset8000"] = M.same_film_sensitivity(feats[[pos[tr_ids[s]] for s in sub]],
                                                          [film[tr_ids[s]] for s in sub], thr, SAME_FILM_K, seed=0)
    # CPU recomputation vs the (MPS-produced, float16) cache on a seeded sample of val images
    rng = np.random.default_rng(2)
    pick = sorted(rng.choice(len(va_ids), size=CONSISTENCY_N, replace=False))
    paths = [image_path(va_ids[k]) for k in pick]
    ok, cpu, st = M.embed_cpu(paths, batch=16, threads=a.threads, log=log)
    cc = (M.l2n(cpu) * M.l2n(feats[[va[pick[k]] for k in ok]])).sum(axis=1)
    res = {
        "gate": "G1 semantic same-scene check; calibration uses ShotQA ONLY",
        "embedder": {"model": M.MODEL, "revision": M.REVISION, "weights_sha256": M.WEIGHTS_SHA256,
                     "feature": "SiglipModel.get_image_features pooler output, L2-normalised, cosine"},
        "shotqa_feature_cache": {"dir": str(a.shotqa_features), "provenance": prov, "n_images": len(ids)},
        "null": {"definition": ("for each ShotQA film-split VALIDATION image: max cosine to a seeded (default_rng(0)) random set of "
                                f"{N_SHOTBENCH_IMAGES} ShotQA TRAIN images (all from different films by the film split)"), **cal},
        "threshold": thr, "rule": f"a CineScale2 image is flagged if its max cosine to any ShotBench test image > {thr}",
        "same_film_sensitivity": sens,
        "cpu_vs_cache_cosine": {"n": len(ok), "min": round(float(cc.min()), 5), "median": round(float(np.median(cc)), 5)},
        "cpu_embed_stats": st,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CAL.write_text(json.dumps(res, indent=1) + "\n")
    log(json.dumps(res, indent=1))


def _pool(a):
    s = json.loads((OUT / "split_v1.json").read_text())
    val = set(s["val"])
    dropped = set().union(*map(set, s["excluded"].values()))
    from cgt.cinescale2 import labels as L
    allrel = [it.relpath for it in L.list_items(guard(a.cs2).resolve())]
    pool = sorted(r for r in allrel if r not in dropped)
    diag = sorted(s["excluded"]["shotbench_film_via_shotqa"])
    return pool, diag, val


def embed(a) -> None:
    cs2_root = guard(a.cs2).resolve()
    pool, diag, _ = _pool(a)
    sb_dir = guard(a.shotbench_images)
    sb = sorted(p for p in sb_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, paths, ids in (("cs2_pool", [cs2_root / r for r in pool], pool), ("cs2_diag", [cs2_root / r for r in diag], diag),
                             ("shotbench", sb, [p.name for p in sb])):
        f = CACHE / f"g1_{name}.npz"
        key = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if f.exists() and str(np.load(f)["key"]) == key:
            log(f"{name}: cached")
            continue
        ok, feats, st = M.embed_cpu(paths, batch=16, threads=a.threads, log=log)
        np.savez(f, ids=np.array([ids[k] for k in ok]), feats=feats.astype(np.float32), key=key, stats=json.dumps(st))
        log(f"{name}: {st}")


def _committed_unchanged(path: Path) -> str:
    rel = str(path.relative_to(ROOT))
    if subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True, check=False).returncode:
        raise SystemExit(f"{rel} is not committed: commit the calibration (and the plan) before applying it to ShotBench")
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        raise SystemExit(f"{rel} differs from HEAD")
    return subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", rel], capture_output=True, text=True, check=False).stdout.strip()


def apply(a) -> None:
    commit = _committed_unchanged(CAL)
    thr = json.loads(CAL.read_text())["threshold"]
    _, _, val = _pool(a)
    z = {n: np.load(CACHE / f"g1_{n}.npz") for n in ("cs2_pool", "cs2_diag", "shotbench")}
    sb_ids = [str(x) for x in z["shotbench"]["ids"]]
    out = {"calibration_commit": commit, "threshold": thr, "n_shotbench_images": len(sb_ids)}
    flags = []
    for name in ("cs2_pool", "cs2_diag"):
        ids = [str(x) for x in z[name]["ids"]]
        m, _ = M.max_cos(z[name]["feats"], z["shotbench"]["feats"])
        hit = m > thr
        per = {}
        for r, h in zip(ids, hit, strict=True):
            c = "/".join(Path(r).parts[-3:-1])
            per.setdefault(c, [0, 0])
            per[c][0] += 1
            per[c][1] += int(h)
        out[name] = {"n": len(ids), "flagged": int(hit.sum()), "fraction": round(float(hit.mean()), 4) if len(ids) else None,
                     "per_class_n_flagged": dict(sorted(per.items())),
                     "max_cos_percentiles": {str(p): round(float(np.percentile(m, p)), 4) for p in (50, 90, 99, 100)}}
        if name == "cs2_pool":
            flags = sorted(r for r, h in zip(ids, hit, strict=True) if h)
            ang_tr = [r for r in ids if "/angle/" in f"/{r}" and r not in val]
            fs = set(flags)
            out["angle_train_v1"] = {"n": len(ang_tr), "flagged": sum(1 for r in ang_tr if r in fs)}
            out["angle_train_v1"]["fraction"] = round(out["angle_train_v1"]["flagged"] / max(len(ang_tr), 1), 4)
    out["diag_note"] = ("cs2_diag = the 268 frames already excluded because they are the same frame as a ShotQA image of a "
                        "ShotBench film (e.g. Hugo); their flag rate is a real-world sensitivity estimate, not used for tuning")
    out["stop_rule"] = "STOP and report if G1 flags > 25% of angle train (before group expansion)"
    (OUT / "g1_result.json").write_text(json.dumps(out, indent=1) + "\n")
    (OUT / "g1_flags.json").write_text(json.dumps({"threshold": thr, "calibration_commit": commit, "flagged": flags}, indent=0) + "\n")
    log(json.dumps(out, indent=1))
    if out["angle_train_v1"]["fraction"] > 0.25:
        log("STOP: G1 flags more than 25% of the angle train images")
        sys.exit(12)


LOOSE_MAD, LOOSE_CORR = 0.08, 0.8   # looser pixel near-duplicate rule vs ShotQA validation


def a2(a) -> None:
    """Advisory A2: CineScale2 pool images above the COMMITTED G1 threshold vs ShotQA VALIDATION images (semantic), plus
    loose pixel near-duplicates of ShotQA validation images (dHash candidate <= 8, MAD < 0.08, corr >= 0.8). No recalibration."""
    from cgt.cinescale2 import contamination as C
    from cgt.p3b_vlm.data import film_split_ids
    commit = _committed_unchanged(CAL)
    thr = json.loads(CAL.read_text())["threshold"]
    ids, feats, prov = shotqa_features(a.shotqa_features)
    if prov.get("model_revision") != M.REVISION:
        raise SystemExit("ShotQA feature cache is from another model")
    _, val, _ = film_split_ids()
    pos = {i: k for k, i in enumerate(ids)}
    vf = feats[[pos[i] for i in sorted(val) if i in pos]]
    z = np.load(CACHE / "g1_cs2_pool.npz")
    pool = [str(x) for x in z["ids"]]
    m, _ = M.max_cos(z["feats"], vf)
    sem = sorted(r for r, h in zip(pool, m > thr, strict=True) if h)
    # loose pixel rule on the hash caches written by scripts/cs2_prepare.py
    def one(pattern):
        fs = list(CACHE.glob(pattern))
        if len(fs) != 1:
            raise SystemExit(f"expected exactly one {pattern} in {CACHE}, found {len(fs)}")
        return np.load(fs[0])

    hc, hv = one("hashes_cs2_*.npz"), one("hashes_shotqa_val_*.npz")
    root = str(guard(a.cs2).resolve()) + "/"
    rel = [str(x).replace(root, "", 1) for x in hc["ids"]]
    pairs = C.candidate_pairs(hc["full"], hc["trim"], hv["full"], hv["trim"], 8)
    mad = C.pair_mad(hc["thumbs"], hv["thumbs"], pairs)
    cor = C.pair_corr(hc["thumbs"], hv["thumbs"], pairs)
    keep = (mad < LOOSE_MAD) & (cor >= LOOSE_CORR)
    pool_set = set(pool)
    loose = sorted({rel[i] for i in pairs[keep, 0]} & pool_set)
    res = {"threshold": thr, "calibration_commit": commit, "n_shotqa_val_images": len(vf), "n_pool": len(pool),
           "semantic_vs_shotqa_val": sem, "loose_pixel_vs_shotqa_val": loose,
           "loose_rule": f"dHash candidate <= 8, thumbnail MAD < {LOOSE_MAD} and corr >= {LOOSE_CORR}",
           "flagged": sorted(set(sem) | set(loose)),
           "max_cos_percentiles": {str(p): round(float(np.percentile(m, p)), 4) for p in (50, 90, 99, 100)}}
    (OUT / "a2_flags.json").write_text(json.dumps(res, indent=0) + "\n")
    log(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["calibrate", "embed", "apply", "a2"])
    ap.add_argument("--cs2", type=Path, default=ROOT / "data/external/cinescale2/unpacked")
    ap.add_argument("--shotbench-images", type=Path)
    ap.add_argument("--shotqa-features", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"calibrate": calibrate, "embed": embed, "apply": apply, "a2": a2}[a.phase](a)


if __name__ == "__main__":
    main()
