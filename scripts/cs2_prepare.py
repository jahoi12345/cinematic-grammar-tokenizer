"""CineScale2 inspection, contamination check and held-out split. CPU only; hashes, thumbnails and titles only.

Reads: the unpacked CineScale2 tree (untrusted: pixels only, nothing executed), ShotBench test IMAGES (dHash + 32x32 grey
thumbnail only: no labels, no questions, no accuracy), ShotQA images of the Phase 3 film split (same), and
docs/data/shotbench_films.tsv (titles). Never reads data/gold or data/dev.
Writes: data/cache/cinescale2/hashes_*.npz (gitignored), results/cinescale2/{inspect,contamination,split_v1}.json.

Rules (fixed before any Step B training; same shape as the Phase 3 ShotQA-vs-ShotBench rule, results/phase3/leakage_verified.md):
- hash candidate: 64-bit dHash distance <= 8 (min over full / border-trimmed hash combinations);
- same frame: candidate whose 32x32 grey thumbnail MAD < 0.05 (min over full / trimmed) AND correlation >= 0.9 (max);
- a CineScale2 image is EXCLUDED if it has a same-frame partner in ShotBench test or in ShotQA validation, OR any hash
  candidate within distance <= 4 there (unverified, conservative); its whole near-duplicate group goes with it;
- a CineScale2 image that is the same frame as a titled ShotQA image (train or val) of a film in the ShotBench film list
  (title match, strict or loose key) is EXCLUDED with its group (partial "remove the film in full": only attributable frames);
- synthetic `gen_` Dutch frames are excluded (rotations of unknown source frames that the hash cannot check);
- same-frame groups whose members carry DIFFERENT labels within one task (e.g. one frame filed as both eye and shoulder)
  are excluded in full (label noise);
- groups: union-find over CineScale2-internal same-frame pairs (non-flat frames); split = seeded 15% of groups -> val.

Run with an isolated interpreter (the repo src is added explicitly):
  PYTHONPATH= nice -n 19 <venv>/bin/python -I scripts/cs2_prepare.py --cs2 data/external/cinescale2/unpacked \
      --shotbench-images <main checkout>/data/external/shotbench/image"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cgt.cinescale2 import contamination as C
from cgt.cinescale2 import labels as L
from cgt.cinescale2 import split as S

CANDIDATE_DISTANCE = 8
UNVERIFIED_EXCLUDE_DISTANCE = 4
GROUP_DISTANCE = 8
VAL_FRAC, SEED = 0.15, 0
CACHE = ROOT / "data/cache/cinescale2"
FORBIDDEN = ("data/gold", "data/dev")


def guard(p: Path) -> Path:
    r = str(Path(p).resolve())
    if any(f in r for f in FORBIDDEN):
        raise SystemExit(f"refusing to read {r}")
    return Path(p)


def cached_hashes(name: str, paths: list[Path], workers: int) -> dict:
    """{'ids', 'full', 'trim', 'size', 'thumbs', 'n_input'} cached by the path list (version 2 = with thumbnails)."""
    key = hashlib.sha256(("v2\n" + "\n".join(map(str, paths))).encode()).hexdigest()[:16]
    f = CACHE / f"hashes_{name}_{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return {"ids": [str(x) for x in z["ids"]], "full": z["full"], "trim": z["trim"], "size": z["size"],
                "thumbs": z["thumbs"], "n_input": int(z["n_input"])}
    ok, full, trim, sizes, thumbs = C.hash_many(paths, workers=workers)
    d = {"ids": [str(paths[i]) for i in ok], "full": full, "trim": trim,
         "size": np.array(sizes, dtype=np.int64).reshape(-1, 2), "thumbs": thumbs, "n_input": len(paths)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: (np.array(v) if k == "ids" else v) for k, v in d.items()})
    return d


def q(v):
    return {k: int(np.percentile(v, p)) for k, p in (("min", 0), ("p5", 5), ("median", 50), ("p95", 95), ("max", 100))}


G1_STOP_FRACTION = 0.25


def derive_split(prev: dict, flagged: set[str], group_of: dict[str, str], by_class, version: int, key: str, note: str) -> dict:
    """`prev` minus every same-frame group that holds a flagged image; train/val membership of the rest is unchanged.
    Summary counts (`info`) are recomputed for the new split."""
    bad = {group_of[i] for i in flagged if i in group_of}
    excluded_prev = set().union(*map(set, prev["excluded"].values()))
    val_prev = prev["val"]
    train_prev = sorted(set(group_of) - set(val_prev) - excluded_prev)
    train = [i for i in train_prev if group_of[i] not in bad]
    val = [i for i in val_prev if group_of[i] not in bad]
    removed = sorted((set(train_prev) | set(val_prev)) - set(train) - set(val))
    ang_prev = [i for i in train_prev if "/angle/" in "/" + i]
    ang = [i for i in train if "/angle/" in "/" + i]
    out = dict(prev)
    out["version"] = version
    out["rule"] = prev["rule"] + f"; v{version} = v{version - 1} minus every group containing an image flagged by {note}"
    out[key] = {"flagged": len(flagged), "removed_with_groups": len(removed), f"angle_train_v{version - 1}": len(ang_prev),
                f"angle_train_v{version}": len(ang), "angle_train_removed_fraction": round(1 - len(ang) / max(len(ang_prev), 1), 4)}
    excluded = {**prev["excluded"], f"{key}_with_group": removed}
    all_excl = set().union(*map(set, excluded.values()))
    out["info"] = {**prev["info"], "n_groups": len({group_of[i] for i in train} | {group_of[i] for i in val}),
                   "n_val_groups": len({group_of[i] for i in val}), "train": len(train), "val": len(val),
                   "excluded_ids": len(all_excl), "dropped_with_group": len(group_of) - len(train) - len(val)}
    out["train_counts"], out["val_counts"] = by_class(train), by_class(val)
    out["excluded"] = excluded
    out["val"] = val
    return out


def split_v2(v1: dict, g1: dict, group_of: dict[str, str], by_class) -> dict:
    out = derive_split(v1, set(g1["flagged"]), group_of, by_class, 2, "g1", "G1 (semantic vs ShotBench)")
    out["g1"] = {"threshold": g1["threshold"], "calibration_commit": g1["calibration_commit"], **out["g1"]}
    out["excluded"]["g1_semantic_with_group"] = out["excluded"].pop("g1_with_group")
    return out


def split_v3(v2: dict, a2: dict, group_of: dict[str, str], by_class) -> dict:
    out = derive_split(v2, set(a2["flagged"]), group_of, by_class, 3, "a2",
                       "A2 (G1 threshold vs ShotQA validation, plus loose pixel near-duplicates of ShotQA validation)")
    out["a2"] = {"threshold": a2["threshold"], "calibration_commit": a2["calibration_commit"],
                 "semantic_vs_shotqa_val": len(a2["semantic_vs_shotqa_val"]), "loose_pixel_vs_shotqa_val": len(a2["loose_pixel_vs_shotqa_val"]),
                 **out["a2"]}
    out["excluded"]["a2_shotqa_val_with_group"] = out["excluded"].pop("a2_with_group")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs2", required=True, type=Path)
    ap.add_argument("--shotbench-images", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--g1-flags", type=Path, default=None,
                    help="results/cinescale2/g1_flags.json from scripts/cs2_g1.py apply: also write split_v2.json")
    ap.add_argument("--a2-flags", type=Path, default=None,
                    help="results/cinescale2/a2_flags.json from scripts/cs2_g1.py a2 (needs --g1-flags): also write split_v3.json")
    a = ap.parse_args()
    cs2_root = guard(a.cs2).resolve()

    # ---- CineScale2 listing, hashes, inspection
    items = L.list_items(cs2_root)
    cs2 = cached_hashes("cs2", [cs2_root / it.relpath for it in items], a.workers)
    by_path = {str(cs2_root / it.relpath): it for it in items}
    readable = [by_path[i] for i in cs2["ids"]]
    rel = [it.relpath for it in readable]
    unreadable = sorted(by_path[p].relpath for p in set(by_path) - set(cs2["ids"]))
    counts = Counter((it.task, it.label) for it in items)
    synth = Counter((it.task, it.label) for it in items if it.synthetic)
    w, h = cs2["size"][:, 0], cs2["size"][:, 1]
    res = Counter(f"{x}x{y}" for x, y in cs2["size"])
    cls_idx: dict[tuple[str, str], list[int]] = {}
    for k, it in enumerate(readable):
        cls_idx.setdefault((it.task, it.label), []).append(k)
    flat_cs2 = C.is_flat(cs2["thumbs"])
    inspect = {
        "n_images": len(items), "n_readable": len(readable), "unreadable": unreadable[:20],
        "counts": {t: {c: counts[(t, c)] for c in L.CLASSES[t]} for t in L.TASKS},
        "synthetic_gen": {t: {c: synth[(t, c)] for c in L.CLASSES[t] if synth[(t, c)]} for t in L.TASKS},
        "extensions": dict(Counter(Path(it.relpath).suffix.lower() for it in items)), "width": q(w), "height": q(h),
        "aspect_w_over_h": {k: round(float(v), 3) for k, v in zip(("p5", "median", "p95"), np.percentile(w / h, [5, 50, 95]), strict=True)},
        "top_resolutions": res.most_common(12), "distinct_resolutions": len(res),
        "per_class_median_wh": {f"{t}/{c}": [int(np.median(w[ix])), int(np.median(h[ix]))] for (t, c), ix in sorted(cls_idx.items())},
        "flat_frames": int(flat_cs2.sum()),
    }

    # ---- reference sets (hash + thumbnail only)
    sb_dir = guard(a.shotbench_images)
    sb_paths = sorted(p for p in sb_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    refs = {"shotbench": cached_hashes("shotbench", sb_paths, a.workers)}
    from cgt.p3_static_attrs.data import SHOTQA_ROOT, assert_shotqa_path, image_films, image_path
    from cgt.p3b_vlm.data import film_split_ids
    qa_train, qa_val, qa_info = film_split_ids()
    qa_film, _ = image_films(assert_shotqa_path(SHOTQA_ROOT / "meta.jsonl"), qa_train | qa_val)
    for name, ids in (("shotqa_val", qa_val), ("shotqa_train", qa_train)):
        refs[name] = cached_hashes(name, [p for p in (image_path(i) for i in sorted(ids)) if p.exists()], a.workers)

    contam = {"rules": {"candidate_distance": CANDIDATE_DISTANCE, "same_frame_mad": C.MAD_SAME, "same_frame_corr": C.CORR_SAME,
                        "unverified_exclude_distance": UNVERIFIED_EXCLUDE_DISTANCE,
                        "hash": "64-bit dHash, full and black-border-trimmed, min over 4 combinations",
                        "thumbnail": "32x32 grey, MAD min over full/trimmed"},
              "reference_sets": {k: {"listed": v["n_input"], "hashed": len(v["ids"])} for k, v in refs.items()},
              "shotqa_split_info": {k: v for k, v in qa_info.items() if k != "val_films"}}
    flagged: dict[str, set[str]] = {}
    attributed: dict[str, dict[str, set[str]]] = {}
    for name, ref in refs.items():
        pairs = C.candidate_pairs(cs2["full"], cs2["trim"], ref["full"], ref["trim"], CANDIDATE_DISTANCE)
        same, mad, corr = C.same_frame(cs2["thumbs"], ref["thumbs"], pairs)
        flat_pair = C.is_flat(cs2["thumbs"])[pairs[:, 0]] if len(pairs) else np.zeros(0, bool)
        table = []
        for d in range(CANDIDATE_DISTANCE + 1):
            m = pairs[:, 2] == d if len(pairs) else np.zeros(0, bool)
            table.append({"distance": d, "pairs": int(m.sum()), "same_frame": int((m & same).sum()),
                          "flat_collisions": int((m & ~same & flat_pair).sum()),
                          "mad_below_threshold_but_uncorrelated": int((m & (mad < C.MAD_SAME) & ~same).sum()),
                          "median_mad": round(float(np.median(mad[m])), 3) if m.any() else None})
        same_imgs = sorted({rel[i] for i in pairs[same, 0]}) if len(pairs) else []
        close_imgs = sorted({rel[i] for i in pairs[pairs[:, 2] <= UNVERIFIED_EXCLUDE_DISTANCE, 0]}) if len(pairs) else []
        ref_matched = len({int(j) for j in pairs[same, 1]}) if len(pairs) else 0
        films = {}
        if name.startswith("shotqa"):
            for i, j in pairs[same, :2] if len(pairs) else []:
                films.setdefault(qa_film.get(Path(ref["ids"][int(j)]).name) or "?", set()).add(rel[int(i)])
        flagged[name] = set(same_imgs) | set(close_imgs)
        if name.startswith("shotqa"):
            attributed[name] = films
        contam[name] = {
            "pairs_by_distance": table, "cs2_images_same_frame": len(same_imgs),
            "cs2_images_within_unverified_distance": len(close_imgs), "cs2_images_flagged": len(flagged[name]),
            "reference_images_with_same_frame_partner": ref_matched,
            "shotqa_films_with_same_frame_partner": {f: len(v) for f, v in sorted(films.items())} if films else None,
            "flagged_per_class": dict(sorted(Counter("/".join(Path(r).parts[-3:-1]) for r in flagged[name]).items())),
            "same_frame_examples": [(rel[int(i)], Path(ref["ids"][int(j)]).name, int(d), round(float(m), 4), round(float(r), 3))
                                    for (i, j, d), m, r in list(zip(pairs[same], mad[same], corr[same], strict=True))[:12]],
        }
    # film attribution through ShotQA titles: CineScale2 frames that are the same frame as a ShotQA image of a ShotBench film
    sb_titles = [r.split("\t")[1] for r in (ROOT / "docs/data/shotbench_films.tsv").read_text().splitlines() if not r.startswith("#")]
    att_films = {f: v for d in attributed.values() for f, v in d.items() if f != "?"}
    sb_film_hits = {f: m for f, m in ((f, C.title_overlap([f], sb_titles)) for f in sorted(att_films)) if m}
    flagged["shotbench_film_via_shotqa"] = {i for f in sb_film_hits for i in att_films[f]}
    contam["film_attribution_via_shotqa"] = {
        "cs2_films_identified": {f: len(v) for f, v in sorted(att_films.items())},
        "of_which_shotbench_films": {f: [m[1] for m in sb_film_hits[f]] for f in sb_film_hits},
        "cs2_images_attributed_to_shotbench_films": len(flagged["shotbench_film_via_shotqa"]),
        "note": ("Only frames that are the same frame as a titled ShotQA image can be attributed; other frames of the same films "
                 "cannot be found by hashing (different shots). A semantic same-scene check is a Step B gate (plan G1)."),
    }
    contam["title_check"] = ("NOT POSSIBLE: CineScale2 v3 ships no film titles or per-frame metadata (labels are folder names, "
                             "files are '<class>_<n>'); the Data in Brief article (Savardi et al. 2023, PMC10562658) names "
                             "only source types (movies, scraped images, cinematography websites, CineScale, MovieShots). "
                             "ShotBench films: docs/data/shotbench_films.tsv (243 rows from the paper's Appendix C).")

    # ---- CineScale2-internal groups, exclusions, split
    ip = C.candidate_pairs(cs2["full"], cs2["trim"], cs2["full"], cs2["trim"], GROUP_DISTANCE, same_set=True)
    isame, _, _ = C.same_frame(cs2["thumbs"], cs2["thumbs"], ip)
    keep = isame & ~flat_cs2[ip[:, 0]] & ~flat_cs2[ip[:, 1]] if len(ip) else np.zeros(0, bool)
    edges = [(int(i), int(j)) for i, j in ip[keep, :2]]
    group_of = S.groups(rel, edges)
    gsize = Counter(group_of.values())
    task_of = {it.relpath: it for it in readable}
    gtasks: dict[str, set[str]] = {}
    glabels: dict[str, set[str]] = {}
    for i, g in group_of.items():
        gtasks.setdefault(g, set()).add(task_of[i].task)
        glabels.setdefault(g, set()).add(f"{task_of[i].task}/{task_of[i].label}")
    synthetic = {it.relpath for it in readable if it.synthetic}
    conflict_groups = {g for g, v in glabels.items() if len({x.split("/")[0] for x in v}) < len(v)}
    conflicting = {i for i, g in group_of.items() if g in conflict_groups}
    exclude = (flagged["shotbench"] | flagged["shotqa_val"] | flagged["shotbench_film_via_shotqa"] | synthetic | conflicting
               | set(unreadable))
    train, val, info = S.group_split(group_of, VAL_FRAC, SEED, exclude=exclude)

    def by_class(ids):
        c = Counter((task_of[i].task, task_of[i].label) for i in ids)
        return {t: {k: c[(t, k)] for k in L.CLASSES[t]} for t in L.TASKS}

    groups_info = {"candidate_pairs": len(ip), "same_frame_edges": len(edges), "n_groups": len(gsize),
                   "singletons": sum(1 for v in gsize.values() if v == 1), "largest_groups": sorted(gsize.values())[-10:],
                   "groups_spanning_both_tasks": sum(1 for v in gtasks.values() if len(v) > 1),
                   "groups_with_conflicting_labels_same_task": sum(
                       1 for v in glabels.values() if len({x.split("/")[0] for x in v}) < len(v))}
    contam["exclusions"] = {"near_shotbench": len(flagged["shotbench"]), "near_shotqa_val": len(flagged["shotqa_val"]),
                            "shotbench_film_via_shotqa": len(flagged["shotbench_film_via_shotqa"]),
                            "synthetic_gen": len(synthetic), "conflicting_label_groups": len(conflicting), "union": len(exclude),
                            "dropped_with_their_groups": info["dropped_with_group"]}
    split = {
        "version": 1, "dataset": "CineScale2 v3 (doi 10.17632/h4n3gn93gz.3)", "unit": "near-duplicate group (no film ids exist)",
        "rule": (f"groups = union-find over CineScale2-internal same-frame pairs (dHash candidate <= {GROUP_DISTANCE}, thumbnail "
                 f"MAD < {C.MAD_SAME} and corr >= {C.CORR_SAME}, non-flat); groups touching an excluded id are dropped in full; groups sorted by smallest "
                 f"relpath, numpy default_rng({SEED}) permutation, first round(n_groups*{VAL_FRAC}) groups = val; every other "
                 "non-excluded image = train"),
        "info": info, "groups": groups_info, "train_counts": by_class(train), "val_counts": by_class(val),
        "excluded": {"near_shotbench": sorted(flagged["shotbench"]), "near_shotqa_val": sorted(flagged["shotqa_val"]),
                     "shotbench_film_via_shotqa": sorted(flagged["shotbench_film_via_shotqa"]),
                     "synthetic_gen": sorted(synthetic), "conflicting_labels": sorted(conflicting), "unreadable": unreadable,
                     "dropped_with_group": sorted(set(group_of) - set(train) - set(val) - exclude)},
        "val": val,
    }
    out = ROOT / "results/cinescale2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inspect.json").write_text(json.dumps(inspect, indent=1) + "\n")
    (out / "contamination.json").write_text(json.dumps(contam, indent=1) + "\n")
    (out / "split_v1.json").write_text(json.dumps(split, indent=0) + "\n")
    if a.g1_flags:
        split2 = split_v2(split, json.loads(Path(a.g1_flags).read_text()), group_of, by_class)
        (out / "split_v2.json").write_text(json.dumps(split2, indent=0) + "\n")
        print(json.dumps({k: split2[k] for k in ("g1", "info", "train_counts", "val_counts")}, indent=1))
        if split2["g1"]["angle_train_removed_fraction"] > G1_STOP_FRACTION:
            print("STOP: G1 removes more than 25% of the v1 angle train images")
            sys.exit(12)
        if a.a2_flags:
            split3 = split_v3(split2, json.loads(Path(a.a2_flags).read_text()), group_of, by_class)
            (out / "split_v3.json").write_text(json.dumps(split3, indent=0) + "\n")
            print(json.dumps({k: split3[k] for k in ("a2", "info", "train_counts", "val_counts")}, indent=1))
        return
    print(json.dumps({"flagged": {k: len(v) for k, v in flagged.items()}, "attribution": contam["film_attribution_via_shotqa"],
                      "same_frame": {k: contam[k]["cs2_images_same_frame"] for k in refs},
                      "exclusions": contam["exclusions"], "split": info, "groups": groups_info,
                      "train_counts": split["train_counts"], "val_counts": split["val_counts"]}, indent=1))


if __name__ == "__main__":
    main()
