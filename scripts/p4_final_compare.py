"""Paired comparison of the three models on the SAME questions of one final-evaluation dataset (docs/phase4_video_plan.md section 8): geometry vs video vs combined, accuracy differences with a
95% bootstrap interval that resamples SOURCE VIDEOS. Reads only the per-question predictions that the sealed evaluator PUBLISHED after each ledger commit (`raw/<dataset>__<tag>.predictions.jsonl`);
it opens no label file and no ledger and runs on the CPU. It refuses when a prediction file is missing or when the three models were not scored on identical questions (same ids and option texts in
the same order, same gold letters). Variants: every variant present for all three models; for ShotBench movement also the pre-declared source-disjoint subset.
Usage: uv run python scripts/p4_final_compare.py --dataset shotbench_movement [--raw-dir DIR] [--out-dir DIR]"""
import argparse
import json
from pathlib import Path

import numpy as np

from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_run as FR

PAIRS = [("combined", "geometry"), ("combined", "video"), ("video", "geometry")]


def load(raw: Path, dataset: str, model: str) -> dict[str, list[dict]]:
    p = Path(raw) / f"{FE.key_of(dataset, FE.MODEL_TAGS[model])}.predictions.jsonl"
    if not p.exists():
        raise SystemExit(f"missing predictions of the {model} model: {p.name} (the evaluation has not been run)")
    out: dict[str, list[dict]] = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["variant"], []).append(r)
    return out


def questions_key(rows: list[dict]) -> str:
    return FE.questions_digest([{"id": r["id"], "clip": r["clip"], "options": r["options"], "gold_letter": r["gold_letter"]} for r in rows], with_gold=True)


def compare(preds: dict[str, dict[str, list[dict]]], reference_sources: set[str] | None = None) -> dict:
    """`preds[model][variant] = rows`. Returns {variant: {accuracy, paired, subset?}} for every variant that exists for all three models."""
    out: dict = {}
    variants = set.intersection(*(set(p) for p in preds.values()))
    for v in sorted(variants):
        keys = {m: questions_key(preds[m][v]) for m in preds}
        if len(set(keys.values())) != 1:
            raise SystemExit(f"variant {v}: the three models were not scored on identical questions: {keys}")
        rows = preds["geometry"][v]
        src = [r["source"] for r in rows]
        hits = {m: np.array([r["hit"] for r in preds[m][v]], dtype=bool) for m in preds}
        entry = {"n_questions": len(rows), "n_sources": len(set(src)), "accuracy": {m: round(float(h.mean()), 4) for m, h in hits.items()},
                 "paired_bootstrap_by_source_1000": {f"{a}_minus_{b}": FE.paired_bootstrap(hits[a], hits[b], src) for a, b in PAIRS}}
        if reference_sources is not None and v == "main":
            keep = np.array([s not in reference_sources for s in src])
            if keep.any():
                s2 = [s for s, k in zip(src, keep, strict=True) if k]
                entry["source_disjoint_subset"] = {"declared_secondary": True, "n_questions": int(keep.sum()), "n_sources": len(set(s2)),
                                                   "accuracy": {m: round(float(h[keep].mean()), 4) for m, h in hits.items()},
                                                   "paired_bootstrap_by_source_1000": {f"{a}_minus_{b}": FE.paired_bootstrap(hits[a][keep], hits[b][keep], s2) for a, b in PAIRS}}
        out[v] = entry
    return out


def render(dataset: str, res: dict) -> str:
    L = [f"# Paired comparison on `{dataset}` (same questions, bootstrap by source video, 1,000 resamples)", ""]
    for v, e in res.items():
        L += [f"## variant `{v}`{' (primary)' if v == 'main' else ' (declared secondary)'}: {e['n_questions']} questions, {e['n_sources']} sources", "",
              "accuracy: " + ", ".join(f"{m} {a:.4f}" for m, a in e["accuracy"].items()), ""]
        L += [f"- {k}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" for k, d in e["paired_bootstrap_by_source_1000"].items()]
        if "source_disjoint_subset" in e:
            s = e["source_disjoint_subset"]
            L += ["", f"source-disjoint subset (declared secondary, small): {s['n_questions']} questions, {s['n_sources']} sources; accuracy " + ", ".join(f"{m} {a:.4f}" for m, a in s["accuracy"].items())]
            L += [f"- {k}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" for k, d in s["paired_bootstrap_by_source_1000"].items()]
        L.append("")
    L.append("Allowed statements: differences among OUR three models on these questions with the intervals above. Not allowed: superiority over any published model, general or "
             "unseen-film claims (docs/phase4_video_plan.md section 9).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=FE.DATASETS, required=True)
    ap.add_argument("--raw-dir", default=str(FR.RAW_DIR))
    ap.add_argument("--out-dir", default=str(FR.OUT_DIR))
    a = ap.parse_args()
    preds = {m: load(Path(a.raw_dir), a.dataset, m) for m in FE.MODELS}
    ref = None
    if a.dataset == "shotbench_movement":
        from cgt.p4_camera_move.labels import load_shotqa_movement_meta
        ref = {m["source"] for m in load_shotqa_movement_meta()}
    res = compare(preds, ref)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.dataset}_paired.json").write_text(json.dumps(res, indent=1))
    (out / f"{a.dataset}_paired.md").write_text(render(a.dataset, res))
    print(render(a.dataset, res))
