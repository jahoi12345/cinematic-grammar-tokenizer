"""Recompute the single-label / label-set answer strata of the recorded ShotBench evaluation from the COMMITTED predictions with the corrected rule
(a label set has more than one DISTINCT label, as parsed by `parse_label_set`; "Long Lens, Long Lens" is one label). Reads the committed predictions and the
ShotBench question file (question/option text only); changes nothing: no scoring, no ledger, no model, no adapter, no committed result.
Writes results/phase3b/strata_corrected.json. Usage: uv run python scripts/p3b_strata_erratum.py [--tsv PATH] [--image-root PATH]"""
import argparse
import json
from pathlib import Path

import numpy as np

from cgt.p3_static_attrs.labels import parse_label_set
from cgt.p3b_vlm.shotbench import load_shotbench_samples

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ["size", "framing", "angle", "lens", "light_type", "light_cond", "composition"]
PRED = ROOT / "results/phase3b/raw_eval_logs/qwen3vl2b_lora_filmsplit.predictions.jsonl"


def strata(hit: np.ndarray, comp: np.ndarray) -> dict:
    return {"n": len(hit), "n_label_set": int(comp.sum()), "correct": int(hit.sum()),
            "acc_single_pct": round(100 * float(hit[~comp].mean()), 1) if (~comp).any() else None,
            "acc_label_set_pct": round(100 * float(hit[comp].mean()), 1) if comp.any() else None,
            "single_correct": int(hit[~comp].sum()), "single_n": int((~comp).sum()), "set_correct": int(hit[comp].sum()), "set_n": int(comp.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=str(ROOT / "data/external/shotbench/test.tsv"))
    ap.add_argument("--image-root", default=str(ROOT / "data/external/shotbench"))
    a = ap.parse_args()
    samples, _ = load_shotbench_samples(Path(a.tsv), Path(a.image_root))
    by_id = {s["id"]: s for s in samples}
    preds = [json.loads(x) for x in PRED.read_text().splitlines()]
    out = {"n_predictions": len(preds), "per_attribute": {}, "changed_questions": []}
    for at in ATTRS:
        r = [p for p in preds if p["attribute"] == at]
        hit = np.array([p["pred"] == p["gold"] for p in r])
        old = np.array([p["compound"] for p in r])
        new = np.array([len(parse_label_set(at, by_id[p["id"]]["options"][p["gold"]])) > 1 for p in r])
        for p, o, n in zip(r, old, new, strict=True):
            if o != n:
                out["changed_questions"].append({"id": p["id"], "attribute": at, "old_label_set": bool(o), "new_label_set": bool(n),
                                                 "answer_text": by_id[p["id"]]["options"][p["gold"]]})
        out["per_attribute"][at] = {"old": strata(hit, old), "new": strata(hit, new)}
    (ROOT / "results/phase3b/strata_corrected.json").write_text(json.dumps(out, indent=1))
    for at in ATTRS:
        o, n = out["per_attribute"][at]["old"], out["per_attribute"][at]["new"]
        print(f"{at:12s} old: sets {o['n_label_set']:3d} single {o['acc_single_pct']} set {o['acc_label_set_pct']} | new: sets {n['n_label_set']:3d} single {n['acc_single_pct']} set {n['acc_label_set_pct']}")
    print("changed questions:", len(out["changed_questions"]))


if __name__ == "__main__":
    main()
