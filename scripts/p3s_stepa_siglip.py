"""Step A static track, C5 input: frozen Phase 3A SigLIP head option scores per deduplicated validation question (CPU, no GPU lock).
Exactly the Phase 3A scoring (cgt.eval.shotbench.evaluate): chosen head kind per attribute, mean of the head outputs over seeds 0,1,2, then
multi-hot set log-likelihood (or log mean class probability for a categorical size head). Writes data/cache/p3s_stepa/siglip3a.jsonl
({key, scores in original option order}; no gold). Reads only ShotQA features and the saved heads.
Usage: uv run python scripts/p3s_stepa_siglip.py"""
import json
from pathlib import Path

import numpy as np
import torch

from cgt.eval import metrics as M
from cgt.p3_static_attrs import features, heads
from cgt.p3_static_attrs.data import SHOTQA_ROOT
from cgt.p3_static_attrs.labels import parse_label_set, vocab
from cgt.p3s_stepa.split import ATTRS, dedup_first
from cgt.p3s_stepa.views import LETTERS

ROOT = Path(__file__).resolve().parents[1]
TAG = "siglip_base_filmsplit_lr1e-2_ep300"
SEEDS = (0, 1, 2)


def head_outputs(attr: str, kind_name: str, x: torch.Tensor, model_dir: Path) -> tuple[np.ndarray, str]:
    v = vocab(attr)
    preds = []
    for s in SEEDS:
        if attr == "size" and kind_name in {"softmax", "coral", "softmax_mlp", "coral_mlp"}:
            kind, hidden = kind_name.split("_")[0], 512 if kind_name.endswith("_mlp") else 0
        else:
            kind, hidden = "multihot", (512 if kind_name == "mlp" else 0)
        h = heads.Head(x.shape[1], len(v), kind, hidden)
        h.load_state_dict(torch.load(model_dir / f"{attr}_{kind_name}_s{s}.pt"))
        h.eval()
        preds.append(h.predict(x))
    return np.mean(preds, axis=0), ("multihot" if kind == "multihot" else "categorical")


def main() -> None:
    chosen = json.loads((ROOT / f"results/phase3/runs/{TAG}/chosen_heads_{TAG}.json").read_text())
    cache = ROOT / "data/cache/p3" / features.slug(features.MODEL)
    features.verify_cache(cache, features.MODEL, SHOTQA_ROOT)  # every cached feature was read from inside ShotQA
    ids, feats = features.load_features(cache)
    fidx = {i: n for n, i in enumerate(ids)}
    X = heads.l2norm(feats)
    qs = dedup_first([json.loads(x) for x in (ROOT / "data/cache/p3b/val.jsonl").read_text().splitlines()])
    out = []
    for attr in ATTRS:
        v = vocab(attr)
        vi = {c: i for i, c in enumerate(v)}
        aq = [q for q in qs if q["attribute"] == attr]
        p, kind = head_outputs(attr, chosen[attr], torch.from_numpy(X[[fidx[q["img_id"]] for q in aq]]), ROOT / "models/p3" / TAG)
        for q, row in zip(aq, p, strict=True):
            opt = [[vi[c] for c in parse_label_set(attr, q["options"][x])] for x in LETTERS]
            sc = M.multihot_option_scores(row, opt) if kind == "multihot" else M.categorical_option_scores(row, opt)
            out.append({"key": q["key"], "scores": [round(float(s), 5) for s in sc]})
    dst = ROOT / "data/cache/p3s_stepa/siglip3a.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"{len(out)} questions scored with {TAG} heads { {a: chosen[a] for a in ATTRS} }")


if __name__ == "__main__":
    main()
