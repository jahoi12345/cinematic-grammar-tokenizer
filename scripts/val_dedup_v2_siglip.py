"""Phase 3A SigLIP option scores (Step A C5 input) for the 60 rows dedup v2 adds, CPU, with the exact scripts/p3s_stepa_siglip.py head code.
Parity: the same code re-scores the first-occurrence partners and must reproduce data/cache/p3s_stepa/siglip3a.jsonl.
Writes data/cache/val_dedup_v2/siglip3a_added.jsonl ({key, scores}; no gold). Usage: nice -n 19 <venv>/bin/python scripts/val_dedup_v2_siglip.py"""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.eval import metrics as M
from cgt.eval.val_dedup import dedup_v2_indices
from cgt.p3_static_attrs import features, heads
from cgt.p3_static_attrs.data import SHOTQA_ROOT
from cgt.p3_static_attrs.labels import parse_label_set, vocab
from cgt.p3s_stepa.split import ATTRS, dedup_first
from cgt.p3s_stepa.views import LETTERS


def main() -> None:
    spec = spec_from_file_location("p3s_stepa_siglip", ROOT / "scripts/p3s_stepa_siglip.py")
    S = module_from_spec(spec)
    spec.loader.exec_module(S)
    chosen = json.loads((ROOT / f"results/phase3/runs/{S.TAG}/chosen_heads_{S.TAG}.json").read_text())
    cache = ROOT / "data/cache/p3" / features.slug(features.MODEL)
    features.verify_cache(cache, features.MODEL, SHOTQA_ROOT)
    ids, feats = features.load_features(cache)
    fidx = {i: n for n, i in enumerate(ids)}
    X = heads.l2norm(feats)
    rows = [json.loads(x) for x in (ROOT / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    old = dedup_first(rows)
    first = {q["line"] for q in old}
    added = [{**rows[n], "key": f"{n}:{rows[n]['id']}"} for n in dedup_v2_indices(rows) if n not in first]
    by_id = {q["id"]: q for q in old}
    partners = [by_id[q["id"]] for q in added]
    ref = {r["key"]: r["scores"] for r in map(json.loads, (ROOT / "data/cache/p3s_stepa/siglip3a.jsonl").read_text().splitlines())}

    def score(qs):
        out = {}
        for attr in ATTRS:
            v = vocab(attr)
            vi = {c: i for i, c in enumerate(v)}
            aq = [q for q in qs if q["attribute"] == attr]
            if not aq:
                continue
            p, kind = S.head_outputs(attr, chosen[attr], torch.from_numpy(X[[fidx[q["img_id"]] for q in aq]]), ROOT / "models/p3" / S.TAG)
            for q, row in zip(aq, p, strict=True):
                opt = [[vi[c] for c in parse_label_set(attr, q["options"][x])] for x in LETTERS]
                sc = M.multihot_option_scores(row, opt) if kind == "multihot" else M.categorical_option_scores(row, opt)
                out[q["key"]] = [round(float(s), 5) for s in sc]
        return out

    par = score(partners)
    diff = max(float(np.abs(np.array(par[k]) - np.array(ref[k])).max()) for k in par)
    print(f"parity on {len(par)} partner rows: max |diff| {diff}")
    assert diff < 1e-4
    sc = score(added)
    assert len(sc) == 60
    (ROOT / "data/cache/val_dedup_v2/siglip3a_added.jsonl").write_text("\n".join(json.dumps({"key": k, "scores": v}) for k, v in sc.items()) + "\n")
    print(f"{len(sc)} added rows scored")


if __name__ == "__main__":
    main()
