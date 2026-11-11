"""Validation-subset construction, metrics and the FIXED selection rule of docs/phase3c_plan.md sections 4-6. Pure python/numpy."""
import numpy as np

from cgt.p3b_vlm.data import load_chat_samples
from cgt.p3c_tokens import aggregate as ag
from cgt.p3c_tokens.methods import PASS_RANK
from cgt.p3c_tokens.vocab import ATTRS, MULTI, SINGLE, labels

PER_ATTR = 214
TIE_PP = 1.0


def stratified_subset(samples: list[dict], per_attr: int = PER_ATTR, seed: int = 0) -> list[dict]:
    """Plan section 4: for each attribute in ATTRS order, the first `per_attr` of rng.permutation(indices of that attribute's questions)."""
    rng = np.random.default_rng(seed)
    out = []
    for a in ATTRS:
        idx = [i for i, s in enumerate(samples) if s["attribute"] == a]
        out += [samples[idx[j]] for j in rng.permutation(len(idx))[:per_attr]]
    return out


def validation_subset() -> list[dict]:
    return stratified_subset(load_chat_samples("val"))


def gold_labels(sample: dict) -> set[str]:
    from cgt.p3_static_attrs.labels import parse_label_set
    return set(parse_label_set(sample["attribute"], sample["options"][sample["answer"]]))


def option_label_indices(sample: dict) -> dict[str, list[int]]:
    from cgt.p3_static_attrs.labels import parse_label_set
    labs = labels(sample["attribute"])
    return {k: [labs.index(x) for x in sorted(parse_label_set(sample["attribute"], v))] for k, v in sample["options"].items()}


def evaluate_method(samples: list[dict], scores: dict[tuple[str, str], np.ndarray]) -> dict:
    """samples: the subset; scores: (sample id, attribute) -> score vector over the full vocabulary (schema order). Method scores depend only on
    (image, attribute), so the few sft.json images that are asked twice (same id) share one score vector. Returns per-attribute metrics (plan section 5)."""
    per = {}
    for a in ATTRS:
        rows = [s for s in samples if s["attribute"] == a and (s["id"], a) in scores]
        labs = labels(a)
        gold = [gold_labels(s) for s in rows]
        sc = [scores[(s["id"], a)] for s in rows]
        prim = [labs[int(np.argmax(x))] for x in sc]
        full = float(np.mean([p in g for p, g in zip(prim, gold)]))
        opt = []
        for s, x in zip(rows, sc):
            o = ag.option_scores(x, option_label_indices(s))
            best = max(sorted(o), key=lambda k: o[k])  # ties: lowest letter
            opt.append(best == s["answer"])
        r = {"n": len(rows), "full_acc": full, "option_acc": float(np.mean(opt)),
             "metric": "acc" if a in SINGLE else "setF1", "value": full}
        if a in MULTI:
            alpha, f1 = ag.pick_alpha(sc, gold, labs)
            r.update({"alpha": alpha, "set_f1": f1, "value": f1})
        per[a] = r
    comp = float(np.mean([per[a]["value"] for a in ATTRS if per[a]["n"]]))
    return {"per_attribute": per, "composite": comp}


def select(results: dict[str, dict]) -> dict:
    """Plan section 6. results: method -> evaluate_method output. Highest composite; within TIE_PP points of it, fewest passes, then 'm' over 'b'."""
    best = max(r["composite"] for r in results.values())
    tied = [m for m, r in results.items() if best - r["composite"] <= TIE_PP / 100 + 1e-12]
    pick = min(tied, key=lambda m: (PASS_RANK[m], m.endswith("b"), m))
    return {"best_composite": best, "within_tie_band": sorted(tied), "selected": pick,
            "top_by_composite": max(results, key=lambda m: results[m]["composite"])}
