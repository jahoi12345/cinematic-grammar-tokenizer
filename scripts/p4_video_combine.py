"""Out-of-fold analysis of the video cross-fit (docs/phase4_video_plan.md Amendment 2 items 2 and 6), TRAIN sources only, no validation label: (1) the per-epoch pooled OOF option
accuracy of the video model and the epoch rule E (highest pooled OOF accuracy, ties to the smaller E); (2) the geometry OOF accuracy on the same questions; (3) the two combination weights
(w_s, w_g) on the grid {0, 0.1, ..., 1}^2 maximising the OOF conditional log-likelihood of the gold option of
`score(option) = w_s * log_softmax(letter logits)[option] + w_g * Bernoulli atom-set log-likelihood of the geometry model`, ties to the smaller w_s + w_g. CPU.
Writes results/phase4/runs/video_<tag>/combine.{json,md}. Usage: uv run python scripts/p4_video_combine.py --tag v1 [--root DIR]"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from cgt.p4_camera_move import crossfit as X
from cgt.p4_camera_move.baseline import option_scores, wilson
from cgt.p4_camera_move.labels import MovementLabelStore

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCD"
GRID = [round(0.1 * i, 1) for i in range(11)]


def lsm(logits) -> np.ndarray:
    z = np.array(logits, dtype=np.float64)
    z = z - z.max()
    return z - math.log(np.exp(z).sum())


def build_questions(rows_by_clip: dict, oof: dict[int, dict[str, dict]], geom: dict, epoch: int) -> list[dict]:
    """One record per OOF question at `epoch`: gold index, semantic log-softmax per letter, geometry set log-likelihood per letter."""
    atoms, probs = geom["atoms"], geom["probs"]
    out = []
    for clip, rec in oof[epoch].items():
        r = rows_by_clip[clip]
        gs = option_scores(r, np.array(probs[clip]), atoms)
        out.append({"clip": clip, "source": r["source"], "gold": LETTERS.index(r["gold_letter"]), "sem": lsm(rec["logits"]), "geo": np.array([gs[c] for c in LETTERS]), "compound": r["kind"] != "single"})
    return out


def cond_ll(qs: list[dict], ws: float, wg: float) -> float:
    tot = 0.0
    for q in qs:
        s = ws * q["sem"] + wg * q["geo"]
        s = s - s.max()
        tot += float(s[q["gold"]] - math.log(np.exp(s).sum()))
    return tot


def acc(qs, ws, wg) -> np.ndarray:
    return np.array([int(np.argmax(ws * q["sem"] + wg * q["geo"])) == q["gold"] for q in qs])  # ties: the lowest letter (argmax picks the first)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--root", default=str(ROOT / "results/phase4/runs"))
    args = ap.parse_args()
    base = Path(args.root) / f"video_{args.tag}"
    man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    rows = {r["clip"]: r for r in MovementLabelStore(man["clips"]["val"]).rows(man["clips"]["train"]) if r["atoms"]}  # TRAIN labels only
    geom = json.loads((base / "geom_oof.json").read_text())
    oof = {e: {} for e in (1, 2, 3)}
    per_fold = {}
    for k in range(X.N_FOLDS):
        for e in (1, 2, 3):
            d = json.loads((base / f"fold{k}" / f"oof_e{e}.json").read_text())
            per_fold[(k, e)] = (d["acc"], d["n"])
            for r in d["records"]:
                oof[e][r["clip"]] = r
    assert all(len(oof[e]) == len(rows) for e in oof), "every train clip must be held out exactly once per epoch"
    hits = {e: [oof[e][c]["pred"] == oof[e][c]["gold"] for c in sorted(rows)] for e in oof}
    big_e, accs = X.choose_epoch(hits)
    res = {"n_oof_questions": len(rows), "epoch_accuracy_pooled": {str(e): round(a, 4) for e, a in accs.items()}, "selected_epoch_E": big_e,
           "per_fold_epoch_accuracy": {f"fold{k}/e{e}": {"acc": round(v[0], 4), "n": v[1]} for (k, e), v in per_fold.items()},
           "epoch_ci95_wilson": {str(e): [round(x, 1) for x in wilson(sum(hits[e]), len(hits[e]))] for e in hits}, "chance": 0.25}
    qs = build_questions(rows, oof, geom, big_e)
    grid = {(ws, wg): cond_ll(qs, ws, wg) for ws, wg in itertools.product(GRID, GRID)}
    best = max(grid.values())
    ws, wg = min((k for k, v in grid.items() if v >= best - 1e-12), key=lambda k: (k[0] + k[1], k))
    a_sem, a_geo, a_comb = acc(qs, 1, 0), acc(qs, 0, 1), acc(qs, ws, wg)
    src = np.array([q["source"] for q in qs])
    usrc = sorted(set(src))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([np.where(src == s)[0] for s in rng.choice(usrc, len(usrc), replace=True)]) for _ in range(1000)]

    def paired(x, y):
        d = [x[ix].mean() - y[ix].mean() for ix in boots]
        return {"diff": round(float(x.mean() - y.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    res.update({"weights": {"w_s": ws, "w_g": wg, "oof_conditional_ll": round(grid[(ws, wg)], 3), "ll_chance": round(len(qs) * math.log(0.25), 3), "ll_semantic_only": round(grid[(1.0, 0.0)], 3), "ll_geometry_only": round(grid[(0.0, 1.0)], 3)},
                "oof_option_accuracy_at_E": {"semantic_only": {"acc": round(float(a_sem.mean()), 4), "ci95": [round(x, 1) for x in wilson(int(a_sem.sum()), len(qs))]},
                                             "geometry_only": {"acc": round(float(a_geo.mean()), 4), "ci95": [round(x, 1) for x in wilson(int(a_geo.sum()), len(qs))]},
                                             "combined": {"acc": round(float(a_comb.mean()), 4), "ci95": [round(x, 1) for x in wilson(int(a_comb.sum()), len(qs))]}},
                "oof_paired_bootstrap_by_source_1000": {"combined_minus_semantic": paired(a_comb, a_sem), "combined_minus_geometry": paired(a_comb, a_geo), "semantic_minus_geometry": paired(a_sem, a_geo)},
                "n_sources": len(usrc), "note": "development estimate on TRAIN sources (out-of-fold); weights are frozen from this and nothing else; validation sources untouched"})
    (base / "combine.json").write_text(json.dumps(res, indent=1))
    L = [f"# Video cross-fit out-of-fold analysis `{args.tag}` (TRAIN sources, 3 folds, {len(rows)} questions, {len(usrc)} sources)", "",
         "| epoch | pooled OOF option accuracy | 95% Wilson |", "|---|---:|---|"]
    L += [f"| {e} | {accs[e]:.4f} | {res['epoch_ci95_wilson'][str(e)]} |" for e in (1, 2, 3)]
    L += ["", f"**Selected epoch E = {big_e}** (highest pooled OOF accuracy, ties to the smaller E); chance 0.25.", "", "| fold / epoch | accuracy | n |", "|---|---:|---:|"]
    L += [f"| fold{k} / e{e} | {per_fold[(k, e)][0]:.4f} | {per_fold[(k, e)][1]} |" for k in range(X.N_FOLDS) for e in (1, 2, 3)]
    o = res["oof_option_accuracy_at_E"]
    p = res["oof_paired_bootstrap_by_source_1000"]
    L += ["", f"## Combination weights (grid, OOF conditional log-likelihood): w_s = {ws}, w_g = {wg}", "",
          f"log-likelihood: combined {res['weights']['oof_conditional_ll']}, semantic only {res['weights']['ll_semantic_only']}, geometry only {res['weights']['ll_geometry_only']}, chance {res['weights']['ll_chance']}", "",
          "| scorer (epoch E) | OOF option accuracy | 95% Wilson |", "|---|---:|---|"]
    L += [f"| {k} | {v['acc']:.4f} | {v['ci95']} |" for k, v in o.items()]
    L += ["", "Paired bootstrap (resampling the sources, 1,000 resamples), accuracy difference [95% interval]:", ""]
    L += [f"- {k}: {v['diff']:+.4f} {v['ci95']}" for k, v in p.items()]
    (base / "combine.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
