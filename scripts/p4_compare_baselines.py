"""Compare the Phase 4 atom baselines on the SAME source-grouped validation sources: macro F1 / AP, option-restricted accuracy, and the tilt-vs-boom and pan-vs-move confusion
rates, with paired bootstrap intervals (resampling validation SOURCES) for differences against a reference run. Validation is the only evaluation; no ShotBench or CameraBench.
Usage: uv run python scripts/p4_compare_baselines.py --ref flow --runs flow,flow_depth,siglip,flow_depth_siglip   -> results/phase4/baseline_comparison.md"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cgt.p4_camera_move.baseline import record_look
from cgt.p4_camera_move.labels import MovementLabelStore

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"tilt vs boom": [("tilt-up", "boom-up"), ("boom-up", "tilt-up"), ("tilt-down", "boom-down"), ("boom-down", "tilt-down")],
          "pan vs move": [("pan-left", "move-left"), ("move-left", "pan-left"), ("pan-right", "move-right"), ("move-right", "pan-right")],
          "push/pull vs zoom": [("push-in", "zoom-in"), ("pull-out", "zoom-out")], "push vs pull": [("push-in", "pull-out"), ("pull-out", "push-in")]}


AUC_GROUPS = {"AUC tilt/boom": [("tilt-up", "boom-up"), ("tilt-down", "boom-down")], "AUC pan/move": [("pan-left", "move-left"), ("pan-right", "move-right")],
              "AUC push/pull": [("push-in", "pull-out")]}


def load(tag):
    v = json.loads((ROOT / f"results/phase4/val_probs_{tag}.json").read_text())
    return v["atoms"], v["clips"], np.array(v["probs"])


def metrics(Y, P, idx, atoms):
    f1s, aps = [], []
    for j in range(len(atoms)):
        y, p = Y[idx, j], P[idx, j]
        if y.sum() == 0:
            continue
        pred = p >= 0.5
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        aps.append(float(average_precision_score(y, p)))
    out = {"macro_f1": float(np.mean(f1s)), "macro_ap": float(np.mean(aps))}
    ai = {a: i for i, a in enumerate(atoms)}
    for name, pairs in GROUPS.items():
        num = den = 0
        for g, b in pairs:
            if g in ai and b in ai:
                gg, gb, pb = Y[idx, ai[g]], Y[idx, ai[b]], P[idx, ai[b]] >= 0.5
                num += int(((gg == 1) & (gb == 0) & pb).sum())
                den += int(gg.sum())
        out[name] = num / den if den else float("nan")
    for name, pairs in AUC_GROUPS.items():  # threshold-free: among clips with exactly one of A, B in the gold set, how often is p_A - p_B larger for the A clips
        vals = []
        for g, b in pairs:
            if g in ai and b in ai:
                ga, gb = Y[idx, ai[g]], Y[idx, ai[b]]
                m = (ga + gb) == 1
                if m.sum() and len(set(ga[m])) == 2:
                    vals.append(float(roc_auc_score(ga[m], P[idx, ai[g]][m] - P[idx, ai[b]][m])))
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="flow")
    ap.add_argument("--runs", default="flow,flow_depth,siglip,flow_depth_siglip")
    args = ap.parse_args()
    runs = args.runs.split(",")
    manifest = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    # reading the validation labels is a LOOK (contract section 4): record it BEFORE the labels are read, then use the locked store with the explicit final flag
    cfgs = [json.loads((ROOT / f"results/phase4/frozen_config_{t}.json").read_text()) for t in runs]
    record_look(ROOT / "results/phase4/validation_looks.jsonl", kind="comparison_report", tag="comparison:" + "+".join(runs), feature_sets=runs,
                config_hash="+".join(c.get("config_hash", "") for c in cfgs))
    rows = {r["clip"]: r for r in MovementLabelStore(manifest["clips"]["val"]).rows(manifest["clips"]["val"], final_evaluation=True)}
    data = {t: load(t) for t in runs}
    atoms, clips, _ = data[runs[0]]
    for t in runs:
        assert data[t][0] == atoms and data[t][1] == clips, f"{t}: different atoms or clips"
    Y = np.array([[a in rows[c]["atoms"] for a in atoms] for c in clips], dtype=int)
    src = np.array([rows[c]["source"] for c in clips])
    usrc = sorted(set(src))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([np.where(src == s)[0] for s in rng.choice(usrc, len(usrc), replace=True)]) for _ in range(1000)]
    full = np.arange(len(clips))
    base = {t: metrics(Y, data[t][2], full, atoms) for t in runs}
    bb = {t: [metrics(Y, data[t][2], ix, atoms) for ix in boots] for t in runs}
    keys = ["macro_f1", "macro_ap", *GROUPS, *AUC_GROUPS]
    L = [f"# Baseline comparison on the same source-grouped validation sources ({len(clips)} clips, {len(usrc)} sources; reference run `{args.ref}`)", "",
         ("Every run used the identical split, atoms, procedure (per-atom logistic regression, C chosen by source-grouped 5-fold CV on train sources only, threshold 0.5) and each run is one logged validation look (`results/phase4/validation_looks.jsonl`); the numbers are descriptive and feature sets are selected by the train-source CV only. "
         "Differences are paired bootstrap (1,000 resamples of validation SOURCES, seed 0), 95% percentile interval of (run - reference); with 19 sources these intervals are wide. Confusion rates = share of gold-A clips predicted as B (and gold lacking B), pooled over the listed pairs (lower is better, but a model that rarely predicts those atoms also has a low rate: read it together with macro F1). The AUC columns are threshold-free: among validation clips whose gold has exactly one of A, B, the probability that p_A - p_B ranks an A clip above a B clip (0.5 = chance; mean over the listed direction pairs)."), "",
         "| run | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for t in runs:
        L.append(f"| {t} | " + " | ".join(f"{base[t][k]:.3f}" for k in keys) + " |")
    cvs = {t: json.loads((ROOT / f"results/phase4/frozen_config_{t}.json").read_text()) for t in runs}
    L += ["", "## Selection criterion: train-source CV macro AP at the chosen C (the ONLY quantity feature sets are selected on; validation numbers above are descriptive)", "",
          "| run | features | chosen C | train-source CV macro AP |", "|---|---|---:|---:|"]
    L += [f"| {t} | {'+'.join(cvs[t]['feature_sets'])} ({cvs[t]['n_features']}) | {cvs[t]['chosen_C']} | {cvs[t]['cv_macro_ap_at_chosen_C']:.3f} |" for t in runs]
    oofj = {t: json.loads((ROOT / f"results/phase4/baseline_atoms_{t}.json").read_text()) for t in runs}
    L += ["", "## Development estimate: train-source out-of-fold confusion rates (903 train clips, same procedure; the quantity feature sets are selected on, with the CV macro AP above)", "",
          "| run | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    for t in runs:
        cells = []
        for pairs in GROUPS.values():
            sel = [c for c in oofj[t]["confusions_train_oof"] if (c["gold"], c["predicted"]) in pairs]
            cells.append(f"{sum(c['count'] for c in sel) / max(1, sum(c['n_gold'] for c in sel)):.3f} ({sum(c['count'] for c in sel)}/{sum(c['n_gold'] for c in sel)})")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", f"## Difference to `{args.ref}` (run minus reference; 95% paired bootstrap interval; * = interval excludes 0)", "", "| run | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for t in runs:
        if t == args.ref:
            continue
        cells = []
        for k in keys:
            d = np.array([x[k] - y[k] for x, y in zip(bb[t], bb[args.ref], strict=True)])
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            cells.append(f"{base[t][k] - base[args.ref][k]:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    (ROOT / "results/phase4/baseline_comparison.md").write_text("\n".join(L) + "\n")
    (ROOT / "results/phase4/baseline_comparison.json").write_text(json.dumps({"base": base, "n_sources": len(usrc), "n_clips": len(clips)}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
