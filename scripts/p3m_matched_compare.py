"""Comparison fixed in a decision record for the sealed matched ShotBench run (tag matched_v1). Reads ONLY committed outputs of the sealed
evaluator (results/p3m_matched/raw_eval_logs/<tag>.predictions.jsonl and results/p3m_matched/shotbench_<tag>.json) and the Phase 3B letter-logit
CSV; never touches ShotBench. Per-attribute accuracy (invalid = wrong), unweighted 7-attribute mean, 95% percentile intervals from 10,000 bootstrap
resamples of questions stratified by attribute (numpy default_rng seed 0, attributes in ATTRS order), invalid rate, and the expected-random-fallback
variant. Wording rule (0018): ABOVE only if lo > published, BELOW if hi < published, otherwise COMPARABLE.
for a letter-logit protocol summary (letterlogit_4b_c5_v1) the 0018 rule does not apply: every verdict field is written as
"n/a (letter-logit protocol)". --root (default: the repository) only relocates the inputs/output (synthetic tests).
Usage: PYTHONPATH=src python scripts/p3m_matched_compare.py [--tag matched_v1]"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ["size", "framing", "angle", "lens", "light_type", "light_cond", "composition"]
LABEL = {"size": "shot size (SS)", "framing": "shot framing (SF)", "angle": "camera angle (CA)", "lens": "lens size (LS)",
         "light_type": "lighting type (LT)", "light_cond": "lighting conditions (LC)", "composition": "composition (SC)"}
# Published ShotBench paper numbers (Table 2), 7 static columns; fixed in a recorded decision.
SHOTVL3B = dict(zip(ATTRS, [77.9, 85.6, 68.8, 59.3, 65.7, 53.1, 57.4]))
GPT4O = dict(zip(ATTRS, [69.3, 83.1, 58.2, 48.9, 63.2, 48.0, 55.2]))
B, SEED = 10_000, 0


NA_LETTERLOGIT = "n/a (letter-logit protocol)"


def verdict(lo: float, hi: float, y: float, applies: bool = True) -> str:
    if not applies:
        return NA_LETTERLOGIT
    return "above" if lo > y else ("below" if hi < y else "comparable")


def bootstrap(per_attr: dict[str, np.ndarray], rng) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Stratified by attribute: each attribute's questions resampled with replacement to its own n. Returns per-attribute and mean draws (%)."""
    draws = {}
    for a in ATTRS:
        v = per_attr[a]
        idx = rng.integers(0, len(v), size=(B, len(v)))
        draws[a] = 100 * v[idx].mean(axis=1)
    return draws, np.mean([draws[a] for a in ATTRS], axis=0)


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="matched_v1")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    pred_p = root / f"results/p3m_matched/raw_eval_logs/{a.tag}.predictions.jsonl"
    summ_p = root / f"results/p3m_matched/shotbench_{a.tag}.json"
    ll_p = root / "results/phase3b/shotbench_qwen3vl2b_lora_filmsplit.csv"
    preds = [json.loads(line) for line in pred_p.read_text().splitlines() if line.strip()]
    summ = json.loads(summ_p.read_text())
    rule_0018 = not str(summ.get("protocol", "")).startswith("p3m-letterlogit")  # A2: no above/below verdicts for the letter-logit protocol
    rows = {r["attribute"]: r for r in summ["rows"]}
    ll_rows = [r for r in csv.DictReader(ll_p.open()) if r["model"] == "qwen3vl2b_lora_filmsplit"]
    letter_logit = {r["attribute"]: 100 * float(r["correct"]) / float(r["n"]) for r in ll_rows}  # exact counts (CSV cells are rounded)
    assert sorted(letter_logit) == sorted(ATTRS)

    correct, fallback, n_inv = {}, {}, {}
    for at in ATTRS:
        r = [p for p in preds if p["attribute"] == at]
        c = np.array([1.0 if p["correct"] else 0.0 for p in r])
        inv = np.array([p["letter"] is None for p in r])
        # expected credit of the official random fallback per invalid answer, 1/(n_choices+1); the predictions file has no n_choices, so the
        # per-attribute mean credit is backed out from the evaluator's expected_fallback_acc (exact for the point estimate).
        k = int(inv.sum())
        credit = ((rows[at]["expected_fallback_acc_pct"] / 100) * len(r) - c.sum()) / k if k else 0.0
        correct[at], fallback[at], n_inv[at] = c, c + credit * inv, k
        assert len(r) == rows[at]["n"] and int(c.sum()) == rows[at]["correct"], f"predictions disagree with the summary for {at}"

    rng = np.random.default_rng(SEED)
    d_acc, d_mean = bootstrap(correct, rng)
    d_fb, d_fb_mean = bootstrap(fallback, rng)

    out = {"tag": a.tag, "bootstrap": {"resamples": B, "seed": SEED, "stratified_by": "attribute", "interval": "percentile 2.5/97.5"},
           "inputs_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (pred_p, summ_p, ll_p)},
           "n": len(preds), "invalid_n": sum(n_inv.values()), "invalid_rate_pct": 100 * sum(n_inv.values()) / len(preds), "attributes": []}
    for at in ATTRS:
        acc = 100 * correct[at].mean()
        lo, hi = ci(d_acc[at])
        out["attributes"].append({
            "attribute": at, "n": len(correct[at]), "correct": int(correct[at].sum()), "acc_pct": acc, "ci95": [lo, hi],
            "invalid_n": n_inv[at], "invalid_rate_pct": 100 * n_inv[at] / len(correct[at]),
            "expected_fallback_acc_pct": 100 * fallback[at].mean(), "expected_fallback_ci95": ci(d_fb[at]),
            "letter_logit_acc_pct": letter_logit[at], "shotvl3b_published": SHOTVL3B[at], "gpt4o_published": GPT4O[at],
            "verdict_vs_shotvl3b": verdict(lo, hi, SHOTVL3B[at], rule_0018), "verdict_vs_gpt4o": verdict(lo, hi, GPT4O[at], rule_0018)})
    m = float(np.mean([100 * correct[at].mean() for at in ATTRS]))
    mlo, mhi = ci(d_mean)
    ys, yg = float(np.mean(list(SHOTVL3B.values()))), float(np.mean(list(GPT4O.values())))
    out["mean"] = {"acc_pct": m, "ci95": [mlo, mhi], "expected_fallback_acc_pct": float(np.mean([100 * fallback[at].mean() for at in ATTRS])),
                   "expected_fallback_ci95": ci(d_fb_mean), "letter_logit_acc_pct": float(np.mean(list(letter_logit.values()))),
                   "shotvl3b_published": ys, "gpt4o_published": yg,
                   "verdict_vs_shotvl3b": verdict(mlo, mhi, ys, rule_0018), "verdict_vs_gpt4o": verdict(mlo, mhi, yg, rule_0018)}
    if not rule_0018:
        out["protocol_note"] = "letter-logit protocol: published numbers listed for reference only; the pre-registered verdict rule does not apply"
    (root / f"results/p3m_matched/shotbench_{a.tag}_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
