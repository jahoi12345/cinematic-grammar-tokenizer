# ruff: noqa: ISC004
"""Diagnostic (NOT part of the selection rule): predicted-label distribution of the candidate methods on the ShotQA validation subset vs the gold-label
distribution, to see whether the full-list method D shows letter-position bias. Writes results/phase3c/position_bias.md."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cgt.p3c_tokens import selection
from cgt.p3c_tokens.vocab import ATTRS, labels

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "data/cache/p3c/select"
METHODS = ["D", "B", "A2m", "A3m"]


def main() -> None:
    samples = selection.validation_subset()
    recs = {m: {(r["id"], r["attr"]): np.array(r["s"]) for r in map(json.loads, (SEL / f"{m}.jsonl").read_text().splitlines())} for m in METHODS}
    L = ["# Diagnostic: predicted-label distribution on the ShotQA validation subset (not part of the selection rule)", "",
         "Per attribute: share of the 214 questions whose gold set contains the label, and share predicted as primary label by each method. "
         "'never' = labels a method never predicts as primary. Position bias would show as over-prediction of the first labels in schema order (letters A, B, ...) by D, which lists all labels in schema order.", ""]
    summ = {m: 0 for m in METHODS}
    for a in ATTRS:
        labs = labels(a)
        rows = [s for s in samples if s["attribute"] == a]
        gold = Counter(lab for s in rows for lab in selection.gold_labels(s))
        pred = {m: Counter(labs[int(np.argmax(recs[m][(s["id"], a)]))] for s in rows) for m in METHODS}
        L += [f"### {a}", "", "| label (schema order) | gold % | " + " | ".join(f"{m} pred %" for m in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for lab in labs:
            L.append(f"| {lab} | {100 * gold[lab] / len(rows):.1f} | " + " | ".join(f"{100 * pred[m][lab] / len(rows):.1f}" for m in METHODS) + " |")
        nv = {m: sum(1 for lab in labs if pred[m][lab] == 0) for m in METHODS}
        for m in METHODS:
            summ[m] += nv[m]
        L += ["", "never predicted (count of labels): " + ", ".join(f"{m} {nv[m]}" for m in METHODS), ""]
    L += ["## Total labels never predicted as primary on validation (out of 51)", "", ", ".join(f"{m}: {v}" for m, v in summ.items()), ""]
    (ROOT / "results/phase3c/position_bias.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-4:]))
    print("\n".join(x for x in L if x.startswith("never")))


if __name__ == "__main__":
    main()
