# ruff: noqa: ISC004
"""Phase 3C step 2 report: metrics for every candidate on the validation subset, the FIXED selection rule, results/phase3c/method_selection.{md,json}."""
import json
from pathlib import Path

import numpy as np

from cgt.p3c_tokens import selection
from cgt.p3c_tokens.methods import METHODS
from cgt.p3c_tokens.vocab import ATTRS

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "data/cache/p3c/select"
OUT = ROOT / "results/phase3c"
PASSES = {"A2m": 26, "A2b": 26, "A3m": 39, "A3b": 39, "B": 21, "C": 51, "D": 7}


def load(m: str) -> dict:
    """(sample id, attribute) -> record. Four subset ids occur twice in sft.json (same image re-asked): method scores depend only on (image, attribute)."""
    return {(r["id"], r["attr"]): r for r in map(json.loads, (SEL / f"{m}.jsonl").read_text().splitlines())}


def main() -> None:
    samples = selection.validation_subset()
    ids = set(json.loads((SEL / "subset_ids.json").read_text()))
    samples = [s for s in samples if s["id"] in ids]
    res, rows, rows_q = {}, {}, {}
    for m in METHODS:
        rec = load(m)
        sc = {k: np.array(v["s"]) for k, v in rec.items()}  # keyed (id, attribute)
        res[m] = selection.evaluate_method(samples, sc)
        rows[m] = float(np.mean([v["passes"] for v in rec.values()]))  # per unique (id, attribute) query
        rows_q[m] = float(np.mean([rec[(s["id"], s["attribute"])]["passes"] for s in samples]))  # weighted per question (1,498; shared duplicates counted twice)
    nat = load("native")
    fix = {r["pos"]: r for r in map(json.loads, (SEL / "native_collisions.jsonl").read_text().splitlines())} if (SEL / "native_collisions.jsonl").exists() else {}
    dup = {i for i, c in __import__("collections").Counter(s["id"] for s in samples).items() if c > 1}
    pos = {id(s): j for j, s in enumerate(samples)}

    by_pos = {r["pos"]: r for r in map(json.loads, (SEL / "native.jsonl").read_text().splitlines()) if "pos" in r}  # new records carry their subset position

    def native_logits(s):
        j = pos[id(s)]
        if j in by_pos:
            return by_pos[j]["logits"]
        return fix[j]["logits"] if s["id"] in dup else nat[(s["id"], s["attribute"])]["logits"]
    native = {}
    for a in ATTRS:
        rr = [s for s in samples if s["attribute"] == a]
        native[a] = float(np.mean([("ABCD"[int(np.argmax(native_logits(s)))]) == s["answer"] for s in rr]))
    native_mean = float(np.mean(list(native.values())))
    rule = selection.select(res)
    eq = json.loads((SEL / "equivalence.json").read_text()) if (SEL / "equivalence.json").exists() else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "method_selection.json").write_text(json.dumps({"results": res, "native_option_acc": native, "native_mean": native_mean,
                                                           "rule": rule, "mean_passes_per_unique_query": rows, "mean_passes_per_question_weighted": rows_q, "equivalence": eq,
                                                           "questions": len(samples)}, indent=1))
    L = ["# Phase 3C method selection (ShotQA film-split validation subset, inference only)",
         f"Subset: {len(samples)} questions ({selection.PER_ATTR} per attribute, `default_rng(0)`, plan section 4). Adapter = frozen `ckpt_100` (sha256 4c030967...). "
         "Rule (docs/phase3c_plan.md section 6): highest composite; methods within 1.0 pp of the best are tied and the one with the fewest passes wins, then 'm' over 'b'.", "",
         "Composite = unweighted mean over the 7 attributes of full-vocabulary primary accuracy (size, lens, angle, framing) and set-F1 at the chosen alpha (light_type, light_cond, composition).", "",
         "## Selection table (the rule's metric)", "",
         "| method | passes/image | " + " | ".join(ATTRS) + " | composite |", "|---|---|" + "---|" * (len(ATTRS) + 1)]
    for m in METHODS:
        r = res[m]
        L.append(f"| {m} | {PASSES[m]} | " + " | ".join(f"{100 * r['per_attribute'][a]['value']:.1f}" for a in ATTRS) + f" | **{100 * r['composite']:.1f}** |")
    L += ["", "(size, lens, angle, framing = full-vocabulary accuracy %; light_type, light_cond, composition = set-F1 %.)", "",
          "## Full-vocabulary primary accuracy (primary label in the gold set), %", "",
          "| method | " + " | ".join(ATTRS) + " | mean |", "|---|" + "---|" * (len(ATTRS) + 1)]
    for m in METHODS:
        v = [res[m]["per_attribute"][a]["full_acc"] for a in ATTRS]
        L.append(f"| {m} | " + " | ".join(f"{100 * x:.1f}" for x in v) + f" | {100 * np.mean(v):.1f} |")
    L += ["", "## Option-restricted accuracy from the method's scores (the question's own 4 options; not in the rule), %", "",
          "| method | " + " | ".join(ATTRS) + " | mean |", "|---|" + "---|" * (len(ATTRS) + 1)]
    for m in METHODS:
        v = [res[m]["per_attribute"][a]["option_acc"] for a in ATTRS]
        L.append(f"| {m} | " + " | ".join(f"{100 * x:.1f}" for x in v) + f" | {100 * np.mean(v):.1f} |")
    L.append("| native 4-option (reference) | " + " | ".join(f"{100 * native[a]:.1f}" for a in ATTRS) + f" | {100 * native_mean:.1f} |")
    L += ["", "## Chosen alpha and set-F1 / primary accuracy for the multi-label attributes", "",
          "| method | " + " | ".join(f"{a} alpha / setF1 / primary-acc" for a in ("light_type", "light_cond", "composition")) + " |", "|---|---|---|---|"]
    for m in METHODS:
        c = [res[m]["per_attribute"][a] for a in ("light_type", "light_cond", "composition")]
        L.append(f"| {m} | " + " | ".join(f"{x['alpha']} / {100 * x['set_f1']:.1f} / {100 * x['full_acc']:.1f}" for x in c) + " |")
    L += ["", "## Decision", "", f"- Top composite: **{rule['top_by_composite']}** ({100 * rule['best_composite']:.1f}).",
          f"- Within 1.0 pp of the best (tie band): {', '.join(rule['within_tie_band'])}.",
          f"- **Selected by the rule: {rule['selected']}.**", "",
          "## Implementation checks", "",
          f"- Cached-prefix scoring vs the stock full forward pass on the native 4-option prompts: {json.dumps(eq)}. Qualification: this is 40/40 argmax agreement, max absolute logit difference 0.25 and mean of the per-question maxima 0.09375 on 4-option prompts; no per-question logit arrays were retained and no acceptance tolerance was pre-registered, so numeric identity is NOT established, and D's 5-12-option prompts are not covered by the diagnostic.",
          "- Mean forward passes per QUESTION (1,498 questions; a repeated (id, attribute) shares one cached query and is counted for each question): " + ", ".join(f"{m} {rows_q[m]:.10g}" for m in METHODS) + ".",
          "- Mean passes per UNIQUE cached (id, attribute) query (1,495 queries): " + ", ".join(f"{m} {rows[m]:.10g}" for m in METHODS)
          + ". Cache records sum to 45,911 passes including the native reference; the retained selection log accounts for 45,483: 428 cached passes (the 14-question smoke run) predate the retained log, so the log is not a complete single-invocation account of every cached score.",
          "- Selected-alpha set-F1 is a SELECTION score: alpha and the method are both chosen on this same subset, so it is not an independent accuracy estimate and not a guarantee for unseen films.",
          "- B semantics: the plan describes B's set as the singleton, but the registered evaluation tunes alpha for B like every other method (light_type and light_cond 0.25); an exactly-singleton B would score 68.14 composite, leave the tie band and still leave D selected (independent recomputation). At alpha 1.0 the set contains every label tied for the maximum score (primary first, lowest schema index), so it is a singleton except for exact score ties.",
          "- Chronology (plan Amendment 1, F4): the candidate methods and selection rule were committed before validation scoring. After scoring, and after the per-method single-label accuracies were visible, an index/name set-F1 bug was corrected. Amendment 1 and the final selection report were committed together (`b385891`); no separately timestamped record establishes that the amendment was written before the first rule calculation, and original report snapshots do not exist. Recomputing the originally specified metric from the retained vectors selects D.",
          "- Headline: on the pre-registered ShotQA film-split validation subset, D's 71.1981% composite is within the registered 1.0pp band of A3m's 71.5751%; the fewest-pass tie rule selects D. These are validation selection scores for the frozen adapter. Its outputs on NOTLD are predictions with unknown accuracy.",
          "- Native 4-option reference on this subset: " + f"{100 * native_mean:.1f}% mean (full-set Phase 3B validation: 81.2%)."]
    (OUT / "method_selection.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
