"""Step B selection (docs/plans/stepb_cinescale2.md section 5; rule in cgt.stepb.select). CPU only; reads validation summaries only.

Baseline: the frozen 4c030967 adapter. Its ShotQA numbers are re-scored on the deduplicated validation set FROM ITS EXISTING Phase 3B
predictions (data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/val_predictions/ckpt_100.predictions.jsonl, sha256 cebea813...,
same scorer, same val.jsonl order; dedup = first occurrence by line). Its CineScale2 numbers come from results/stepb/runs/baseline/.
Writes results/stepb/selection.{json,md}."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.p3b_vlm.guards import sha256_file
from cgt.p3b_vlm.score import summarize
from cgt.stepb.data import RUN_ORDER
from cgt.stepb.evaluate import shotqa_points
from cgt.stepb.select import ATTRS, select

FROZEN_PRED = ROOT / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/val_predictions/ckpt_100.predictions.jsonl"
FROZEN_PRED_SHA = "cebea813002eb24971d6ae652b38a90a0f7fc7105ceed92d9a0da3c64770ece6"
RUNS = ROOT / "results/stepb/runs"


def baseline_shotqa() -> dict:
    if sha256_file(FROZEN_PRED) != FROZEN_PRED_SHA:
        raise SystemExit("frozen predictions file does not match its recorded sha256")
    preds = [json.loads(x) for x in FROZEN_PRED.read_text().splitlines()]
    val = [json.loads(x) for x in (ROOT / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    if [p["id"] for p in preds] != [v["id"] for v in val]:
        raise SystemExit("frozen predictions are not in val.jsonl order")
    seen, keep = set(), []
    for p in preds:
        if p["id"] not in seen:
            seen.add(p["id"])
            keep.append(p)
    full = shotqa_points(summarize(preds))
    return {**shotqa_points(summarize(keep)), "non_dedup_primary": full["primary"]}


def main() -> None:
    base_qa = baseline_shotqa()
    base_cs2 = json.loads((RUNS / "baseline/val_frozen.summary.json").read_text())["cs2"]
    base = {"primary": base_qa["primary"], "per_attribute": base_qa["per_attribute"], "secondary": base_cs2["acc"],
            "n": base_qa["n"], "non_dedup_primary": base_qa["non_dedup_primary"], "cs2": base_cs2}
    cks, missing = [], []
    for cand in RUN_ORDER:
        for pct in (50, 100):
            f = RUNS / f"stepb_{cand}" / f"val_ckpt{pct:03d}.summary.json"
            if not f.exists():
                missing.append(f"{cand} {pct}%")
                continue
            s = json.loads(f.read_text())
            cks.append({"candidate": cand, "pct": pct, "primary": s["shotqa"]["primary"], "per_attribute": s["shotqa"]["per_attribute"],
                        "secondary": s["cs2"]["acc"], "cs2": s["cs2"], "adapter_sha256": s["adapter_sha256"], "n": s["shotqa"]["n"]})
    if any(c["n"] != base["n"] for c in cks):
        raise SystemExit("validation sizes differ from the baseline")
    res = select(cks, base, RUN_ORDER)
    out = {"baseline": base, "missing_checkpoints": missing, **res}
    (ROOT / "results/stepb/selection.json").write_text(json.dumps(out, indent=1) + "\n")
    w = res["winner"]
    lines = ["# Step B selection (rule fixed in docs/plans/stepb_cinescale2.md section 5)", "",
             (f"Primary: unweighted 7-attribute mean on the deduplicated ShotQA film-held-out validation set ({base['n']} questions). "
              f"Secondary: CineScale2 split-v3 val angle accuracy ({base_cs2['n']} questions)."), "",
             "| model | primary | " + " | ".join(ATTRS) + " | CineScale2 acc | CineScale2 balanced recall | eligible | reasons |",
             "|---|---|" + "---|" * len(ATTRS) + "---|---|---|---|",
             f"| frozen 4c030967 (baseline) | {base['primary']:.2f} | " + " | ".join(f"{base['per_attribute'][x]:.2f}" for x in ATTRS)
             + f" | {base_cs2['acc']:.2f} | {base_cs2['balanced_recall']:.2f} | - | - |"]
    for t in res["table"]:
        lines.append(f"| {t['candidate']} {t['pct']}% | {t['primary']:.2f} | " + " | ".join(f"{t['per_attribute'][x]:.2f}" for x in ATTRS)
                     + f" | {t['secondary']:.2f} | {t['cs2']['balanced_recall']:.2f} | {'yes' if t['eligible'] else 'no'} | "
                     + ("; ".join(t["reasons"]) or "-") + " |")
    lines += ["", f"Missing checkpoints (not run or not finished): {', '.join(missing) or 'none'}", "",
              (f"**Winner: {w['candidate']} {w['pct']}%**, adapter sha256 `{w['adapter_sha256']}` (tie set {res['tie_set']})."
               if w else "**No eligible checkpoint: the frozen 4c030967 model stays.**"), "",
              f"Baseline non-deduplicated mean (check against the Phase 3B record 81.16): {base['non_dedup_primary']:.2f}."]
    (ROOT / "results/stepb/selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
