"""Step C selection (docs/plans/stepc_qwen3vl4b.md section 5; rule in cgt.stepc.select). CPU only; reads validation summaries and
per-question correctness only. Baseline = the frozen 2B 4c030967 on dedup_v2, re-scored from its existing Phase 3B predictions
(ckpt_100.predictions.jsonl, sha256 cebea813..., all 5,826 val.jsonl lines in order) on the dedup_v2 lines. Writes AGGREGATES only to
results/stepc/selection.{json,md} (+ --baseline-only: results/stepc/baseline_dedup_v2.json).
  PYTHONPATH=src <venv>/bin/python scripts/stepc_select.py [--baseline-only]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.p3_static_attrs.data import image_films
from cgt.p3b_vlm.guards import sha256_file
from cgt.p3b_vlm.score import summarize
from cgt.stepb.evaluate import shotqa_points
from cgt.stepc import config as C
from cgt.stepc.data import val_dedup_v2
from cgt.stepc.select import ATTRS, film_bootstrap, select

RUN = C.RESULTS / "runs/stepc_4b"


def baseline() -> tuple[dict, dict[int, int]]:
    """(dedup_v2 points of the frozen 2B, {val line: 0/1 correct})."""
    if sha256_file(C.FROZEN_PRED) != C.FROZEN_PRED_SHA:
        raise SystemExit("frozen predictions file does not match its recorded sha256")
    preds = [json.loads(x) for x in C.FROZEN_PRED.read_text().splitlines()]
    lines, rows = val_dedup_v2()
    all_rows = [json.loads(x) for x in C.VAL_JSONL.read_text().splitlines()]
    if [p["id"] for p in preds] != [r["id"] for r in all_rows]:
        raise SystemExit("frozen predictions are not in val.jsonl order")
    keep = [preds[n] for n in lines]
    if [p["attribute"] for p in keep] != [r["attribute"] for r in rows]:
        raise SystemExit("frozen predictions do not align with dedup_v2 rows")
    return shotqa_points(summarize(keep)), {n: int(p["pred"] == p["gold"]) for n, p in zip(lines, keep, strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()
    base, base_hit = baseline()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "baseline_dedup_v2.json").write_text(json.dumps({"model": "frozen 2B 4c030967", "predictions_sha256": C.FROZEN_PRED_SHA,
                                                                  "val_sha256": C.VAL_SHA, **base}, indent=1) + "\n")
    if a.baseline_only:
        print(json.dumps(base, indent=1))
        return
    cks, missing = [], []
    for pct in (50, 100):
        f = RUN / f"val_ckpt{pct:03d}.summary.json"
        if not f.exists():
            missing.append(f"{pct}%")
            continue
        s = json.loads(f.read_text())
        q = s["shotqa_dedup_v2"]
        if q["n"] != base["n"]:
            raise SystemExit(f"checkpoint {pct}% scored {q['n']} questions, baseline {base['n']}")
        cks.append({"pct": pct, "primary": q["primary"], "per_attribute": q["per_attribute"], "n": q["n"],
                    "adapter_sha256": s["adapter_sha256"], "predictions_file": s["predictions_file"],
                    "predictions_sha256": s["predictions_sha256"]})
    res = select(cks, base)
    w = res["winner"]
    out = {"rule": "docs/plans/stepc_qwen3vl4b.md section 5", "baseline": base, "missing_checkpoints": missing,
           **{k: v for k, v in res.items() if k != "winner"}, "winner": None if w is None else {k: w[k] for k in ("pct", "primary", "adapter_sha256")}}
    if w is not None:
        pf = Path(w["predictions_file"])
        if sha256_file(pf) != w["predictions_sha256"]:
            raise SystemExit("winner predictions changed since validation")
        preds = [json.loads(x) for x in pf.read_text().splitlines()]
        lines, rows = val_dedup_v2()
        if [p["line"] for p in preds] != lines:
            raise SystemExit("winner predictions are not the dedup_v2 lines in order")
        films = image_films(C.SHOTQA_ROOT / "meta.jsonl", {r["img_id"] for r in rows})[0]
        if any(not films.get(r["img_id"]) for r in rows):
            raise SystemExit("a validation image has no film key")
        out["bootstrap_descriptive"] = film_bootstrap([int(p["pred"] == p["gold"]) for p in preds], [base_hit[n] for n in lines],
                                                      [r["attribute"] for r in rows], [films[r["img_id"]] for r in rows])
    for t in out["table"]:
        t.pop("predictions_file", None)
    (C.RESULTS / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# Step C selection (rule fixed in docs/plans/stepc_qwen3vl4b.md section 5)", "",
          f"Primary: unweighted 7-attribute mean on the dedup_v2 ShotQA film-held-out validation set ({base['n']} questions).", "",
          "| model | primary | " + " | ".join(ATTRS) + " | eligible | reasons |", "|---|---|" + "---|" * len(ATTRS) + "---|---|",
          f"| frozen 2B 4c030967 (baseline) | {base['primary']:.2f} | " + " | ".join(f"{base['per_attribute'][x]:.2f}" for x in ATTRS) + " | - | - |"]
    for t in out["table"]:
        md.append(f"| 4B LoRA {t['pct']}% | {t['primary']:.2f} | " + " | ".join(f"{t['per_attribute'][x]:.2f}" for x in ATTRS)
                  + f" | {'yes' if t['eligible'] else 'no'} | " + ("; ".join(t["reasons"]) or "-") + " |")
    md += ["", f"Missing checkpoints (dropped or not produced): {', '.join(missing) or 'none'}", ""]
    if w is None:
        md.append("**No gain: no eligible checkpoint, the frozen 2B 4c030967 stays.**")
    else:
        b = out["bootstrap_descriptive"]
        md += [f"**Winner: 4B LoRA {w['pct']}%**, adapter sha256 `{w['adapter_sha256']}` (tie set {res['tie_set']}).", "",
               (f"Gain over the frozen 2B: {b['gain']:+.2f} points; film-clustered bootstrap 95% interval [{b['ci95'][0]:+.2f}, "
                f"{b['ci95'][1]:+.2f}] ({b['n_films']} films, {b['n_boot']} draws, seed {b['seed']}; descriptive, not a test).")]
    (C.RESULTS / "selection.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
