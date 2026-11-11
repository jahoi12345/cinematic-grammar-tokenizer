"""Base-generation experiment: paired comparison at the matched fraction (docs/plans/base_gen_matched_budget.md sections 5-6).
CPU only. Reads the two per-question prediction files named in results/base_gen/val_{3b,2b}_ckpt<pct>.summary.json (sha-checked),
checks both are the dedup_v2 lines in order, computes d = 3B - 2B and the film-clustered bootstrap 95% interval (cgt.stepc.select.
film_bootstrap: seed 0, 2,000 draws, as Step C) and applies the pre-stated reading rule (cgt.base_gen.reading). Aggregates only ->
results/base_gen/compare_ckpt<pct>.{json,md}.
  PYTHONPATH=src <venv>/bin/python scripts/base_gen_compare.py --pct 25"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.base_gen import config as C
from cgt.base_gen.reading import EQUIV_MARGIN, read
from cgt.p3_static_attrs.data import image_films
from cgt.p3b_vlm.guards import sha256_file
from cgt.stepc import config as SC
from cgt.stepc.data import val_dedup_v2
from cgt.stepc.select import ATTRS, film_bootstrap


def load_arm(arm: str, pct: int, lines: list[int]) -> tuple[dict, list[int]]:
    s = json.loads((C.RESULTS / f"val_{arm}_ckpt{pct:03d}.summary.json").read_text())
    pf = Path(s["predictions_file"])
    if sha256_file(pf) != s["predictions_sha256"]:
        raise SystemExit(f"{arm}: predictions changed since validation")
    preds = [json.loads(x) for x in pf.read_text().splitlines()]
    if [p["line"] for p in preds] != lines:
        raise SystemExit(f"{arm}: predictions are not the dedup_v2 lines in order")
    return s, [int(p["pred"] == p["gold"]) for p in preds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, choices=(25, 50), required=True)
    a = ap.parse_args()
    lines, rows = val_dedup_v2()
    s3, h3 = load_arm("3b", a.pct, lines)
    s2, h2 = load_arm("2b", a.pct, lines)
    films = image_films(SC.SHOTQA_ROOT / "meta.jsonl", {r["img_id"] for r in rows})[0]
    if any(not films.get(r["img_id"]) for r in rows):
        raise SystemExit("a validation image has no film key")
    b = film_bootstrap(h3, h2, [r["attribute"] for r in rows], [films[r["img_id"]] for r in rows])
    verdict = read(b["gain"], *b["ci95"])
    p3, p2 = s3["shotqa_dedup_v2"], s2["shotqa_dedup_v2"]
    out = {"pct": a.pct, "rule": "docs/plans/base_gen_matched_budget.md section 6", "equiv_margin": EQUIV_MARGIN,
           "3b": {"model": s3["model"], "adapter_sha256": s3["adapter_sha256"], "primary": p3["primary"], "per_attribute": p3["per_attribute"]},
           "2b": {"model": s2["model"], "adapter_sha256": s2["adapter_sha256"], "primary": p2["primary"], "per_attribute": p2["per_attribute"]},
           "d_3b_minus_2b": b, "reading": verdict}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / f"compare_ckpt{a.pct:03d}.json").write_text(json.dumps(out, indent=1) + "\n")
    md = [f"# Base generation at matched budget: {a.pct}% of the epoch (rule fixed in docs/plans/base_gen_matched_budget.md)", "",
          "| model | primary | " + " | ".join(ATTRS) + " |", "|---|---|" + "---|" * len(ATTRS)]
    for k, p in (("Qwen2.5-VL-3B LoRA", p3), ("Qwen3-VL-2B LoRA", p2)):
        md.append(f"| {k} {a.pct}% | {p['primary']:.2f} | " + " | ".join(f"{p['per_attribute'][x]:.2f}" for x in ATTRS) + " |")
    md += ["", (f"d = 3B - 2B = {b['gain']:+.2f} points, film-clustered bootstrap 95% interval [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] "
                f"({b['n_films']} films, {b['n_boot']} draws, seed {b['seed']})."), "", f"**Reading (pre-stated rule): {verdict}**"]
    (C.RESULTS / f"compare_ckpt{a.pct:03d}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
