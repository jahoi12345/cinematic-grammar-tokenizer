"""Step 3 selection (CPU, docs/plans/step3_gap.md section 5, BINDING) -> results/step3/selection.{json,md}.
Reads each candidate's results/step3/runs/step3_<G>/val_ckpt100.summary.json (aggregates) and applies cgt.step3.select.select
against the plain 4B baseline (C.BASELINE). Descriptive only: a film-clustered paired bootstrap of the winner's gain (needs the
per-question predictions under data/cache, gitignored) and the robustness-proxy table (accuracy on the 1,400-question subset,
original vs re-sampled + JPEG-50 images). Neither changes the decision. No model is loaded."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.p3_static_attrs.data import image_films
from cgt.p3b_vlm import guards
from cgt.step3 import config as C
from cgt.step3.data import proxy_subset, val_dedup_v2
from cgt.step3.select import ATTRS, film_bootstrap, select, target_mean


def _preds(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def per_attr_points(rows: list[dict]) -> dict:
    out = {}
    for a in ATTRS:
        r = [x for x in rows if x["attribute"] == a]
        out[a] = 100.0 * sum(x["pred"] == x["gold"] for x in r) / len(r)
    return out


def main() -> None:
    base = C.BASELINE
    cands, missing = [], []
    for cand in C.RUN_ORDER:
        f = C.RESULTS / "runs" / f"step3_{cand}" / "val_ckpt100.summary.json"
        if not f.exists():
            missing.append(cand)
            continue
        s = json.loads(f.read_text())
        q = s["shotqa_dedup_v2"]
        cands.append({"cand": cand, "primary": q["primary"], "per_attribute": q["per_attribute"], "n": q["n"],
                      "adapter_sha256": s["adapter_sha256"], "predictions_sha256": s.get("predictions_sha256"),
                      "predictions_file": s.get("predictions_file")})
    res = select(cands, base)
    out = {"rule": "docs/plans/step3_gap.md section 5", "baseline": {**base, "target_mean": target_mean(base["per_attribute"])},
           "missing_candidates": missing, **{k: res[k] for k in ("table", "p_star", "tie_set", "extra_steps_only")},
           "winner": res["winner"]["cand"] if res["winner"] else None}
    if res["winner"]:
        out["winner"] = {k: res["winner"][k] for k in ("cand", "primary", "per_attribute", "target_mean", "adapter_sha256")}
    # descriptive: film-clustered bootstrap of every finished candidate vs the baseline (same dedup_v2 questions)
    lines, val = val_dedup_v2()
    fm = image_films(C.SHOTQA_ROOT / "meta.jsonl", {r["img_id"] for r in val})[0]
    if any(not fm.get(r["img_id"]) for r in val):
        raise SystemExit("a validation image has no film key")
    film_of = {ln: fm[r["img_id"]] for ln, r in zip(lines, val, strict=True)}
    boot = {}
    if guards.sha256_file(C.BASE_PRED) == C.BASE_PRED_SHA:
        bp = {r["line"]: r for r in _preds(C.BASE_PRED)}
        for c in cands:
            pf = c.get("predictions_file")
            if not pf or not Path(pf).exists() or guards.sha256_file(pf) != c["predictions_sha256"]:
                continue
            cp = {r["line"]: r for r in _preds(pf)}
            ln = [x for x in lines if x in cp and x in bp]
            boot[c["cand"]] = film_bootstrap([int(cp[x]["pred"] == cp[x]["gold"]) for x in ln],
                                             [int(bp[x]["pred"] == bp[x]["gold"]) for x in ln],
                                             [cp[x]["attribute"] for x in ln], [film_of[x] for x in ln])
    out["bootstrap_descriptive"] = boot
    # descriptive: robustness proxy (accuracy on the subset: original images from the full val predictions vs transformed)
    sub_lines = {lines[p] for p in proxy_subset(val)}
    proxy = {}
    for name, full_pred, pf in [("base", C.BASE_PRED, C.RESULTS / "runs/base/proxy.summary.json")] + [
            (c["cand"], c.get("predictions_file"), C.RESULTS / f"runs/step3_{c['cand']}/proxy_ckpt100.summary.json") for c in cands]:
        if not pf.exists() or not full_pred or not Path(full_pred).exists():
            continue
        orig = per_attr_points([r for r in _preds(full_pred) if r["line"] in sub_lines])
        shift = json.loads(pf.read_text())["shotqa_dedup_v2"]["per_attribute"]
        proxy[name] = {"original": orig, "transformed": shift, "drop": {a: shift[a] - orig[a] for a in ATTRS},
                       "mean_original": sum(orig.values()) / 7, "mean_transformed": sum(shift.values()) / 7}
    out["robustness_proxy_descriptive"] = proxy
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "selection.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    md = ["# Step 3 selection (docs/plans/step3_gap.md section 5)", "",
          f"Baseline (plain 4B): primary {base['primary']:.2f}, target mean {target_mean(base['per_attribute']):.2f}", "",
          "| cand | primary | target mean | " + " | ".join(ATTRS) + " | eligible | reasons |", "|" + "---|" * (len(ATTRS) + 5)]
    for t in res["table"]:
        md.append(f"| {t['cand']} | {t['primary']:.2f} | {t['target_mean']:.2f} | "
                  + " | ".join(f"{t['per_attribute'][a]:.2f}" for a in ATTRS) + f" | {t['eligible']} | {'; '.join(t['reasons'])} |")
    w = res["winner"]
    md += ["", f"Winner: {w['cand'] + (' (extra steps only, not the variant)' if w['cand'] == 'G0' else '') if w else 'none (no gain; the plain 4B stays)'}",
           f"Missing (dropped/unfinished): {missing or 'none'}", "",
           "Descriptive only (not used for selection): film-clustered bootstrap " + json.dumps(boot, default=str), "",
           "Robustness proxy (descriptive; NOT ShotBench; a ShotQA-val gain may not close the ShotBench gap): "
           + json.dumps({k: {"mean_original": v["mean_original"], "mean_transformed": v["mean_transformed"]} for k, v in proxy.items()})]
    (C.RESULTS / "selection.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
