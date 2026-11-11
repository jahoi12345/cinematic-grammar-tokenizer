"""Step A static track: rank the pre-registered configurations on the selection half and apply the decision rule
(docs/acc_step_a_static_plan.md sections 5-6). CPU only; reads the cached view logits, the Phase 3A option scores and the validation labels.
Writes results/p3s_stepa/static_val.{json,md} (aggregate numbers only). Usage: uv run python scripts/p3s_stepa_report.py [--gpu-hours X]"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3s_stepa import combine as C
from cgt.p3s_stepa.split import ATTRS, load_split, shotqa_film_of
from cgt.p3s_stepa.views import RUN_ORDER, applies

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PRED = ROOT / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/val_predictions/ckpt_100.predictions.jsonl"
FROZEN_PRED_SHA256 = "cebea813002eb24971d6ae652b38a90a0f7fc7105ceed92d9a0da3c64770ece6"
NAMES = {"size": "shot size", "lens": "lens size", "angle": "camera angle", "framing": "shot framing", "light_type": "lighting type",
         "light_cond": "lighting conditions", "composition": "composition"}


def load_views(qs: list[dict], vdir: Path) -> tuple[dict, dict]:
    views, complete = {}, {}
    for v in RUN_ORDER:
        f = vdir / f"{v}.jsonl"
        d = {}
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                d[r["key"]] = r["logits"]
        need = [q["key"] for q in qs if applies(v, q["attribute"])]
        complete[v] = {"done": sum(k in d for k in need), "needed": len(need)}
        views[v] = d
    return views, complete


def needed_views(name: str) -> set[str]:
    vs = set(C.CONFIGS[name][0])
    if "C2*" in vs:
        vs = (vs - {"C2*"}) | {"res262k", "res590k"}
    return vs


def parity(qs: list[dict], views: dict) -> dict:
    if hashlib.sha256(FROZEN_PRED.read_bytes()).hexdigest() != FROZEN_PRED_SHA256:
        raise SystemExit("frozen predictions file hash mismatch")
    P = FROZEN_PRED.read_text().splitlines()
    diffs, same = [], 0
    for q in qs:
        r = json.loads(P[q["line"]])
        assert r["id"] == q["id"]
        a, b = np.array(views["base"][q["key"]]), np.array(r["logits"])
        diffs.append(float(np.abs(a - b).max()))
        same += int(np.argmax(a) == np.argmax(b))
    return {"questions": len(qs), "max_abs_logit_diff": max(diffs), "same_argmax": same}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args()
    sp = load_split(ROOT / "data/cache/p3b/val.jsonl", shotqa_film_of)
    qs = sp["questions"]
    views, complete = load_views(qs, ROOT / "data/cache/p3s_stepa/views")
    sig = {r["key"]: r["scores"] for r in map(json.loads, (ROOT / "data/cache/p3s_stepa/siglip3a.jsonl").read_text().splitlines())}
    full_views = {v for v, c in complete.items() if c["done"] == c["needed"]}
    available = [n for n in C.ORDER if needed_views(n) <= full_views]
    sel = [q for q in qs if q["half"] == "sel"]
    cal = [q for q in qs if q["half"] == "cal"]
    res, hits_sel = {}, {}
    for n in available:
        r = C.run_config(n, qs, views, sig)
        res[n] = {"selection": C.accuracy(sel, r["scores"]), "calibration": C.accuracy(cal, r["scores"]), "full": C.accuracy(qs, r["scores"]),
                  "fit": r["info"]}
        hits_sel[n] = C.hits(sel, r["scores"])
        print(f"{n}: selection {100 * res[n]['selection']['mean']:.2f}  full {100 * res[n]['full']['mean']:.2f}", flush=True)
    boot = {}
    for n in available:
        if n != "C0":
            boot[n] = {cl: C.paired_bootstrap(sel, hits_sel["C0"], hits_sel[n], cluster=cl, n_boot=a.n_boot, seed=0) for cl in ("film", "question")}
    dec = C.decide({n: res[n]["selection"]["mean"] for n in available}, {n: b["film"] for n, b in boot.items()}, available)
    sig_only = {"selection": C.accuracy(sel, sig), "full": C.accuracy(qs, sig)}  # exploratory reference, cannot win
    out = {"time": now_iso(), "plan": "docs/acc_step_a_static_plan.md", "split": sp["info"], "views_complete": complete,
           "available_configs": available, "not_run": [n for n in C.ORDER if n not in available], "results": res, "bootstrap_vs_C0": boot,
           "decision": dec, "c0_parity_vs_frozen_predictions": parity(qs, views), "phase3a_alone_exploratory": sig_only, "gpu_hours": a.gpu_hours,
           "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()}
    od = ROOT / "results/p3s_stepa"
    od.mkdir(parents=True, exist_ok=True)
    (od / "static_val.json").write_text(json.dumps(out, indent=1))
    (od / "static_val_tables.md").write_text(tables(out))
    print(json.dumps(dec, indent=1))


def tables(o: dict) -> str:
    """Markdown tables of the aggregate numbers (pasted into static_val.md)."""
    res, boot = o["results"], o["bootstrap_vs_C0"]
    L = []
    for part, n_key in (("selection", "sel"), ("full", None)):
        n_attr = o["split"][n_key]["per_attribute"] if n_key else None
        hdr = "| config | " + " | ".join(NAMES[a] + (f" (n={n_attr[a]})" if n_attr else "") for a in ATTRS) + " | **7-attr mean** | diff vs C0 |"
        L += [f"### {part} {'half' if n_key else 'validation (5,762)'}", "", hdr, "|---" * (len(ATTRS) + 3) + "|"]
        for n, r in res.items():
            acc = r[part]
            d = 100 * (acc["mean"] - res["C0"][part]["mean"])
            L.append(f"| {n} | " + " | ".join(f"{100 * acc['per_attribute'][a]:.2f}" for a in ATTRS) + f" | **{100 * acc['mean']:.2f}** | {d:+.2f} |")
        e = o["phase3a_alone_exploratory"][part]
        L += ["| (Phase 3A heads alone, exploratory) | " + " | ".join(f"{100 * e['per_attribute'][a]:.2f}" for a in ATTRS) + f" | {100 * e['mean']:.2f} | |", ""]
    L += ["### Paired bootstrap vs C0 on the selection half (points of the 7-attribute mean; 10,000 resamples, seed 0)", "",
          "| config | diff | film-clustered 95% interval (decides) | question-level 95% interval (secondary) |", "|---|---|---|---|"]
    for n, b in boot.items():
        f, q = b["film"], b["question"]
        L.append(f"| {n} | {f['diff_points']:+.2f} | [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}] | [{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}] |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
