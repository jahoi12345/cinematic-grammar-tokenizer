"""Step A-4B report (docs/plans/stepa_4b.md sections 5-6). CPU only. Reads the 4B view logits (data/cache/stepa_4b/views), the Phase 3A
option scores (Step A's 5,762 + the 60 dedup_v2 added rows) and the validation labels; combines with the UNCHANGED Step A code
(cgt.p3s_stepa.combine.run_config: every fit on half == 'cal' only), ranks the runnable configurations on the selection half, applies
Step A's rule (cgt.p3s_stepa.combine.decide) and the distillation gate. Writes results/stepa_4b/val.json + val_tables.md (aggregates only).
--repro-2b: recompute Step A's 2B results from its cached views (dedup_first, 5,762) and assert they equal results/p3s_stepa/static_val.json
(selection means, fitted parameters, decision); writes results/stepa_4b/repro_2b.json.
  PYTHONPATH=src nice -n 19 <venv>/bin/python scripts/stepa_4b_report.py [--gpu-hours X] [--repro-2b]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cgt.p3b_vlm.guards import sha256_file
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3s_stepa import combine as C
from cgt.p3s_stepa.split import ATTRS, load_split
from cgt.p3s_stepa.views import RUN_ORDER, applies
from cgt.stepa_4b import config as K
from cgt.stepa_4b import core
from cgt.stepc import config as SC

NAMES = {"size": "shot size", "lens": "lens size", "angle": "camera angle", "framing": "shot framing", "light_type": "lighting type",
         "light_cond": "lighting conditions", "composition": "composition"}


def load_views(qs: list[dict], vdir: Path, names=RUN_ORDER) -> tuple[dict, dict]:
    views, complete = {}, {}
    for v in names:
        f = Path(vdir) / f"{v}.jsonl"
        d = {}
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d[r["key"]] = r["logits"]
        need = [q["key"] for q in qs if applies(v, q["attribute"])]
        complete[v] = {"done": sum(k in d for k in need), "needed": len(need)}
        views[v] = d
    return views, complete


def needed_views(name: str) -> set[str]:
    vs = set(C.CONFIGS[name][0])
    return (vs - {"C2*"}) | {"res262k", "res590k"} if "C2*" in vs else vs


def load_sig(files: list[tuple[Path, str]]) -> dict:
    sig = {}
    for f, sha in files:
        if sha256_file(f) != sha:
            raise SystemExit(f"{f} does not match its recorded sha256")
        for r in map(json.loads, f.read_text().splitlines()):
            sig[r["key"]] = r["scores"]
    return sig


def rank(qs: list[dict], views: dict, sig: dict, available: list[str], n_boot: int) -> dict:
    sel = [q for q in qs if q["half"] == "sel"]
    cal = [q for q in qs if q["half"] == "cal"]
    res, hits_sel, hits_full = {}, {}, {}
    for n in available:
        r = C.run_config(n, qs, views, sig)
        res[n] = {"selection": C.accuracy(sel, r["scores"]), "calibration": C.accuracy(cal, r["scores"]), "full": C.accuracy(qs, r["scores"]),
                  "fit": r["info"]}
        hits_sel[n], hits_full[n] = C.hits(sel, r["scores"]), C.hits(qs, r["scores"])
        print(f"{n}: selection {100 * res[n]['selection']['mean']:.2f}  full {100 * res[n]['full']['mean']:.2f}", flush=True)
    boot = {n: {cl: C.paired_bootstrap(sel, hits_sel["C0"], hits_sel[n], cluster=cl, n_boot=n_boot, seed=0) for cl in ("film", "question")}
            for n in available if n != "C0"}
    dec = C.decide({n: res[n]["selection"]["mean"] for n in available}, {n: b["film"] for n, b in boot.items()}, available)
    full_ci = None
    if dec["top"] != "C0":   # reported, never used for choosing (the calibration-half part is in-sample for fitted configurations)
        full_ci = C.paired_bootstrap(qs, hits_full["C0"], hits_full[dec["top"]], cluster="film", n_boot=n_boot, seed=0)
    return {"results": res, "bootstrap_vs_C0": boot, "decision": dec, "top_vs_C0_full_val_film_ci_reported_only": full_ci}


def repro_2b(n_boot: int) -> dict:
    """Step A's 2B numbers from its cached views with the unchanged code: must equal results/p3s_stepa/static_val.json."""
    ref = json.loads((ROOT / "results/p3s_stepa/static_val.json").read_text())
    qs = load_split(SC.VAL_JSONL, core.shotqa_film_of)["questions"]
    views, complete = load_views(qs, K.STEPA_VIEWS_2B)
    assert all(c["done"] == c["needed"] for c in complete.values()), complete
    sig = load_sig([(K.SIGLIP, K.SIGLIP_SHA)])
    out = rank(qs, views, sig, list(C.ORDER), n_boot)
    diffs = {}
    for n in C.ORDER:
        a, b = out["results"][n]["selection"]["mean"], ref["results"][n]["selection"]["mean"]
        diffs[n] = {"selection_mean": a, "recorded": b, "equal": abs(a - b) < 1e-12,
                    "fit_equal": json.dumps(out["results"][n]["fit"], sort_keys=True) == json.dumps(ref["results"][n]["fit"], sort_keys=True)}
    ok = all(d["equal"] and d["fit_equal"] for d in diffs.values()) and out["decision"] == ref["decision"]
    res = {"time": now_iso(), "ok": ok, "per_config": diffs, "decision": out["decision"], "recorded_decision": ref["decision"],
           "code_commit": git_head()}
    K.RESULTS.mkdir(parents=True, exist_ok=True)
    (K.RESULTS / "repro_2b.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"ok": ok, "decision": out["decision"]}, indent=1))
    return res


def c0_parity(qs: list[dict], views: dict) -> dict:
    if sha256_file(K.STEPC_PRED) != K.STEPC_PRED_SHA:
        raise SystemExit("Step C predictions hash mismatch")
    P = {r["line"]: r["logits"] for r in map(json.loads, K.STEPC_PRED.read_text().splitlines())}
    d = [float(np.abs(np.array(views["base"][q["key"]]) - np.array(P[q["line"]])).max()) for q in qs]
    same = sum(int(np.argmax(views["base"][q["key"]]) == np.argmax(P[q["line"]])) for q in qs)
    return {"questions": len(qs), "max_abs_logit_diff": max(d), "same_argmax": same}


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def tables(o: dict) -> str:
    res, boot = o["results"], o["bootstrap_vs_C0"]
    L = []
    for part, n_key in (("selection", "sel"), ("full", None)):
        n_attr = o["split"][n_key]["per_attribute"] if n_key else None
        hdr = "| config | " + " | ".join(NAMES[a] + (f" (n={n_attr[a]})" if n_attr else "") for a in ATTRS) + " | **7-attr mean** | diff vs C0 |"
        L += [f"### {part} {'half' if n_key else 'validation (dedup_v2, 5,822)'}", "", hdr, "|---" * (len(ATTRS) + 3) + "|"]
        for n, r in res.items():
            acc = r[part]
            d = 100 * (acc["mean"] - res["C0"][part]["mean"])
            L.append(f"| {n} | " + " | ".join(f"{100 * acc['per_attribute'][a]:.2f}" for a in ATTRS) + f" | **{100 * acc['mean']:.2f}** | {d:+.2f} |")
        L.append("")
    L += ["### Paired bootstrap vs C0 on the selection half (points; 10,000 resamples, seed 0)", "",
          "| config | diff | film-clustered 95% interval (decides) | question-level 95% interval (secondary) |", "|---|---|---|---|"]
    for n, b in boot.items():
        f, q = b["film"], b["question"]
        L.append(f"| {n} | {f['diff_points']:+.2f} | [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}] | [{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}] |")
    d = o["decision"]
    L += ["", (f"Decision (Step A rule): top {d['top']}, winner **{d['winner']}** ({d['reason']}). "
               f"Distillation gate: {o['distillation_gate']['reason']}.")]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--repro-2b", action="store_true")
    a = ap.parse_args()
    if a.repro_2b:
        sys.exit(0 if repro_2b(a.n_boot)["ok"] else 1)
    sp = core.v2_split(core.load_val_rows(), core.shotqa_film_of)
    qs = sp["questions"]
    views, complete = load_views(qs, K.DATA / "views", K.VIEWS)
    sig = load_sig([(K.SIGLIP, K.SIGLIP_SHA), (K.SIGLIP_ADDED, K.SIGLIP_ADDED_SHA)])
    missing = [q["key"] for q in qs if q["key"] not in sig]
    if missing:
        raise SystemExit(f"{len(missing)} questions without Phase 3A scores")
    full = {v for v, c in complete.items() if c["done"] == c["needed"]}
    available = [n for n in K.RUNNABLE if needed_views(n) <= full]
    if "C0" not in available:
        raise SystemExit("base view incomplete: nothing can be ranked")
    out = {"time": now_iso(), "plan": K.PLAN, "model": f"{SC.REPO}@{SC.REVISION} + LoRA {K.ADAPTER_SHA[:16]}...", "val": "dedup_v2",
           "split": sp["info"], "views_complete": complete, "available_configs": available,
           "not_run": [n for n in C.ORDER if n not in available], **rank(qs, views, sig, available, a.n_boot)}
    out["distillation_gate"] = core.distill_gate(out["decision"])
    out["c0_parity_vs_stepc_predictions"] = c0_parity(qs, views)
    out["c0_full_equals_stepc_primary"] = abs(100 * out["results"]["C0"]["full"]["mean"] - K.STEPC_PRIMARY) < 1e-9
    out["phase3a_alone_exploratory"] = {"selection": C.accuracy([q for q in qs if q["half"] == "sel"], sig), "full": C.accuracy(qs, sig)}
    out["gpu_hours"], out["code_commit"] = a.gpu_hours, git_head()
    K.RESULTS.mkdir(parents=True, exist_ok=True)
    (K.RESULTS / "val.json").write_text(json.dumps(out, indent=1) + "\n")
    (K.RESULTS / "val_tables.md").write_text(tables(out))
    print(json.dumps({"decision": out["decision"], "gate": out["distillation_gate"]}, indent=1))


if __name__ == "__main__":
    main()
