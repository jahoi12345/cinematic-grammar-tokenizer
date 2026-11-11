"""Post-training step of docs/phase3b_plan.md sections 6, 7 and 9 (INFERENCE ONLY; ShotBench is never touched):
evaluate every saved named checkpoint on the FULL film-split validation set, compare with the Phase 3A heads on the same questions,
apply the pre-registered selection rule (highest mean over the 7 attributes of option-restricted accuracy; ties go to the later checkpoint),
and write results/phase3b/frozen_model.md plus results/phase3b/frozen_manifest.json (adapter SHA-256 taken here, before any ShotBench access).
Usage: uv run python scripts/p3b_validate_freeze.py --tag <tag> [--val-n N --out-dir DIR  (pipeline check only)]"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import shrunk_path
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import now_iso, plan_original_unchanged, write_status
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
IMG_CACHE = ROOT / "data/cache/p3b/images_147456_32"
ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
NAMES = {"size": "shot size", "lens": "lens size", "angle": "camera angle", "framing": "shot framing", "light_type": "lighting type",
         "light_cond": "lighting conditions", "composition": "composition"}
A3 = ROOT / "results/phase3/runs/siglip_base_filmsplit_lr1e-2_ep300/prior_only_baseline_val.csv"  # 3A frozen heads on the same film-val questions
LEDGER = ROOT / "results/phase3/shotbench_evaluated.json"
NEW_TAG = "qwen3vl2b_lora_filmsplit"


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a) -> None:
    run_dir = ROOT / "data/cache/p3b/runs" / a.tag
    logd = ROOT / "results/phase3b/runs" / a.tag
    manifest = json.loads((logd / "manifest.json").read_text())
    guards.verify_manifest_files(manifest, run_dir)  # every checkpoint must still match its recorded sha256
    ledger_before = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    if NEW_TAG in ledger_before:
        raise guards.FailClosed(f"ledger already contains {NEW_TAG}: ShotBench was touched before the freeze")
    ledger_sha_before = sha(LEDGER)
    val = [json.loads(x) for x in (ROOT / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    if a.val_n:  # pipeline check only: a seeded subset of the validation questions and a separate output directory
        import numpy as np
        val = [val[i] for i in sorted(np.random.default_rng(0).permutation(len(val))[: a.val_n])]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    guards.require_samples_under_root(val)
    items = [{**s, "images": [shrunk_path(s["images"][0], IMG_CACHE)]} for s in val]
    guards.require_images_readable([i["images"][0] for i in items])
    model, processor = load(pinned_model_path(REPO, REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
    model.eval()
    ds = VisionDataset(ListView([]), cfg, processor, None, True)
    pdir = run_dir / "val_predictions"
    pdir.mkdir(exist_ok=True)
    summaries, t_eval = {}, {}
    for pct in sorted(int(k) for k in manifest["checkpoints"]):
        ck = manifest["checkpoints"][str(pct)]
        file = run_dir / ck["file"]
        model.load_weights(list(mx.load(str(file)).items()), strict=False)
        t0 = time.perf_counter()
        res = evaluate_samples(model, ds, items, ids4)
        t_eval[pct] = time.perf_counter() - t0
        summaries[pct] = summarize(res)
        pf = pdir / f"ckpt_{pct:03d}.predictions.jsonl"
        pf.write_text("\n".join(json.dumps(r) for r in res) + "\n")
        summaries[pct]["predictions_sha256"] = sha(pf)
        summaries[pct]["adapter_sha256"] = guards.sha256_file(file)
        (logd / f"val_full_ckpt{pct:03d}.summary.json").write_text(json.dumps(summaries[pct], indent=1))
        print(f"checkpoint {pct}%: mean acc {summaries[pct]['mean_acc']:.4f} ({t_eval[pct]:.0f}s)", flush=True)
        write_status(logd, a.tag, state=f"validating checkpoints on the full film-split validation set ({pct}% done)")

    # selection rule (plan section 7): highest mean; ties -> later checkpoint
    best = max(summaries, key=lambda p: (round(summaries[p]["mean_acc"], 12), p))
    sel = summaries[best]
    a3 = pd.read_csv(A3).set_index("attribute")
    for at in ATTRS:  # same question set as 3A (full runs only)
        if not a.val_n and int(a3.loc[at, "n_val"]) != sel["per_attribute"][at]["n"]:
            raise guards.FailClosed(f"validation question counts differ from Phase 3A for {at}")
    a3acc = {at: float(a3.loc[at, "frozen_model_val_acc"]) for at in ATTRS}
    a3mean = sum(a3acc.values()) / 7
    prov = [json.loads(x) for x in (logd / "provenance.jsonl").read_text().splitlines()]
    sup = [json.loads(x) for x in (logd / "supervisor.jsonl").read_text().splitlines()] if (logd / "supervisor.jsonl").exists() else []
    restarts = len([x for x in (logd / "restarts.jsonl").read_text().splitlines() if x.strip()]) if (logd / "restarts.jsonl").exists() else 0
    plan = plan_original_unchanged()
    selected_file = run_dir / manifest["checkpoints"][str(best)]["file"]
    digest = guards.sha256_file(selected_file)
    assert digest == manifest["checkpoints"][str(best)]["sha256"] == sel["adapter_sha256"]
    ledger_sha_after = sha(LEDGER)
    frozen = {"tag": a.tag, "shotbench_ledger_tag_for_the_evaluation": NEW_TAG, "selected_checkpoint_percent": best,
              "selected_update": manifest["checkpoints"][str(best)]["update"], "adapter_file": selected_file.name,
              "adapter_sha256": digest, "hash_taken_at": now_iso(), "taken_before_any_shotbench_access": True,
              "shotbench_ledger_sha256_before_and_after": [ledger_sha_before, ledger_sha_after],
              "shotbench_ledger_entries": sorted(json.loads(LEDGER.read_text())), "plan": plan,
              "code_commits_seen_in_provenance": sorted({p["code_commit"] for p in prov}), "restarts": restarts,
              "supervisor_events": [e.get("event") for e in sup],
              "selection": {str(p): round(summaries[p]["mean_acc"], 6) for p in summaries},
              "validation_questions": sel["n"], "predictions_sha256": {str(p): summaries[p]["predictions_sha256"] for p in summaries},
              "all_checkpoint_sha256": {str(p): summaries[p]["adapter_sha256"] for p in summaries}}
    (out_dir / "frozen_manifest.json").write_text(json.dumps(frozen, indent=1))

    def pc(x):
        return f"{100 * x:.1f}"
    rows = [f"| {NAMES[at]} | {sel['per_attribute'][at]['n']} | " + " | ".join(pc(summaries[p]["per_attribute"][at]["acc"]) for p in sorted(summaries)) +
            f" | {pc(a3acc[at])} | {100 * (sel['per_attribute'][at]['acc'] - a3acc[at]):+.1f} |" for at in ATTRS]
    ci = []
    for at in ATTRS:
        v = sel["per_attribute"][at]
        lo, hi = wilson(v["correct"], v["n"])
        ci.append(f"| {NAMES[at]} | {v['correct']}/{v['n']} | {100 * v['acc']:.1f} ({lo:.1f}-{hi:.1f}) | {v['n_compound']} | "
                  f"{'-' if v['acc_single'] is None else pc(v['acc_single'])} | {'-' if v['acc_compound'] is None else pc(v['acc_compound'])} |")
    cks = sorted(summaries)
    hdr = " | ".join(f"ckpt {p}% (update {manifest['checkpoints'][str(p)]['update']})" for p in cks)
    check_note = f"\n**PIPELINE CHECK ONLY: {a.val_n} validation questions, not a freeze.**\n" if a.val_n else ""
    txt = f"""# Frozen Phase 3B model: Qwen3-VL-2B LoRA, selected on film-split validation (inference only; ShotBench not touched)
Written {now_iso()} by `scripts/p3b_validate_freeze.py`.{check_note} Run tag `{a.tag}`; pre-registered plan `docs/phase3b_plan.md` at commit `{plan['plan_commit']}` (original plan text unchanged: {plan['unchanged']}).

## Selected checkpoint
- **Selected: the {best}% checkpoint (update {frozen['selected_update']})**, adapter file `{selected_file.name}`, **SHA-256 `{digest}`**.
- Rule (plan section 7, fixed in advance): highest unweighted mean over the 7 attributes of option-restricted accuracy on the full {sel['n']}-question film-split validation set among the saved checkpoints; ties go to the later one. Means: {', '.join(f'{p}%: {100 * summaries[p]['mean_acc']:.2f}' for p in cks)}.
- Hash taken at {frozen['hash_taken_at']}, **before any ShotBench access**: the ShotBench ledger `results/phase3/shotbench_evaluated.json` has entries {frozen['shotbench_ledger_entries']} (none for `{NEW_TAG}`) and its SHA-256 was `{ledger_sha_before[:16]}...` before and `{ledger_sha_after[:16]}...` after this step. Manifest: `results/phase3b/frozen_manifest.json`.
- Code commits recorded in the run provenance: {', '.join('`' + c[:10] + '`' for c in frozen['code_commits_seen_in_provenance'])}; supervisor restarts: {restarts}.

## Film-split validation, all saved checkpoints, and the Phase 3A heads on the SAME {sel['n']} questions (option-restricted accuracy %, from letter logits)
| attribute | n | {hdr} | Phase 3A frozen-SigLIP heads | selected minus 3A (points) |
|---|---|---|---|---|
{chr(10).join(rows)}
| **unweighted mean** | {sel['n']} | {' | '.join(f'**{100 * summaries[p]['mean_acc']:.1f}**' for p in cks)} | **{100 * a3mean:.1f}** | **{100 * (sel['mean_acc'] - a3mean):+.1f}** |

Phase 3A numbers are the seed-ensemble of the selected heads of `siglip_base_filmsplit_lr1e-2_ep300` on the same film-held-out questions (`results/phase3/runs/.../prior_only_baseline_val.csv`). This comparison is observational (different model families, training recipes and scoring: the VLM reads letter logits; the probe ranks label sets). It does not influence the selection.

## Selected checkpoint with nominal 95% Wilson intervals (integer counts; film clustering not modelled) and answer strata
| attribute | correct/n | accuracy % (interval) | label-set answers | acc, single-label answers % | acc, label-set answers % |
|---|---|---|---|---|---|
{chr(10).join(ci)}

Chance for the 4-way task is 25%. Validation numbers are ShotQA-internal (film-held-out) and are not comparable to ShotBench numbers.
"""
    (out_dir / "frozen_model.md").write_text(txt)
    write_status(logd, a.tag, state=f"FROZEN: selected checkpoint {best}% (sha256 {digest[:16]}...). Waiting for the operator; no ShotBench was run.")
    print(txt[:1500])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--val-n", type=int, default=None, help="pipeline check only: evaluate a seeded subset of this many questions")
    ap.add_argument("--out-dir", default=str(ROOT / "results/phase3b"), help="where frozen_model.md / frozen_manifest.json are written")
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        write_status(ROOT / "results/phase3b/runs" / a.tag, a.tag, state=f"validate/freeze STOPPED: {type(e).__name__}", note=str(e))
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
