"""GRPO pilot validation (docs/plans/grpo_pilot.md section 3), built from scripts/step3_eval.py (not changed): the letter-logit scorer
of Phase 3B / Step C unchanged (cgt.p3b_vlm.score.evaluate_samples/summarize) on the dedup_v2 ShotQA film-held-out validation set
(5,822 q, 1x images). The adapter must be registered by sha256 in --manifest and is loaded with the fail-closed key check
(cgt.p3b_vlm.adapter_load.load_adapter_checked). --limit N: pipeline check, the first N questions, TIMING ONLY (no accuracy).
Per-question results go to data/cache/grpo_pilot/predictions/ (gitignored); aggregates to <out>.summary.json, with the descriptive
difference to the plain 4B (cgt.grpo.config.BASELINE, results/stepc/selection.json). Never reads ShotBench, CameraBench, gold or dev.
  HF_HUB_OFFLINE=1 nice -n 19 <venv>/bin/python scripts/grpo_eval.py --adapter <a.safetensors> --manifest <m.json> --out <prefix>"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.grpo import config as C
from cgt.p3b_vlm import guards
from cgt.p3b_vlm.adapter_load import load_adapter_checked
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize
from cgt.step3.data import val_dedup_v2, val_image
from cgt.stepb.evaluate import shotqa_points


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def compare_to_baseline(points: dict, base: dict = C.BASELINE) -> dict:
    """Descriptive differences (points) to the plain 4B: primary and per attribute. No eligibility rule (the pilot adopts nothing)."""
    return {"primary_delta": points["primary"] - base["primary"],
            "per_attribute_delta": {a: points["per_attribute"][a] - v for a, v in base["per_attribute"].items()}}


def run(a) -> None:
    adapter = Path(a.adapter)
    sha = guards.require_registered_hash(Path(a.manifest), adapter)
    lines, val = val_dedup_v2()
    pos = list(range(len(val)))[: a.limit] if a.limit else list(range(len(val)))
    samples = [{**val[p], "images": [val_image(val[p], "base")]} for p in pos]
    guards.require_images_readable(sorted({s["images"][0] for s in samples}))
    t_load = time.perf_counter()
    model, processor = load(pinned_model_path(C.REPO, C.REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    guards.require_token_lengths(processor, samples, C.MAX_SEQ, C.IMAGE_TOKENS)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=C.RANK, alpha=C.ALPHA, dropout=C.DROPOUT, verbose=False)
    n_t = load_adapter_checked(model, adapter)
    model.eval()
    load_s = time.perf_counter() - t_load
    ds = VisionDataset(ListView([]), cfg, processor, None, True)
    t0 = time.perf_counter()
    res = evaluate_samples(model, ds, samples, ids4,
                           progress=lambda d, n: print(f"scored {d}/{n}", flush=True) if d % 500 == 0 else None)
    secs = time.perf_counter() - t0
    if guards.sha256_file(adapter) != sha:
        raise guards.AdapterHashError("adapter changed during evaluation")
    m = json.loads(Path(a.manifest).read_text())["checkpoints"]
    upd = next((c.get("update") for c in m.values() if c.get("sha256") == sha), None)
    out = {"adapter": str(adapter), "adapter_sha256": sha, "adapter_update": upd, "tensors_matched": n_t,
           "model": f"{C.REPO}@{C.REVISION}", "val": "dedup_v2", "val_sha256": C.VAL_SHA, "n": len(res),
           "load_seconds": round(load_s, 1), "seconds": round(secs, 1), "sec_per_question": round(secs / max(len(res), 1), 4),
           "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "mlx_env_seen": {k: os.environ.get(k) for k in C.MLX_ENV}}
    if a.limit:
        out["note"] = "pipeline check: timing only, accuracy deliberately not computed"
    else:
        if len(res) != C.N_VAL_V2:
            raise guards.DataRootViolation(f"scored {len(res)} questions, expected {C.N_VAL_V2}")
        pts = shotqa_points(summarize(res))
        out["shotqa_dedup_v2"] = pts
        out["vs_plain_4b"] = {"baseline": C.BASELINE, **compare_to_baseline(pts),
                              "note": "descriptive only; the pilot adopts nothing"}
        pred_dir = C.DATA / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pf = pred_dir / f"{Path(a.out).name}_{sha[:12]}.jsonl"
        pf.write_text("\n".join(json.dumps({**r, "line": lines[p]}) for r, p in zip(res, pos, strict=True)) + "\n")
        out["predictions_file"] = str(pf)
        out["predictions_sha256"] = guards.sha256_file(pf)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("shotqa_dedup_v2", "vs_plain_4b")}, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="pipeline check only (timing, no accuracy)")
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


if __name__ == "__main__":
    main()
