"""Base-generation experiment: validation of ONE checkpoint of either arm (inference only, docs/plans/base_gen_matched_budget.md
section 5). FULL dedup_v2 ShotQA film-held-out validation set (5,822 questions), letter-logit scorer of Phase 3B / Step C unchanged
(cgt.p3b_vlm.score.evaluate_samples/summarize); the adapter must be registered (sha256) in its run manifest and must not change.
  --arm 3b: Qwen2.5-VL-3B-Instruct + data/cache/base_gen/runs/base_gen_3b/ckpt_<pct>.safetensors, images 112,896 px / 28
  --arm 2b: Qwen3-VL-2B-Instruct + the Phase 3B run qwen3vl2b_lora_filmsplit_run1 ckpt_<pct>.safetensors (read only), images 147,456 px / 32
Per-question results -> data/cache/base_gen/predictions/ (gitignored); aggregates -> results/base_gen/val_<arm>_ckpt<pct>.summary.json.
--limit N: first N questions, TIMING ONLY (no accuracy). --adapter none (with --limit only): base model, pipeline/timing check.
Never reads ShotBench, CameraBench, gold or dev.
  HF_HUB_OFFLINE=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=10 nice -n 19 <venv>/bin/python scripts/base_gen_eval.py --arm 3b --pct 25"""
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

from cgt.base_gen import config as C
from cgt.p3b_vlm import guards
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize
from cgt.stepb.evaluate import shotqa_points
from cgt.stepc import config as SC
from cgt.stepc.data import cached_image, val_dedup_v2


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def arm_spec(arm: str, pct: int) -> dict:
    if arm == "3b":
        return {"repo": C.REPO_3B, "rev": C.REVISION_3B, "img": C.IMG_CACHE_3B,
                "adapter": C.DATA / "runs/base_gen_3b" / f"ckpt_{pct:03d}.safetensors",
                "manifest": C.RESULTS / "runs/base_gen_3b/manifest.json"}
    return {"repo": C.REPO_2B, "rev": C.REVISION_2B, "img": C.IMG_CACHE_2B,
            "adapter": C.RUN_2B_DIR / f"ckpt_{pct:03d}.safetensors", "manifest": C.MANIFEST_2B}


def run(a) -> None:
    spec = arm_spec(a.arm, a.pct)
    adapter = None if a.adapter == "none" else Path(a.adapter or spec["adapter"])
    if adapter is None and not a.limit:
        raise SystemExit("--adapter none is a pipeline/timing check: it needs --limit")
    sha = guards.require_registered_hash(Path(a.manifest or spec["manifest"]), adapter) if adapter else None
    lines, val = val_dedup_v2()
    if a.limit:
        lines, val = lines[: a.limit], val[: a.limit]
    samples = [{**s, "images": [cached_image(s["images"][0], spec["img"])]} for s in val]
    guards.require_images_readable(sorted({s["images"][0] for s in samples}))
    t_load = time.perf_counter()
    model, processor = load(pinned_model_path(spec["repo"], spec["rev"]))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=C.RANK, alpha=C.ALPHA, dropout=C.DROPOUT, verbose=False)
    if adapter:
        model.load_weights(list(mx.load(str(adapter)).items()), strict=False)
    model.eval()
    load_s = time.perf_counter() - t_load
    ds = VisionDataset(ListView([]), cfg, processor, None, True)
    t0 = time.perf_counter()
    res = evaluate_samples(model, ds, samples, ids4,
                           progress=lambda d, n: print(f"scored {d}/{n}", flush=True) if d % 500 == 0 else None)
    secs = time.perf_counter() - t0
    if adapter and guards.sha256_file(adapter) != sha:
        raise guards.AdapterHashError("adapter changed during evaluation")
    out = {"arm": a.arm, "pct": a.pct, "adapter": str(adapter), "adapter_sha256": sha, "model": f"{spec['repo']}@{spec['rev']}",
           "image_cache": str(spec["img"]), "val": "dedup_v2", "val_sha256": SC.VAL_SHA, "n": len(res), "load_seconds": round(load_s, 1),
           "seconds": round(secs, 1), "sec_per_question": round(secs / max(len(res), 1), 4), "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
           "mlx_env_seen": {k: os.environ.get(k) for k in C.MLX_ENV}}
    if a.limit:
        out["note"] = "pipeline check: timing only, accuracy deliberately not computed"
    else:
        if len(res) != SC.N_VAL_V2:
            raise guards.DataRootViolation(f"scored {len(res)} questions, expected {SC.N_VAL_V2}")
        out["shotqa_dedup_v2"] = shotqa_points(summarize(res))
        pred_dir = C.DATA / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pf = pred_dir / f"val_{a.arm}_ckpt{a.pct:03d}_{sha[:12]}.jsonl"
        pf.write_text("\n".join(json.dumps({**r, "line": n}) for r, n in zip(res, lines, strict=True)) + "\n")
        out["predictions_file"] = str(pf)
        out["predictions_sha256"] = guards.sha256_file(pf)
    dst = Path(a.out or C.RESULTS / f"val_{a.arm}_ckpt{a.pct:03d}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Path(str(dst) + ".summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "shotqa_dedup_v2"}, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("3b", "2b"))
    ap.add_argument("--pct", type=int, choices=(25, 50), default=25)
    ap.add_argument("--adapter", default=None, help="default: the arm's ckpt_<pct>; 'none' = base model (with --limit only)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="pipeline check only (timing, no accuracy)")
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


if __name__ == "__main__":
    main()
