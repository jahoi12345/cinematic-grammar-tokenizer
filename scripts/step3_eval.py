"""Step 3 validation of one model (inference only, docs/plans/step3_gap.md sections 4-5), built from scripts/stepc_eval.py (not
changed): the letter-logit scorer of Phase 3B/Step B/Step C unchanged (cgt.p3b_vlm.score.evaluate_samples/summarize).
  --cand G0|G1|G2   a Step 3 checkpoint (must be registered by sha256 in --manifest); images at the candidate's TRAINING budget
  --cand base       the plain Step C 4B (INIT_ADAPTER, sha256 aba56883... checked); 1x images
  default           the FULL dedup_v2 ShotQA film-held-out validation set (5,822 q)
  --proxy           the robustness proxy: the fixed 1,400-question subset with re-sampled + JPEG-50 images (reported only)
  --limit N         pipeline check: the first N questions, TIMING ONLY (no accuracy)
Per-question results go to data/cache/step3/predictions/ (gitignored); aggregates to <out>.summary.json.
Never reads ShotBench, CameraBench, gold or dev.
  HF_HUB_OFFLINE=1 nice -n 19 <venv>/bin/python scripts/step3_eval.py --cand G1 --adapter <ckpt> --manifest <m.json> --out <prefix>"""
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

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize
from cgt.step3 import config as C
from cgt.step3.data import proxy_image, proxy_subset, val_dedup_v2, val_image
from cgt.stepb.evaluate import shotqa_points


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def run(a) -> None:
    adapter = Path(a.adapter) if a.adapter else C.INIT_ADAPTER
    if a.cand == "base":
        sha = guards.sha256_file(adapter)
        if sha != C.INIT_SHA:
            raise guards.AdapterHashError(f"baseline adapter {adapter} does not match {C.INIT_SHA}")
    else:
        sha = guards.require_registered_hash(Path(a.manifest), adapter)
    lines, val = val_dedup_v2()
    pos = list(range(len(val)))
    if a.proxy:
        pos = proxy_subset(val)
    if a.limit:
        pos = pos[: a.limit]
    img = proxy_image if a.proxy else val_image
    samples = [{**val[p], "images": [img(val[p], a.cand)]} for p in pos]
    guards.require_images_readable(sorted({s["images"][0] for s in samples}))
    t_load = time.perf_counter()
    model, processor = load(pinned_model_path(C.REPO, C.REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    image_tokens = C.CANDIDATES[a.cand]["image_tokens"] if a.cand in C.CANDIDATES else 144
    guards.require_token_lengths(processor, samples, C.MAX_SEQ, image_tokens)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=C.RANK, alpha=C.ALPHA, dropout=C.DROPOUT, verbose=False)
    model.load_weights(list(mx.load(str(adapter)).items()), strict=False)
    model.eval()
    load_s = time.perf_counter() - t_load
    ds = VisionDataset(ListView([]), cfg, processor, None, True)
    t0 = time.perf_counter()
    res = evaluate_samples(model, ds, samples, ids4,
                           progress=lambda d, n: print(f"scored {d}/{n}", flush=True) if d % 500 == 0 else None)
    secs = time.perf_counter() - t0
    if guards.sha256_file(adapter) != sha:
        raise guards.AdapterHashError("adapter changed during evaluation")
    out = {"cand": a.cand, "adapter": str(adapter), "adapter_sha256": sha, "model": f"{C.REPO}@{C.REVISION}",
           "val": "dedup_v2 proxy subset (re-sampled x0.6 + JPEG 50)" if a.proxy else "dedup_v2", "val_sha256": C.VAL_SHA,
           "budget_px": C.CANDIDATES[a.cand]["budget_px"] if a.cand in C.CANDIDATES else C.BUDGET_1X,
           "n": len(res), "load_seconds": round(load_s, 1), "seconds": round(secs, 1),
           "sec_per_question": round(secs / max(len(res), 1), 4), "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
           "mlx_env_seen": {k: os.environ.get(k) for k in C.MLX_ENV}}
    if a.limit:
        out["note"] = "pipeline check: timing only, accuracy deliberately not computed"
    else:
        want = C.PROXY_PER_ATTR * 7 if a.proxy else C.N_VAL_V2
        if len(res) != want:
            raise guards.DataRootViolation(f"scored {len(res)} questions, expected {want}")
        out["shotqa_dedup_v2"] = shotqa_points(summarize(res))
        pred_dir = C.DATA / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pf = pred_dir / f"{Path(a.out).name}_{sha[:12]}.jsonl"
        pf.write_text("\n".join(json.dumps({**r, "line": lines[p]}) for r, p in zip(res, pos, strict=True)) + "\n")
        out["predictions_file"] = str(pf)
        out["predictions_sha256"] = guards.sha256_file(pf)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "shotqa_dedup_v2"}, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=[*C.CANDIDATES, "base"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="pipeline check only (timing, no accuracy)")
    a = ap.parse_args()
    if a.cand != "base" and not (a.adapter and a.manifest):
        ap.error("--adapter and --manifest are required for a Step 3 candidate")
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


if __name__ == "__main__":
    main()
