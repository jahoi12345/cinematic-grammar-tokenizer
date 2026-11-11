"""Step C validation of one 4B checkpoint (inference only, docs/plans/stepc_qwen3vl4b.md section 4): the FULL dedup_v2 ShotQA
film-held-out validation set (5,822 questions, cgt.eval.val_dedup.dedup_v2), letter-logit scorer of Phase 3B/Step B unchanged
(cgt.p3b_vlm.score.evaluate_samples/summarize). The adapter must be registered (sha256) in the run manifest and must not change during
scoring. Per-question results go to data/cache/stepc/predictions/ (gitignored); the summary (aggregates) to <out>.summary.json.
--limit N (pipeline check only) scores the first N dedup_v2 questions and writes TIMING ONLY (no accuracy).
Never reads ShotBench, CameraBench, gold or dev.
  HF_HUB_OFFLINE=1 nice -n 19 <venv>/bin/python scripts/stepc_eval.py --adapter <ckpt> --manifest <manifest.json> --out <prefix>"""
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
from cgt.stepb.evaluate import shotqa_points
from cgt.stepc import config as C
from cgt.stepc.data import cached_image, val_dedup_v2


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def run(a) -> None:
    adapter = Path(a.adapter)
    sha = guards.require_registered_hash(Path(a.manifest), adapter)
    lines, val = val_dedup_v2()
    if a.limit:
        lines, val = lines[: a.limit], val[: a.limit]
    samples = [{**s, "images": [cached_image(s["images"][0])]} for s in val]
    guards.require_images_readable(sorted({s["images"][0] for s in samples}))
    t_load = time.perf_counter()
    model, processor = load(pinned_model_path(C.REPO, C.REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
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
    out = {"adapter": str(adapter), "adapter_sha256": sha, "model": f"{C.REPO}@{C.REVISION}", "val": "dedup_v2",
           "val_sha256": C.VAL_SHA, "n": len(res), "load_seconds": round(load_s, 1), "seconds": round(secs, 1),
           "sec_per_question": round(secs / max(len(res), 1), 4), "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
           "mlx_env_seen": {k: os.environ.get(k) for k in C.MLX_ENV}}
    if a.limit:
        out["note"] = "pipeline check: timing only, accuracy deliberately not computed"
    else:
        if len(res) != C.N_VAL_V2:
            raise guards.DataRootViolation(f"scored {len(res)} questions, expected {C.N_VAL_V2}")
        out["shotqa_dedup_v2"] = shotqa_points(summarize(res))
        pred_dir = C.DATA / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pf = pred_dir / f"{Path(a.out).name}_{sha[:12]}.jsonl"
        pf.write_text("\n".join(json.dumps({**r, "line": n}) for r, n in zip(res, lines, strict=True)) + "\n")
        out["predictions_file"] = str(pf)
        out["predictions_sha256"] = guards.sha256_file(pf)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "shotqa_dedup_v2"}, indent=1), flush=True)


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
