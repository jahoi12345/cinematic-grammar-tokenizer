"""Evaluate Qwen3-VL-2B (+ optional LoRA adapter) on ShotQA film-split VALIDATION questions by letter logits (inference only).
Never reads ShotBench, gold, dev or our films. Fail closed (exit codes as scripts/p3b_train.py): 5 image outside ShotQA, 6 unreadable image,
7 adapter missing/not registered in --manifest. Usage:
  uv run python scripts/p3b_eval_val.py --adapter data/cache/p3b/runs/<tag>/ckpt_100.safetensors --n 500 --out results/phase3b/runs/<tag>/val_ckpt100
  uv run python scripts/p3b_eval_val.py --adapter none --n 500 --out ...     (base model, no adapter)"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import shrunk_path
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
IMG_CACHE = ROOT / "data/cache/p3b/images_147456_32"


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=None, help="subset size (default: the full validation set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="run manifest.json: the adapter's sha256 must be registered there (selection/freeze runs)")
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


def run(a) -> None:
    if a.adapter != "none" and a.manifest:
        guards.require_registered_hash(Path(a.manifest), Path(a.adapter))
    elif a.adapter != "none":
        guards.sha256_file(Path(a.adapter))
    val = [json.loads(x) for x in (ROOT / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    if a.n:
        val = [val[i] for i in sorted(np.random.default_rng(0).permutation(len(val))[: a.n])]
    guards.require_samples_under_root(val)
    try:
        items = [{**s, "images": [shrunk_path(s["images"][0], IMG_CACHE)]} for s in val]
    except (OSError, ValueError) as e:
        raise guards.UnreadableImage(str(e)) from e
    guards.require_images_readable([i["images"][0] for i in items])
    model, processor = load(pinned_model_path(REPO, REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    adapter_sha = None
    if a.adapter != "none":
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
        model.load_weights(list(mx.load(a.adapter).items()), strict=False)
        adapter_sha = guards.sha256_file(Path(a.adapter))
    model.eval()
    ds = VisionDataset(ListView([]), cfg, processor, None, True)
    t0 = time.perf_counter()
    res = evaluate_samples(model, ds, items, ids4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summ = summarize(res)
    summ.update({"adapter": a.adapter, "adapter_sha256": adapter_sha, "questions": len(res), "seconds": time.perf_counter() - t0,
                 "sec_per_question": (time.perf_counter() - t0) / len(res), "answer_token_ids": ids4})
    Path(str(out) + ".summary.json").write_text(json.dumps(summ, indent=1))
    Path(str(out) + ".predictions.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
    print(json.dumps({k: (round(v["acc"], 4) if isinstance(v, dict) and "acc" in v else v) for k, v in summ["per_attribute"].items()}))
    print(f"mean acc {summ['mean_acc']:.4f} on {len(res)} questions, {summ['sec_per_question']:.3f} s/question")


if __name__ == "__main__":
    main()
