"""Phase 3B gating benchmark, MLX: N LoRA training steps with mlx-vlm's own train() loop on ShotQA film-split TRAIN chat samples.
Records sec/step (step 1 and steps 2..N), peak memory, samples/s, loss curve, image-token count. No ShotBench/gold/dev/films.
Usage: uv run python scripts/p3b_bench_mlx.py --model Qwen/Qwen2-VL-2B-Instruct --pixels 112896 --multiple 28 --batch-size 2"""
import argparse
import contextlib
import io
import json
import re
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from datasets import Dataset
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, _dataset_image_token_id, train
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.p3b_vlm.images import shrink_to_budget
from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"Iter (\d+): Train loss ([0-9.eE+-]+), .*It/sec ([0-9.]+), Tokens/sec ([0-9.]+), Trained Tokens (\d+), Peak mem ([0-9.]+) GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--pixels", type=int, default=336 * 336, help="image pixel budget (area), aspect ratio kept")
    ap.add_argument("--multiple", type=int, default=28, help="image side multiple (28 for Qwen2-VL family)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--ip-attrs", default=None, help="JSON attributes set on processor.image_processor (e.g. SmolVLM no image splitting)")
    ap.add_argument("--stock-loss", action="store_true", help="use mlx-vlm's own loss (leaks the answer for some models; timing only)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"mlx_{a.model.split('/')[-1]}_{int(a.pixels ** 0.5)}px_bs{a.batch_size}"
    out = ROOT / "results/phase3b/raw"
    out.mkdir(parents=True, exist_ok=True)

    need = a.steps * a.batch_size
    rows = [json.loads(x) for x in (ROOT / "data/cache/p3b/bench_subset.jsonl").read_text().splitlines()][:need]
    assert len(rows) == need, f"bench subset has only {len(rows)} rows, need {need}"
    cache = ROOT / "data/cache/p3b/bench_images" / f"{a.pixels}_{a.multiple}"
    items = []
    for r in rows:
        dst = cache / (Path(r["images"][0]).stem + ".jpg")
        items.append({"images": [str(shrink_to_budget(r["images"][0], dst, a.pixels, a.multiple))], "messages": r["messages"]})
    ds = Dataset.from_list(items)

    t0 = time.perf_counter()
    model, processor = load(a.model, processor_config={"trust_remote_code": True})
    for k, v in (json.loads(a.ip_attrs) if a.ip_attrs else {}).items():
        setattr(processor.image_processor, k, v)
    load_s = time.perf_counter() - t0
    train_ds = VisionDataset(ds, model.config.__dict__, processor, image_resize_shape=None, train_on_completions=True)
    probe = [train_ds.process(it) for it in items[:20]]
    tok_id = _dataset_image_token_id(train_ds)
    img_tokens = [int((np.array(x["input_ids"]).reshape(-1) == tok_id).sum()) for x in probe]
    img_tok = int(np.median(img_tokens))
    seq_lens = [int(np.array(x["input_ids"]).reshape(-1).shape[0]) for x in probe]
    sup_tokens = [int(np.array(x["completion_mask"]).sum()) for x in probe]
    leak = leakage_probe(model, probe[0])
    print("leakage probe (max logit change before the altered token):", leak, flush=True)
    del probe

    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=a.lora_rank, alpha=a.lora_alpha,
                           dropout=a.lora_dropout, verbose=False)
    n_trainable = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.trainable_parameters()))
    n_total = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
    mx.reset_peak_memory()
    args = TrainingArgs(batch_size=a.batch_size, iters=a.steps, steps_per_report=1, steps_per_eval=10**9, steps_per_save=10**9,
                        val_batches=0, max_seq_length=a.max_seq, adapter_file=str(out / f"{tag}_adapter.safetensors"),
                        grad_checkpoint=not a.no_grad_checkpoint, learning_rate=a.lr, grad_clip=None,
                        gradient_accumulation_steps=1, full_finetune=False)
    buf = io.StringIO()
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        kw = {} if a.stock_loss else {"loss_fn": causal_loss_fn}
        train(model=model, optimizer=optim.Adam(learning_rate=a.lr), train_dataset=train_ds, val_dataset=None, args=args,
              train_on_completions=True, **kw)
    wall = time.perf_counter() - t1
    (out / f"{tag}.log").write_text(ANSI.sub("", buf.getvalue()))
    recs = [LINE.search(ANSI.sub("", ln)) for ln in buf.getvalue().splitlines()]
    recs = [m for m in recs if m]
    sec = [1.0 / float(m.group(3)) for m in recs]
    loss = [float(m.group(2)) for m in recs]
    peak_gb = max(float(m.group(6)) for m in recs)
    steady = sec[1:] if len(sec) > 1 else sec
    res = {"tag": tag, "framework": "mlx-vlm", "model": a.model, "steps": len(recs), "batch_size": a.batch_size,
           "pixel_budget": a.pixels, "image_tokens_per_image": img_tok, "image_tokens_min_mean_max_20": [min(img_tokens), float(np.mean(img_tokens)), max(img_tokens)], "supervised_tokens_per_sample_mean": float(np.mean(sup_tokens)), "mean_seq_len_tokens": float(np.mean(seq_lens)),
           "max_seq_len_allowed": a.max_seq, "ip_attrs": a.ip_attrs, "loss": "mlx-vlm stock" if a.stock_loss else "causal_loss_fn (mask=None)", "leakage_probe": leak, "grad_checkpoint": not a.no_grad_checkpoint,
           "lora": {"rank": a.lora_rank, "alpha": a.lora_alpha, "dropout": a.lora_dropout, "trainable_params": int(n_trainable),
                    "total_params": int(n_total)},
           "sec_step1": sec[0], "sec_per_step_median": statistics.median(steady), "sec_per_step_mean": float(np.mean(steady)),
           "samples_per_sec": a.batch_size / statistics.median(steady), "wall_s_train": wall, "model_load_s": load_s,
           "peak_mem_gb_mlx": peak_gb, "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
           "loss_first10_mean": float(np.mean(loss[:10])), "loss_last10_mean": float(np.mean(loss[-10:])), "loss_curve": loss}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=1))


if __name__ == "__main__":
    main()
