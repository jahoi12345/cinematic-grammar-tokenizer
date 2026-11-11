"""Phase 4 video-LoRA GATING BENCHMARK (no real training run): N LoRA steps with mlx-vlm's own train() loop on ShotQA movement TRAIN clips, K frames per clip as K images, the
base Qwen3-VL-2B (pinned revision, NOT the Phase 3B adapter), the Phase 3B causal-safe loss and a mandatory leak probe (exit 4 unless without_mask is exactly 0.0). Labels come from
the locked MovementLabelStore (TRAIN rows only). No ShotBench movement, no CameraBench. Takes the GPU lock /tmp/cgt_gpu.lock for its duration (exit 11 if another job holds it).
Records sec/step (step 1 and steps 2..N), peak memory, visual tokens and sequence length per clip, loss behaviour; projects hours for 3 epochs on the 903 train clips.
Usage: uv run python scripts/p4_video_bench.py --frames 8 --tokens 64 --steps 50 --tag f8t64"""
import argparse
import contextlib
import io
import json
import os
import re
import resource
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from datasets import Dataset
from mlx.utils import tree_flatten
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, _dataset_image_token_id, train
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.labels import MovementLabelStore

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
LOCK = Path("/tmp/cgt_gpu.lock")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"Iter (\d+): Train loss ([0-9.eE+-]+), .*It/sec ([0-9.]+), Tokens/sec ([0-9.]+), Trained Tokens (\d+), Peak mem ([0-9.]+) GB")
N_TRAIN_CLIPS_WITH_ATOMS = 903


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64, help="per-frame image-token budget")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--job", default="p4-video-bench")
    a = ap.parse_args()
    if LOCK.exists():
        print(f"GPU lock held: {LOCK.read_text().strip()}", file=sys.stderr)
        raise SystemExit(11)
    LOCK.write_text(f"{a.job} pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')} tag={a.tag}\n")
    try:
        run(a)
    finally:
        LOCK.unlink(missing_ok=True)


def run(a) -> None:
    man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    store = MovementLabelStore(man["clips"]["val"])
    order = np.random.default_rng(0).permutation(len(man["clips"]["train"]))
    cands = [man["clips"]["train"][i] for i in order[: a.steps + 80]]
    trows = [r for r in store.rows(cands) if r["atoms"]][: a.steps + 20]  # TRAIN labels only (the store would raise on a validation clip)
    t0 = time.perf_counter()
    items = [V.build_sample(r, V.extract_frames(QA / r["clip"], a.frames, a.tokens, ROOT / "data/cache/p4/video_frames")) for r in trows]
    prep_s = time.perf_counter() - t0
    path = pinned_model_path()
    t0 = time.perf_counter()
    model, processor = load(path)
    V.configure_processor(processor)
    load_s = time.perf_counter() - t0
    cfg = model.config.__dict__
    ds = Dataset.from_list([{"images": it["images"], "messages": it["messages"]} for it in items[: a.steps]])
    train_ds = VisionDataset(ds, cfg, processor, image_resize_shape=None, train_on_completions=True)
    probe = [train_ds.process({"images": it["images"], "messages": it["messages"]}) for it in items[a.steps: a.steps + 20]]
    tok_id = _dataset_image_token_id(train_ds)
    vis = [int((np.array(x["input_ids"]).reshape(-1) == tok_id).sum()) for x in probe]
    seq = [int(np.array(x["input_ids"]).reshape(-1).shape[0]) for x in probe]
    leak = leakage_probe(model, probe[0])
    print("leakage probe:", leak, flush=True)
    if leak["without_mask"] != 0.0:
        raise SystemExit(4)
    del probe
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
    n_tr = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    mx.reset_peak_memory()
    out = ROOT / "results/phase4/video_bench"
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArgs(batch_size=1, iters=a.steps, steps_per_report=1, steps_per_eval=10**9, steps_per_save=10**9, val_batches=0, max_seq_length=a.max_seq,
                        adapter_file=str(ROOT / "data/cache/p4/video_bench_adapter.safetensors"), grad_checkpoint=True, learning_rate=a.lr, grad_clip=None,
                        gradient_accumulation_steps=1, full_finetune=False)
    buf = io.StringIO()
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        train(model=model, optimizer=optim.Adam(learning_rate=a.lr), train_dataset=train_ds, val_dataset=None, args=args, train_on_completions=True, loss_fn=causal_loss_fn)
    wall = time.perf_counter() - t1
    text = ANSI.sub("", buf.getvalue())
    (out / f"{a.tag}.log").write_text(text)
    recs = [m for m in (LINE.search(x) for x in text.splitlines()) if m]
    sec = [1.0 / float(m.group(3)) for m in recs]
    loss = [float(m.group(2)) for m in recs]
    steady = sec[1:] if len(sec) > 1 else sec
    med = statistics.median(steady)
    res = {"tag": a.tag, "frames": a.frames, "tokens_per_frame_budget": a.tokens, "steps": len(recs), "batch_size": 1, "model": "Qwen/Qwen3-VL-2B-Instruct (pinned, base, no adapter)",
           "visual_tokens_per_clip_median_min_max_20": [float(np.median(vis)), min(vis), max(vis)], "sequence_tokens_median_min_max_20": [float(np.median(seq)), min(seq), max(seq)],
           "skipped_examples_warning": "Skipping example" in text, "leakage_probe": leak, "lora_trainable_params": int(n_tr), "sec_step1": sec[0], "sec_per_step_median": med,
           "sec_per_step_mean": float(np.mean(steady)), "samples_per_sec": 1 / med, "peak_mem_gb_mlx": max(float(m.group(6)) for m in recs), "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
           "loss_first10_mean": float(np.mean(loss[:10])), "loss_last10_mean": float(np.mean(loss[-10:])), "loss_first": loss[0], "loss_finite": bool(np.isfinite(loss).all()), "loss_curve": loss,
           "wall_s_train": wall, "model_load_s": load_s, "frame_prep_s": prep_s, "projected_hours_3_epochs_903_clips_batch1": round(3 * N_TRAIN_CLIPS_WITH_ATOMS * med / 3600, 2),
           "projected_hours_1_epoch_903_clips_batch1": round(N_TRAIN_CLIPS_WITH_ATOMS * med / 3600, 2),
           "caveat": "extrapolation from a short batch-1 run with an update per step; the planned loop accumulates micro-batches (per-sample time is the quantity used); excludes validation, checkpoints, model loading"}
    (out / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=1))


if __name__ == "__main__":
    main()
