"""Phase 3B gating benchmark, torch + peft on MPS: N LoRA training steps on ShotQA film-split TRAIN chat samples (same samples,
image budget, LoRA settings and completion-only loss as scripts/p3b_bench_mlx.py). HF models are causal by construction.
Usage: uv run python scripts/p3b_bench_torch.py --model HuggingFaceTB/SmolVLM-500M-Instruct --batch-size 1 --ip-attrs '{...}'"""
import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from cgt.p3b_vlm.images import shrink_to_budget

ROOT = Path(__file__).resolve().parents[1]
TARGET = r".*(text_model|language_model).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def build_batch(proc, rows, device, dtype):
    """Right-padded batch with labels = -100 everywhere except the assistant answer (completion-only loss)."""
    texts, prompts, images = [], [], []
    for r in rows:
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}
        full = proc.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": a}]}], add_generation_prompt=False)
        prompt = proc.apply_chat_template([user], add_generation_prompt=True)
        texts.append(full)
        prompts.append(prompt)
        images.append([Image.open(r["images"][0]).convert("RGB")])
    enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
    pen = proc(text=prompts, images=images, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    plen = pen["attention_mask"].sum(1)
    for i in range(labels.shape[0]):
        labels[i, : int(plen[i])] = -100
    labels[enc["attention_mask"] == 0] = -100
    batch = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in enc.items()}
    batch["labels"] = labels.to(device)
    return batch, int((labels != -100).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--pixels", type=int, default=262144)
    ap.add_argument("--multiple", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--ip-attrs", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"torch_{a.model.split('/')[-1]}_bs{a.batch_size}"
    out = ROOT / "results/phase3b/raw"
    out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, a.dtype)
    need = a.steps * a.batch_size
    rows = [json.loads(x) for x in (ROOT / "data/cache/p3b/bench_subset.jsonl").read_text().splitlines()][:need]
    assert len(rows) == need
    cache = ROOT / "data/cache/p3b/bench_images" / f"{a.pixels}_{a.multiple}"
    for r in rows:
        r["images"] = [str(shrink_to_budget(r["images"][0], cache / (Path(r["images"][0]).stem + ".jpg"), a.pixels, a.multiple))]

    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "right"
    for k, v in (json.loads(a.ip_attrs) if a.ip_attrs else {}).items():
        setattr(proc.image_processor, k, v)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=dtype).to(dev)
    model = get_peft_model(model, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=TARGET, task_type="CAUSAL_LM"))
    if not a.no_grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    load_s = time.perf_counter() - t0
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    model.train()
    secs, losses, peaks, seq_lens, sup = [], [], [], [], []
    for s in range(a.steps):
        batch, n_sup = build_batch(proc, rows[s * a.batch_size:(s + 1) * a.batch_size], dev, dtype)
        torch.mps.synchronize()
        t = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        secs.append(time.perf_counter() - t)
        losses.append(float(loss))
        peaks.append(torch.mps.driver_allocated_memory() / 1e9)
        seq_lens.append(int(batch["input_ids"].shape[1]))
        sup.append(n_sup)
    steady = secs[1:]
    img_tok = None
    if "input_ids" in batch:
        tid = getattr(model.config, "image_token_id", None) or getattr(model.config, "image_token_index", None)
        if tid is not None:
            img_tok = int((batch["input_ids"][0] == tid).sum())
    res = {"tag": tag, "framework": "torch+peft (MPS)", "model": a.model, "steps": a.steps, "batch_size": a.batch_size,
           "pixel_budget": a.pixels, "image_tokens_per_image": img_tok, "mean_seq_len_tokens": float(np.mean(seq_lens)),
           "supervised_tokens_per_step_mean": float(np.mean(sup)), "dtype": a.dtype, "grad_checkpoint": not a.no_grad_checkpoint,
           "ip_attrs": a.ip_attrs, "lora": {"rank": a.lora_rank, "alpha": a.lora_alpha, "dropout": a.lora_dropout,
                                            "trainable_params": n_tr, "total_params": n_all},
           "sec_step1": secs[0], "sec_per_step_median": statistics.median(steady), "sec_per_step_mean": float(np.mean(steady)),
           "samples_per_sec": a.batch_size / statistics.median(steady), "model_load_s": load_s,
           "peak_mem_gb_mps_driver": max(peaks), "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
           "loss_first10_mean": float(np.mean(losses[:10])), "loss_last10_mean": float(np.mean(losses[-10:])), "loss_curve": losses}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=1))


if __name__ == "__main__":
    main()
