"""an earlier change torch TRAINING parity (docs/plans/torch_train_parity.md): the Step B LoRA recipe (scripts/stepb_train.py, control B0) in MLX vs
transformers + peft on MPS. Each mode runs in its own process (MLX and torch are never loaded together).
  --mode dump-mlx-inputs   CPU, mlx-vlm processor only: VisionDataset items of samples 0-47
  --mode compare-inputs    CPU, HF processor only: torch-native items vs the mlx-vlm items
  --mode mlx               GPU (MLX): (a) losses 0-31, (b) mean grad 0-15, (c) one Adam step at peak lr, losses 16-31 after it
  --mode torch             GPU (MPS): the same in torch
  --mode torch-updates     GPU (MPS): (d) N updates with dropout 0.05 and the B0 schedule
  --mode report            CPU: results/torch_train_parity/report.json (aggregates only)
Per-sample losses, gradients and parameter deltas stay under data/cache/torch_train_parity (gitignored). Never reads gold, dev, ShotBench,
CameraBench. GPU modes: /tmp/cgt_gpu.lock (wait if held), memory guard >= 25% free, 75-minute total GPU cap enforced across modes.
  nice -n 19 <venv>/bin/python scripts/torch_train_parity.py --mode mlx"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from cgt.p3b_vlm import torch_train as tt
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.worktrees import sibling

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
WT = ROOT.parent  # the directory holding the sibling worktrees
FROZEN = sibling("phase2") / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/ckpt_100.safetensors"
FROZEN_CFG = FROZEN.parent / "adapter_config.json"
FROZEN_SHA = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"
B0 = sibling("cinescale2") / "data/cache/stepb/B0.jsonl"
B0_SHA = json.loads((ROOT / "results/stepb/data_summary.json").read_text())["files"]["B0"]["sha256"]
B0_LOG = ROOT / "results/stepb/runs/stepb_B0/updates.jsonl"
CACHE = ROOT / "data/cache/torch_train_parity"
OUT = ROOT / "results/torch_train_parity/report.json"
RANK, ALPHA, SCALE, MAX_SEQ = 16, 32.0, 2.0, 1024
N_A, N_ACC, N_INPUTS = 32, 16, 48
GPU_CAP_S = 75 * 60
ALLOWED_IMG_ROOT = (sibling("cinescale2") / "data/cache/p3b/images_147456_32").resolve()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_samples(n: int) -> list[dict]:
    if sha256(B0) != B0_SHA:
        raise SystemExit(f"B0 stream sha mismatch: {B0}")
    out = []
    with open(B0) as f:
        for line in f:
            if len(out) >= n:
                break
            s = json.loads(line)
            if ALLOWED_IMG_ROOT not in Path(s["shrunk"]).resolve().parents:
                raise SystemExit(f"image outside the allowed shrunk cache: {s['shrunk']}")
            out.append(s)
    return out


def check_adapter() -> None:
    if sha256(FROZEN) != FROZEN_SHA:
        raise SystemExit("frozen adapter sha mismatch")


# ---------------------------------------------------------------- GPU budget, lock, memory
class Gpu:
    """GPU lock (wait if held) + memory guard + the 75-minute total cap shared by all GPU modes (CACHE/gpu_seconds.json)."""

    def __init__(self, job: str):
        from cgt.pipeline.gpulock import GpuLock

        self.lock = GpuLock(f"torch-train-parity {job}")
        self.f = CACHE / "gpu_seconds.json"
        self.used = 0.0

    def _read_used(self) -> float:
        return json.loads(self.f.read_text())["used"] if self.f.exists() else 0.0

    def __enter__(self):
        from cgt.cinescale2.semantic import wait_for_memory
        from cgt.pipeline.gpulock import LockHeld

        while True:
            try:
                self.lock.acquire()
                break
            except LockHeld as e:
                print(f"waiting: {e}", flush=True)
                time.sleep(30)
        self.used = self._read_used()  # read under the lock (a job that waited must see the previous holder's seconds)
        if self.used >= GPU_CAP_S:
            self.lock.release()
            raise SystemExit(f"GPU cap reached ({self.used / 60:.1f} min used)")
        wait_for_memory(25, log=lambda m: print(m, flush=True), poll_s=60)
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + GPU_CAP_S - self.used
        return self

    def check(self):
        if time.perf_counter() > self.deadline:
            raise SystemExit("GPU cap reached inside the run")

    def __exit__(self, *exc):
        el = time.perf_counter() - self.t0
        CACHE.mkdir(parents=True, exist_ok=True)
        self.f.write_text(json.dumps({"used": self.used + el}))
        self.lock.release()
        self.elapsed = el


# ---------------------------------------------------------------- inputs diagnostic (CPU)
def dump_mlx_inputs(samples):
    import mlx.core as mx
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.utils import load_config, load_image_processor, load_processor

    mx.set_default_device(mx.cpu)
    path = Path(pinned_model_path())
    proc = load_processor(path, True)
    ip = load_image_processor(path)
    if ip is not None:
        proc.image_processor = ip
    ds = VisionDataset(None, load_config(path), proc, None, True)
    d = CACHE / "mlx_inputs"
    d.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(samples):
        it = ds.process({"images": [s["shrunk"]], "messages": s["messages"]})
        np.savez(d / f"{k:04d}.npz", ids=np.array(it["input_ids"]).reshape(-1), cm=np.array(it["completion_mask"]).reshape(-1),
                 pix=np.array(it["pixel_values"].astype(mx.float32)), grid=np.array(it["image_grid_thw"]))
    return {"n": len(samples)}


def compare_inputs(samples):
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(pinned_model_path())
    ids_ok = cm_ok = grid_ok = 0
    pix_max, n_sup = 0.0, []
    for k, s in enumerate(samples):
        z = np.load(CACHE / "mlx_inputs" / f"{k:04d}.npz")
        it = tt.build_item(proc, s, Image.open(s["shrunk"]).convert("RGB"))
        ids_ok += np.array_equal(z["ids"], it["input_ids"][0].numpy())
        cm_ok += np.array_equal(z["cm"], it["completion_mask"][0].numpy())
        grid_ok += np.array_equal(z["grid"], it["image_grid_thw"].numpy())
        p = it["pixel_values"].float().numpy()
        pix_max = max(pix_max, float(np.abs(z["pix"] - p).max()) if z["pix"].shape == p.shape else float("inf"))
        n_sup.append(int(it["completion_mask"][0, 1:].sum()))
    return {"n": len(samples), "input_ids_identical": ids_ok, "completion_mask_identical": cm_ok, "image_grid_thw_identical": grid_ok,
            "pixel_values_max_absdiff": pix_max, "supervised_targets_per_sample": sorted(set(n_sup))}


# ---------------------------------------------------------------- MLX reference
def run_mlx(samples, dtype: str = "bfloat16"):
    """dtype bfloat16 = the reference run; float32 = diagnostic only (base cast to fp32, files suffixed _float32)."""
    sfx = "" if dtype == "bfloat16" else f"_{dtype}"
    from functools import partial

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import iterate_batches
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model, grad_checkpoint
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.loss import causal_loss_fn

    class LV:
        def __init__(self, x):
            self.x = x

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i]

    check_adapter()
    with Gpu("mlx") as g:
        mx.random.seed(0)
        t_load = time.perf_counter()
        model, processor = load(pinned_model_path())
        if dtype == "float32":
            model.set_dtype(mx.float32)
        cfg = model.config.__dict__
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=RANK, alpha=ALPHA, dropout=0.0, verbose=False)
        model.load_weights(list(mx.load(str(FROZEN)).items()), strict=False)
        load_s = time.perf_counter() - t_load
        for module in model.children().values():
            if hasattr(module, "layers"):
                grad_checkpoint(module.layers[0])
        loss_and_grad = nn.value_and_grad(model, partial(causal_loss_fn, train_on_completions=True))
        items = [{"images": [s["shrunk"]], "messages": s["messages"]} for s in samples[:N_A]]

        def batches(xs):
            return iterate_batches(VisionDataset(LV(xs), cfg, processor, None, True), 1, MAX_SEQ, train=False)

        model.train()
        mx.reset_peak_memory()
        pre, acc = [], None
        it = batches(items)
        t_up = time.perf_counter()
        for i in range(N_A):
            g.check()
            b = next(it)
            if i < N_ACC:
                loss, grad = loss_and_grad(model, b)
                acc = grad if acc is None else tree_map(lambda x, y: x + y, acc, grad)
                mx.eval(loss, acc)
                mx.clear_cache()
            else:
                loss = causal_loss_fn(model, b, train_on_completions=True)
                mx.eval(loss)
            pre.append(float(loss))
            if i == N_ACC - 1:
                acc = tree_map(lambda x: x / N_ACC, acc)
                grads = {k: np.array(v) for k, v in tree_flatten(acc)}
                clipped, gnorm = optim.clip_grad_norm(acc, tt.CLIP)
                before = {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}
                opt = optim.Adam(learning_rate=tt.PEAK_LR, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
                opt.init(model.trainable_parameters())
                # keep the pre-step parameters for losses 16-31 before the step: apply the step only after them
                sec_fb = time.perf_counter() - t_up
        # (c) the step, then losses 16-31 again
        t_s = time.perf_counter()
        opt.update(model, clipped)
        mx.eval(model.parameters(), opt.state)
        step_s = time.perf_counter() - t_s
        delta = {k: np.array(v) - before[k] for k, v in tree_flatten(model.trainable_parameters())}
        post = []
        it = batches(items[N_ACC:N_A])
        for _ in range(N_A - N_ACC):
            g.check()
            loss = causal_loss_fn(model, next(it), train_on_completions=True)
            mx.eval(loss)
            post.append(float(loss))
        peak = mx.get_peak_memory() / 1e9
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"mlx_grads{sfx}.npz", **grads)
    np.savez(CACHE / f"mlx_delta{sfx}.npz", **delta)
    meta = {"pre": pre, "post": post, "gnorm": float(gnorm), "sec_16_fwd_bwd": sec_fb, "sec_step": step_s, "load_s": load_s,
            "peak_gb": peak, "gpu_s": g.elapsed, "n_tensors": len(grads)}
    meta["dtype"] = dtype
    (CACHE / f"mlx{sfx}.json").write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("pre", "post")}, indent=1))


# ---------------------------------------------------------------- torch
def load_torch(dropout: float, grad_ckpt: bool, dtype: str = "bfloat16"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from cgt.p3b_vlm import torch_port as tp

    check_adapter()
    pdir = CACHE / f"peft_ckpt100_do{dropout}"
    conv = tp.write_peft_adapter(FROZEN, pdir, RANK, SCALE, FROZEN_CFG)
    cfg = json.loads((pdir / "adapter_config.json").read_text())
    cfg.update({"lora_dropout": dropout, "inference_mode": False})
    (pdir / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    path = pinned_model_path()
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=getattr(torch, dtype)).to("mps")
    model = PeftModel.from_pretrained(model, str(pdir), is_trainable=True)
    tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(tr) != conv["tensors"] or any("lora_" not in n for n, _ in tr):
        raise SystemExit(f"trainable tensors {len(tr)} != converted {conv['tensors']} (or a non-LoRA tensor is trainable)")
    if any(p.dtype != torch.float32 for _, p in tr):
        raise SystemExit("LoRA weights are not fp32")
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, proc, tr


def torch_item(proc, s, dtype: str = "bfloat16"):
    import torch
    from PIL import Image

    it = tt.build_item(proc, s, Image.open(s["shrunk"]).convert("RGB"))
    if it["input_ids"].shape[1] > MAX_SEQ:
        raise SystemExit(f"{s['id']}: {it['input_ids'].shape[1]} tokens > {MAX_SEQ}")
    return {k: (v.to("mps", getattr(torch, dtype)) if v.is_floating_point() else v.to("mps")) for k, v in it.items()}


def torch_loss(model, it):
    kw = {k: v for k, v in it.items() if k != "completion_mask"}
    out = model(**kw, use_cache=False)
    return tt.completion_loss(out.logits, it["input_ids"], it["completion_mask"])


def mem_gb():
    import torch

    return torch.mps.driver_allocated_memory() / 1e9


def run_torch(samples, grad_ckpt: bool, dtype: str = "bfloat16"):
    """dtype bfloat16 = the gated run; float32 = diagnostic only (files suffixed _float32)."""
    sfx = "" if dtype == "bfloat16" else f"_{dtype}"
    import torch

    with Gpu("torch") as g:
        t_load = time.perf_counter()
        model, proc, tr = load_torch(0.0, grad_ckpt, dtype)
        load_s = time.perf_counter() - t_load
        items = [torch_item(proc, s, dtype) for s in samples[:N_A]]
        pre, peak = [], 0.0
        torch.mps.synchronize()
        t_up = time.perf_counter()
        for i, it in enumerate(items):
            g.check()
            if i < N_ACC:
                loss = torch_loss(model, it)
                loss.backward()
            else:
                with torch.no_grad():
                    loss = torch_loss(model, it)
            pre.append(float(loss.detach()))
            peak = max(peak, mem_gb())
            if i == N_ACC - 1:
                for _, p in tr:
                    p.grad /= N_ACC
                grads = {tt.mlx_key_of_torch_param(n): p.grad.float().cpu().numpy().T.copy() for n, p in tr}
                torch.mps.synchronize()
                sec_fb = time.perf_counter() - t_up
        before = {n: p.detach().float().cpu().numpy().copy() for n, p in tr}
        t_s = time.perf_counter()
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in tr], tt.CLIP)
        opt = torch.optim.Adam([p for _, p in tr], lr=tt.PEAK_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        opt.step()
        torch.mps.synchronize()
        step_s = time.perf_counter() - t_s
        delta = {tt.mlx_key_of_torch_param(n): (p.detach().float().cpu().numpy() - before[n]).T.copy() for n, p in tr}
        post = []
        with torch.no_grad():
            for it in items[N_ACC:N_A]:
                g.check()
                post.append(float(torch_loss(model, it)))
        peak = max(peak, mem_gb())
    np.savez(CACHE / f"torch_grads{sfx}.npz", **grads)
    np.savez(CACHE / f"torch_delta{sfx}.npz", **delta)
    meta = {"pre": pre, "post": post, "gnorm": float(gnorm), "sec_16_fwd_bwd": sec_fb, "sec_step": step_s, "load_s": load_s,
            "peak_driver_gb": peak, "gpu_s": g.elapsed, "n_tensors": len(grads), "grad_checkpointing": grad_ckpt}
    meta["dtype"] = dtype
    (CACHE / f"torch{sfx}.json").write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("pre", "post")}, indent=1))


def run_torch_updates(samples, n_updates: int, grad_ckpt: bool):
    import torch

    torch.manual_seed(0)
    with Gpu("torch-updates") as g:
        model, proc, tr = load_torch(0.05, grad_ckpt)
        params = [p for _, p in tr]
        opt = torch.optim.Adam(params, lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        rows, peak = [], 0.0
        for u in range(1, n_updates + 1):
            if time.perf_counter() > g.deadline:
                break
            torch.mps.synchronize()
            t = time.perf_counter()
            losses = []
            for s in samples[(u - 1) * tt.ACCUM: u * tt.ACCUM]:
                loss = torch_loss(model, torch_item(proc, s))
                loss.backward()
                losses.append(float(loss.detach()))
            for p in params:
                p.grad /= len(losses)
            gnorm = torch.nn.utils.clip_grad_norm_(params, tt.CLIP)
            for pg in opt.param_groups:
                pg["lr"] = tt.lr_at(u)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            peak = max(peak, mem_gb())
            rows.append({"update": u, "loss": float(np.mean(losses)), "lr": tt.lr_at(u), "gnorm": float(gnorm), "sec": time.perf_counter() - t})
            print(rows[-1], flush=True)
    (CACHE / "torch_updates.json").write_text(json.dumps({"rows": rows, "peak_driver_gb": peak, "gpu_s": g.elapsed, "grad_checkpointing": grad_ckpt}))


# ---------------------------------------------------------------- report (CPU)
def windows(xs, w=10):
    return [round(float(np.mean(xs[i:i + w])), 5) for i in range(0, len(xs), w)]


def report():
    m, t = json.loads((CACHE / "mlx.json").read_text()), json.loads((CACHE / "torch.json").read_text())
    a = tt.loss_stats(m["pre"], t["pre"])
    gm, gt = dict(np.load(CACHE / "mlx_grads.npz")), dict(np.load(CACHE / "torch_grads.npz"))
    b = tt.grad_stats(gm, gt)
    dm, dt = dict(np.load(CACHE / "mlx_delta.npz")), dict(np.load(CACHE / "torch_delta.npz"))
    dstat = tt.grad_stats(dm, dt)
    c = tt.loss_stats(m["post"], t["post"])
    ch_m = float(np.mean(m["post"]) - np.mean(m["pre"][N_ACC:]))
    ch_t = float(np.mean(t["post"]) - np.mean(t["pre"][N_ACC:]))
    crit = {"A1_absdiff_mean_le_0.02": a["absdiff_mean"] <= 0.02, "A2_absdiff_max_le_0.10": a["absdiff_max"] <= 0.10,
            "A3_abs_signed_mean_le_0.01": abs(a["signed_diff_mean"]) <= 0.01,
            "B1_cos_median_ge_0.99": b["cos_median"] >= 0.99, "B2_cos_min_ge_0.95": b["cos_min"] >= 0.95,
            "B3_global_rel_norm_diff_le_0.05": b["global_rel_norm_diff"] <= 0.05,
            "C1_post_absdiff_mean_le_0.02": c["absdiff_mean"] <= 0.02,
            "C2_change_same_sign_and_close": bool(np.sign(ch_m) == np.sign(ch_t) and abs(ch_t - ch_m) <= 0.25 * abs(ch_m) + 0.005)}
    log = [json.loads(x) for x in B0_LOG.read_text().splitlines()]
    rep = {"plan": "docs/plans/torch_train_parity.md", "reference": "Step B control B0 (scripts/stepb_train.py), frozen adapter 4c030967..., "
           "B0 stream samples in stream order", "b0_stream_sha256": B0_SHA, "adapter_sha256": FROZEN_SHA,
           "engines": {"mlx": "mlx 0.32.3 / mlx-vlm 0.7.4, bf16 base, fp32 LoRA, grad checkpointing", "torch": "torch / transformers / peft on MPS, bf16 base, fp32 LoRA"},
           "dropout_ab_c": 0.0,
           "a_forward_loss_samples_0_31": a,
           "a_stricter_example_tolerance_mean_le_0.01_max_le_0.05": bool(a["absdiff_mean"] <= 0.01 and a["absdiff_max"] <= 0.05),
           "b_mean_grad_samples_0_15": b, "b_gnorm_preclip": {"mlx": m["gnorm"], "torch": t["gnorm"]},
           "c_one_adam_step_lr": tt.PEAK_LR, "c_post_step_loss_samples_16_31": c,
           "c_mean_loss_change_post_minus_pre": {"mlx": ch_m, "torch": ch_t},
           "c_param_update_similarity": {k: dstat[k] for k in ("cos_median", "cos_min", "cos_p05", "global_cos", "rel_norm_diff_median",
                                                               "global_rel_norm_diff")},
           "criteria": crit, "pass": all(crit.values()),
           "timing": {"mlx_sec_16_fwd_bwd": m["sec_16_fwd_bwd"], "torch_sec_16_fwd_bwd": t["sec_16_fwd_bwd"],
                      "mlx_sec_step": m["sec_step"], "torch_sec_step": t["sec_step"],
                      "b0_logged_sec_per_update_median": float(np.median([r["sec"] for r in log])),
                      "torch_grad_checkpointing": t["grad_checkpointing"]},
           "peak_memory_gb": {"mlx_peak": m["peak_gb"], "torch_mps_driver_peak": t["peak_driver_gb"],
                              "b0_logged_peak_max": max(r["peak_gb"] for r in log)},
           "gpu_minutes": {"mlx": m["gpu_s"] / 60, "torch": t["gpu_s"] / 60}}
    norms = {k: float(np.linalg.norm(v)) for k, v in gm.items()}
    med = float(np.median(list(norms.values())))
    rep["b_lowest_cos_tensor_norm_over_median_norm"] = {x["key"]: round(norms[x["key"]] / med, 4) for x in b["lowest_cos_tensors"]}
    f32 = CACHE / "torch_float32.json"
    if f32.exists():  # diagnostic only (not gated): is the low-cosine set bf16 noise, i.e. also present torch bf16 vs torch fp32?
        t32 = json.loads(f32.read_text())
        g32 = dict(np.load(CACHE / "torch_grads_float32.npz"))
        keys = [x["key"] for x in b["lowest_cos_tensors"]]

        def cos_of(r, o):
            return {k: round(float(r[k].ravel() @ o[k].ravel() / (np.linalg.norm(r[k]) * np.linalg.norm(o[k]))), 5) for k in keys}

        sb, sm = tt.grad_stats(g32, gt), tt.grad_stats(g32, gm)
        pick = ("cos_median", "cos_min", "cos_p05", "n_cos_below_0.95", "global_cos", "global_rel_norm_diff")
        rep["diagnostic_torch_float32"] = {
            "torch_bf16_vs_torch_fp32_grad": {k: sb[k] for k in pick}, "mlx_bf16_vs_torch_fp32_grad": {k: sm[k] for k in pick},
            "lowest5_cos_torch_bf16_vs_fp32": cos_of(g32, gt), "lowest5_cos_mlx_vs_torch_fp32": cos_of(g32, gm),
            "loss_0_31_torch_fp32_vs_mlx": tt.loss_stats(m["pre"], t32["pre"]), "loss_0_31_torch_fp32_vs_torch_bf16": tt.loss_stats(t["pre"], t32["pre"]),
            "gnorm_fp32": t32["gnorm"], "gpu_minutes": t32["gpu_s"] / 60}
        rep["gpu_minutes"]["torch_float32_diag"] = t32["gpu_s"] / 60
    m32f = CACHE / "mlx_float32.json"
    if m32f.exists() and f32.exists():  # diagnostic only: both engines in fp32
        m32 = json.loads(m32f.read_text())
        gm32 = dict(np.load(CACHE / "mlx_grads_float32.npz"))
        keys = [x["key"] for x in b["lowest_cos_tensors"]]
        s1, s2 = tt.grad_stats(gm32, g32), tt.grad_stats(gm32, gm)
        pick = ("cos_median", "cos_min", "cos_p05", "n_cos_below_0.95", "global_cos", "global_rel_norm_diff")
        rep["diagnostic_mlx_float32"] = {
            "mlx_fp32_vs_torch_fp32_grad": {k: s1[k] for k in pick}, "mlx_bf16_vs_mlx_fp32_grad": {k: s2[k] for k in pick},
            "lowest5_cos_mlx_fp32_vs_torch_fp32": {k: round(float(gm32[k].ravel() @ g32[k].ravel() / (np.linalg.norm(gm32[k]) * np.linalg.norm(g32[k]))), 5) for k in keys},
            "lowest5_cos_mlx_bf16_vs_mlx_fp32": {k: round(float(gm32[k].ravel() @ gm[k].ravel() / (np.linalg.norm(gm32[k]) * np.linalg.norm(gm[k]))), 5) for k in keys},
            "loss_0_31_mlx_fp32_vs_torch_fp32": tt.loss_stats(m32["pre"], t32["pre"]), "gnorm_fp32": m32["gnorm"], "gpu_minutes": m32["gpu_s"] / 60}
        rep["gpu_minutes"]["mlx_float32_diag"] = m32["gpu_s"] / 60
    inp = CACHE / "inputs.json"
    if inp.exists():
        rep["inputs_torch_vs_mlx_vlm"] = json.loads(inp.read_text())
    upd = CACHE / "torch_updates.json"
    if upd.exists():
        u = json.loads(upd.read_text())
        rows = u["rows"]
        n = len(rows)
        bl = [r["loss"] for r in log[:n]]
        rep["d_updates_dropout_0.05_descriptive"] = {
            "n_updates": n, "torch_loss_window_means_10": windows([r["loss"] for r in rows]), "b0_loss_window_means_10": windows(bl),
            "torch_mean": float(np.mean([r["loss"] for r in rows])), "b0_mean": float(np.mean(bl)),
            "torch_gnorm_window_means_10": windows([r["gnorm"] for r in rows]), "b0_gnorm_window_means_10": windows([r["gnorm"] for r in log[:n]]),
            "lr_identical_to_b0_log": bool(all(abs(r["lr"] - x["lr"]) <= 1e-12 + 1e-6 * x["lr"] for r, x in zip(rows, log))),
            "torch_sec_per_update_median": float(np.median([r["sec"] for r in rows])),
            "b0_sec_per_update_median_same_updates": float(np.median([r["sec"] for r in log[:n]])),
            "torch_peak_driver_gb": u["peak_driver_gb"], "gpu_minutes": u["gpu_s"] / 60, "grad_checkpointing": u["grad_checkpointing"]}
        rep["gpu_minutes"]["torch_updates"] = u["gpu_s"] / 60
    rep["gpu_minutes"]["total"] = sum(v for v in rep["gpu_minutes"].values())
    try:
        import peft
        import torch
        import transformers

        rep["engines"]["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    except ImportError:
        pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps({"criteria": crit, "pass": rep["pass"]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dump-mlx-inputs", "compare-inputs", "mlx", "torch", "torch-updates", "report"])
    ap.add_argument("--n-updates", type=int, default=30)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="--mode mlx / torch; float32 = diagnostic only")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.mode == "report":
        return report()
    if a.mode == "torch-updates":
        return run_torch_updates(load_samples(a.n_updates * tt.ACCUM), a.n_updates, not a.no_grad_ckpt)
    samples = load_samples(N_INPUTS)
    if a.mode == "dump-mlx-inputs":
        print(dump_mlx_inputs(samples))
    elif a.mode == "compare-inputs":
        res = compare_inputs(samples)
        (CACHE / "inputs.json").write_text(json.dumps(res))
        print(res)
    elif a.mode == "mlx":
        run_mlx(samples, a.dtype)
    else:
        run_torch(samples, not a.no_grad_ckpt, a.dtype)


if __name__ == "__main__":
    main()
