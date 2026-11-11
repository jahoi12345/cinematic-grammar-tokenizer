"""Qwen3-VL-4B local feasibility: cost measurement only (docs/plans/qwen3vl4b_feasibility.md). No accuracy is computed or written.
Orchestrator (default): waits for /tmp/cgt_gpu.lock, holds it, runs one child per model (2B, then 4B) with the remaining part of a
45-minute GPU cap as a hard timeout, and writes results/qwen3vl4b_feasibility/report.json (+ .md). Child (--child 2b|4b): fresh LoRA
with the Step B recipe; letter-logit scoring timing on the first 50 ShotQA val questions; 2 warm-up + 10 measured training updates on
the first 192 ShotQA TRAIN samples. Never reads gold, dev, ShotBench or CameraBench. No downloads (HF_HUB_OFFLINE=1).
  HF_HUB_OFFLINE=1 caffeinate -i nice -n 19 <venv>/bin/python scripts/qwen3vl4b_feasibility.py"""
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.cinescale2.semantic import memory_free_percent
from cgt.p3b_vlm.feasibility import n_updates, project_hours, timing_summary
from cgt.pipeline.gpulock import GpuLock, LockHeld

MODELS = {"2b": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
          "4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17")}
P3B = ROOT / "data/cache/p3b"
IMG_CACHE = P3B / "images_147456_32"
SCRATCH = ROOT / "data/cache/qwen3vl4b_feasibility"
OUT = ROOT / "results/qwen3vl4b_feasibility"
RANK, ALPHA, DROPOUT, PEAK_LR, END_LR, WARMUP_FRAC, ACCUM, CLIP, MAX_SEQ = 16, 32.0, 0.05, 2e-5, 2e-6, 0.03, 16, 1.0, 1024
FULL_TRAIN, FULL_VAL = 51_111, 5_822
WARM_UPDATES, MEAS_UPDATES, VAL_N, VAL_WARM = 2, 10, 50, 2
GPU_CAP_S, MEM_FLOOR = 45 * 60, 25


class MemoryLow(RuntimeError):
    pass


def free_pct(samples: list, where: str) -> int | None:
    f = memory_free_percent()
    samples.append({"where": where, "free_pct": f})
    if f is not None and f < MEM_FLOOR:
        raise MemoryLow(f"free memory {f}% < {MEM_FLOOR}% at {where}")
    return f


def child(key: str, out: Path) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import iterate_batches
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model, grad_checkpoint
    from mlx_vlm.utils import load

    from cgt.p3b_vlm import guards
    from cgt.p3b_vlm.images import shrunk_path
    from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
    from cgt.p3b_vlm.model_path import pinned_model_path
    from cgt.p3b_vlm.score import answer_token_ids, letter_logits

    class ListView:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    mem: list = []
    res = {"model": f"{MODELS[key][0]}@{MODELS[key][1]}", "status": "started", "memory_free_samples": mem}

    def save():
        out.write_text(json.dumps(res, indent=1))

    try:
        free_pct(mem, "before load")
        n_train = (WARM_UPDATES + MEAS_UPDATES) * ACCUM
        train = [json.loads(x) for x in (P3B / "train.jsonl").read_text().splitlines()[:n_train]]
        val = [json.loads(x) for x in (P3B / "val.jsonl").read_text().splitlines()[:VAL_N]]
        train_items = [{"id": s["id"], "images": [shrunk_path(s["images"][0], IMG_CACHE)], "messages": s["messages"]} for s in train]
        val_items = [{**s, "images": [shrunk_path(s["images"][0], IMG_CACHE)]} for s in val]

        mx.random.seed(0)
        t0 = time.perf_counter()
        model, processor = load(pinned_model_path(*MODELS[key]))
        res["load_s"] = round(time.perf_counter() - t0, 1)
        cfg = model.config.__dict__
        tl = guards.require_token_lengths(processor, train_items + val_items, MAX_SEQ, 144)
        res["token_lengths"] = {"n": tl["n"], "max_tokens": tl["max_tokens"], "limit": MAX_SEQ}
        ids4 = answer_token_ids(processor)
        model = get_peft_model(model, find_all_linear_names(model.language_model), rank=RANK, alpha=ALPHA, dropout=DROPOUT, verbose=False)
        res["lora"] = {"rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "dropout": DROPOUT,
                       "trainable_params": int(sum(v.size for _, v in tree_flatten(model.trainable_parameters()))),
                       "total_params": int(sum(v.size for _, v in tree_flatten(model.parameters())))}
        ds = VisionDataset(ListView(train_items), cfg, processor, None, True)
        leak = leakage_probe(model, ds.process(train_items[0]))
        res["leak_probe"] = leak
        guards.require_zero_leak(leak)

        # 1) validation scoring throughput (timings only; predictions/logits are discarded)
        free_pct(mem, "before val")
        model.eval()
        eval_ds = VisionDataset(ListView([]), cfg, processor, None, True)
        mx.reset_peak_memory()
        vsec = []
        for s in val_items:
            t = time.perf_counter()
            letter_logits(model, eval_ds.process({"images": s["images"], "messages": s["messages"]}), ids4)
            vsec.append(time.perf_counter() - t)
        res["val"] = {"questions": len(vsec), "warmup": VAL_WARM, "sec_per_question": timing_summary(vsec, VAL_WARM),
                      "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "source": "first 50 of data/cache/p3b/val.jsonl"}
        free_pct(mem, "after val")
        save()

        # 2) training: the Step B loop, fresh LoRA, fresh Adam
        total = n_updates(FULL_TRAIN, ACCUM)
        warmup = round(WARMUP_FRAC * total)
        schedule = optim.join_schedules([optim.linear_schedule(0.0, PEAK_LR, max(warmup, 1)),
                                         optim.cosine_decay(PEAK_LR, max(total - warmup, 1), END_LR)], [warmup])
        optimizer = optim.Adam(learning_rate=schedule, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
        optimizer.init(model.trainable_parameters())
        for module in model.children().values():
            if hasattr(module, "layers"):
                grad_checkpoint(module.layers[0])
        loss_and_grad = nn.value_and_grad(model, partial(causal_loss_fn, train_on_completions=True))
        batches = iterate_batches(ds, 1, MAX_SEQ, train=False)
        model.train()
        mx.reset_peak_memory()
        usec, ups = [], []
        for update in range(1, WARM_UPDATES + MEAS_UPDATES + 1):
            t = time.perf_counter()
            acc, losses = None, []
            for _ in range(ACCUM):
                loss, grad = loss_and_grad(model, next(batches))
                acc = grad if acc is None else tree_map(lambda x, y: x + y, acc, grad)
                mx.eval(loss, acc)
                mx.clear_cache()
                losses.append(float(loss))
            acc = tree_map(lambda g: g / ACCUM, acc)
            acc, gnorm = optim.clip_grad_norm(acc, CLIP)
            optimizer.update(model, acc)
            mx.eval(model.parameters(), optimizer.state)
            sec = time.perf_counter() - t
            usec.append(sec)
            finite = all(math.isfinite(x) for x in losses) and math.isfinite(float(gnorm))
            ups.append({"update": update, "sec": round(sec, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "finite": finite})
            print(f"[{key}] update {update} {sec:.2f}s peak {ups[-1]['peak_gb']} GB", flush=True)
            free_pct(mem, f"after update {update}")
        res["train"] = {"samples": n_train, "warmup_updates": WARM_UPDATES, "measured_updates": MEAS_UPDATES,
                        "sec_per_update": timing_summary(usec, WARM_UPDATES), "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
                        "all_losses_finite": all(u["finite"] for u in ups), "updates": ups,
                        "source": f"first {n_train} samples of data/cache/p3b/train.jsonl"}
        res["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
        res["status"] = "ok"
    except MemoryLow as e:
        res["status"] = f"stopped: {e}"
    save()


def wait_lock(job: str) -> GpuLock:
    while True:
        try:
            return GpuLock(job).acquire()
        except LockHeld as e:
            print(f"lock held, waiting 60 s: {e}", flush=True)
            time.sleep(60)


def write_md(rep: dict) -> str:
    lines = ["# Qwen3-VL-4B local feasibility (cost only, no accuracy)", "",
             (f"Plan: `docs/plans/qwen3vl4b_feasibility.md`. Code commit {rep['code_commit']}. GPU minutes used: {rep['gpu_minutes']:.1f} "
              f"(cap {GPU_CAP_S // 60})."), "",
             ("| model | s/update (median of 10) | train peak GB | val s/question (median of 48) | val peak GB | load s | min free % | "
              "1 epoch h (3,195 updates) | full val h (5,822 q) |"), "|---|---|---|---|---|---|---|---|---|"]
    for k, r in rep["models"].items():
        if r.get("status") != "ok":
            lines.append(f"| {k} | {r.get('status')} | | | | | | | |")
            continue
        p = rep["projections"][k]
        lines.append(f"| {k} | {r['train']['sec_per_update']['median']:.2f} | {r['train']['peak_gb']} | "
                     f"{r['val']['sec_per_question']['median']:.3f} | {r['val']['peak_gb']} | {r['load_s']} | {r['memory_free_min_pct']} | "
                     f"{p['train_1epoch_hours']:.2f} | {p['full_val_hours']:.2f} |")
    if "ratio_4b_over_2b" in rep:
        rr = rep["ratio_4b_over_2b"]
        lines += ["", (f"Ratio 4B/2B: training {rr['sec_per_update']:.2f}x, validation {rr['sec_per_question']:.2f}x, "
                       f"training peak memory {rr['train_peak_gb']:.2f}x.")]
    lines += ["", "Assumptions: " + " ".join(rep["assumptions"])]
    return "\n".join(lines) + "\n"


def orchestrate(models: list[str]) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = wait_lock("qwen3vl4b_feasibility")
    t_lock = time.perf_counter()
    rep = {"plan": "docs/plans/qwen3vl4b_feasibility.md", "code_commit": commit, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "accuracy_reported": False, "models": {}}
    try:
        for k in models:
            remaining = GPU_CAP_S - (time.perf_counter() - t_lock)
            if remaining < 60:
                rep["models"][k] = {"status": "not run: GPU cap reached"}
                continue
            f = memory_free_percent()
            if f is not None and f < MEM_FLOOR:
                rep["models"][k] = {"status": f"not run: free memory {f}% < {MEM_FLOOR}%"}
                continue
            part = SCRATCH / f"{k}.json"
            part.unlink(missing_ok=True)
            try:
                subprocess.run([sys.executable, __file__, "--child", k, "--out", str(part)], cwd=ROOT, timeout=remaining, check=False)
                r = json.loads(part.read_text()) if part.exists() else {"status": "child wrote nothing"}
            except subprocess.TimeoutExpired:
                r = json.loads(part.read_text()) if part.exists() else {}
                r["status"] = "killed at the GPU cap"
            fr = [m["free_pct"] for m in r.get("memory_free_samples", []) if m["free_pct"] is not None]
            r["memory_free_min_pct"], r["memory_free_max_pct"] = (min(fr), max(fr)) if fr else (None, None)
            rep["models"][k] = r
    finally:
        rep["gpu_minutes"] = round((time.perf_counter() - t_lock) / 60, 2)
        lock.release()
    nu = n_updates(FULL_TRAIN, ACCUM)
    rep["projections"] = {}
    for k, r in rep["models"].items():
        if r.get("status") == "ok":
            rep["projections"][k] = {"train_updates_1epoch": nu,
                                     "train_1epoch_hours": project_hours(r["train"]["sec_per_update"]["median"], nu),
                                     "full_val_questions": FULL_VAL,
                                     "full_val_hours": project_hours(r["val"]["sec_per_question"]["median"], FULL_VAL)}
    ok = {k for k, r in rep["models"].items() if r.get("status") == "ok"}
    if {"2b", "4b"} <= ok:
        a, b = rep["models"]["4b"], rep["models"]["2b"]
        rep["ratio_4b_over_2b"] = {"sec_per_update": a["train"]["sec_per_update"]["median"] / b["train"]["sec_per_update"]["median"],
                                   "sec_per_question": a["val"]["sec_per_question"]["median"] / b["val"]["sec_per_question"]["median"],
                                   "train_peak_gb": a["train"]["peak_gb"] / b["train"]["peak_gb"]}
    rep["assumptions"] = [
        "Linear extrapolation from 10 measured updates (after 2 warm-up) on the first 192 TRAIN samples and 48 timed val questions (after 2 warm-up).",
        "Excludes model load, checkpoint saves, monitoring and memory-guard pauses; no thermal throttling over hours is measured.",
        "One heavy job on the machine (GPU lock held); both models measured in the same session, 2B first.",
        "1 epoch = ceil(51,111/16) = 3,195 updates; full val = 5,822 questions (dedup v2).",
        "Fresh LoRA (B = 0) for both models; no accuracy computed."]
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text(write_md(rep))
    print(write_md(rep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--models", default="2b,4b")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.out)
    else:
        orchestrate(a.models.split(","))


if __name__ == "__main__":
    main()
