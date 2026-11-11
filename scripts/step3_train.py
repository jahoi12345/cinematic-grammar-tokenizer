"""Step 3 training of one candidate (docs/plans/step3_gap.md, BINDING): CONTINUE the Step C 4B LoRA (ckpt_100, sha256 aba56883...,
hash-checked, exit 7) on the seeded 9,600-sample ShotQA TRAIN replay subset (cgt.step3.data.replay_stream; the same samples in the
same order for every candidate), 600 updates. Built from scripts/stepc_train.py (not changed): same model, LoRA layout, loss, leak
probe, micro-batch 1 x accumulation 16, clipping, max length, memory guard, resume state every 25 updates, MLX buffer env logged.
Continued-LoRA optimiser as Step B: fresh Adam, peak LR 2e-5, 3% linear warm-up, cosine to 2e-6; Step B stopping rule (non-finite
loss, or mean of the last 50 updates > 1.0 after update 100 -> exit 3). The candidates differ ONLY in the fed image:
G0 the Phase 3B 1x copy; G1 a label-preserving augmentation of it (cgt.step3.augment, per-attribute rules); G2 a 2x-budget copy.
One checkpoint: ckpt_100 at update 600 (registered in the run manifest). Never reads ShotBench, CameraBench, gold, dev or val.
Exit codes as Step C: 0 finished; 3 stopping rule; 4 leak; 5 data; 6 image; 7 adapter hash; 8 length; 9 testing stop; 10 dirty.
  HF_HUB_OFFLINE=1 nice -n 19 <venv>/bin/python scripts/step3_train.py --cand G1 [--n-samples 32 --tag step3_pc_G1]"""
import argparse
import collections
import json
import logging
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import iterate_batches
from mlx_vlm.trainer.utils import (
    find_all_linear_names,
    get_peft_model,
    grad_checkpoint,
    save_adapter,
)
from mlx_vlm.utils import load

from cgt.cinescale2.semantic import wait_for_memory
from cgt.p3b_vlm import guards
from cgt.p3b_vlm.adapter_load import load_adapter_checked
from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import (
    append_jsonl,
    now_iso,
    software_provenance,
    write_status,
)
from cgt.step3 import config as C
from cgt.step3.data import replay_stream, train_image


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class LazyItems:
    """Training items built on access, so G1's augmented copy of sample `micro` is written only when it is fed."""
    def __init__(self, samples, cand, offset=0):
        self.samples, self.cand, self.offset = samples, cand, offset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {"images": [train_image(s, self.offset + i, self.cand)], "messages": s["messages"]}


class SkipWatcher(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.message = None

    def emit(self, record):
        if "Skipping example" in record.getMessage():
            self.message = record.getMessage()


def milestones_for(total: int) -> dict[int, int]:
    """Update -> percent: one checkpoint, 100% = the last update (plan section 4)."""
    return {total: 100}


def run(a, log) -> None:
    run_dir = C.DATA / "runs" / a.tag
    logd = Path(a.results_dir)
    git = guards.git_state(C.CODE_PATHS)
    if git["dirty_files"] and not a.allow_dirty:
        raise guards.DirtyTree(f"uncommitted changes under {C.CODE_PATHS}: {git['dirty_files'][:6]}")
    append_jsonl(logd / "provenance.jsonl", {"tag": a.tag, "code_commit": git["commit"], "dirty_files": git["dirty_files"],
                                             "model": f"{C.REPO}@{C.REVISION}", "software": software_provenance(), "args": vars(a),
                                             "cand": a.cand, "variant": C.CANDIDATES[a.cand], "init_adapter_sha256": C.INIT_SHA,
                                             "train_sha256": C.TRAIN_SHA, "started": now_iso()})
    mlx_env = {k: os.environ.get(k) for k in C.MLX_ENV}
    append_jsonl(logd / "provenance.jsonl", {"tag": a.tag, "mlx_env_seen": mlx_env})
    log(f"code commit {git['commit']} dirty={bool(git['dirty_files'])}; MLX buffer env seen: {mlx_env}")

    if guards.sha256_file(C.INIT_ADAPTER) != C.INIT_SHA:
        raise guards.AdapterHashError(f"initial adapter {C.INIT_ADAPTER} does not match {C.INIT_SHA}")
    spec = C.CANDIDATES[a.cand]
    used = replay_stream()
    if a.n_samples:
        used = used[: a.n_samples]
    items_all = LazyItems(used, a.cand)
    if not spec["augment"]:      # 1x copies must exist (exit 6); 2x copies are made here from the ShotQA originals
        guards.require_images_readable(sorted({items_all[i]["images"][0] for i in range(len(used))}))
    n_micro = len(used)
    total = math.ceil(n_micro / C.ACCUM)
    warmup = round(C.WARMUP_FRAC * total)
    milestones = milestones_for(total)
    log(f"tag={a.tag} samples={n_micro} updates={total} warmup={warmup} milestones={milestones} model={C.REPO}@{C.REVISION}")

    mx.random.seed(C.MX_SEED)
    model, processor = load(pinned_model_path(C.REPO, C.REVISION))
    cfg = model.config.__dict__
    tl = guards.require_token_lengths(processor, used, C.MAX_SEQ, spec["image_tokens"])
    log(f"token lengths: max {tl['max_tokens']} of {C.MAX_SEQ}, over limit {tl['n_over_limit']}")
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=C.RANK, alpha=C.ALPHA, dropout=C.DROPOUT,
                           verbose=False)
    n_init = load_adapter_checked(model, C.INIT_ADAPTER)  # fails closed unless every adapter tensor matches the model
    log(f"initial adapter loaded: {C.INIT_ADAPTER.name} sha256 {C.INIT_SHA}, {n_init} tensors matched; variant {a.cand} {spec}")
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"LoRA trainable parameters: {n_train}")

    schedule = optim.join_schedules([optim.linear_schedule(0.0, C.PEAK_LR, max(warmup, 1)),
                                     optim.cosine_decay(C.PEAK_LR, max(total - warmup, 1), C.END_LR)], [warmup])
    optimizer = optim.Adam(learning_rate=schedule, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    manifest_f = logd / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {"checkpoints": {}, "model": f"{C.REPO}@{C.REVISION}"}
    start_update, start_micro, loss_hist = 0, 0, []
    res_json = run_dir / "resume.json"
    if res_json.exists():
        guards.verify_manifest_files(manifest, run_dir)
        st = json.loads(res_json.read_text())
        model.load_weights(list(mx.load(str(run_dir / "resume_adapter.safetensors")).items()), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(run_dir / "resume_opt.safetensors")).items()))
        start_update, start_micro, loss_hist = st["update"], st["next_micro"], st["loss_hist"]
        mx.random.seed(C.RESUME_SEED_BASE + start_update)   # Phase 3B amendment 1: dropout RNG reseeded deterministically
        log(f"RESUMED from update {start_update} (next sample {start_micro})")
        append_jsonl(logd / "restarts.jsonl", {"resumed_at_update": start_update})
    hist = collections.deque(loss_hist[-C.STOP_WINDOW:], maxlen=C.STOP_WINDOW)

    probe_item = items_all[0]
    leak = leakage_probe(model, VisionDataset(ListView([probe_item]), cfg, processor, None, True).process(probe_item))
    if a.simulate_probe_failure:
        leak = {**leak, "without_mask": 0.125}
    append_jsonl(logd / "leak_probes.jsonl", {"tag": a.tag, "resumed_at_update": start_update, **leak})
    log(f"leak probe: {leak}")
    guards.require_zero_leak(leak)

    for module in model.children().values():
        if hasattr(module, "layers"):
            grad_checkpoint(module.layers[0])
    loss_and_grad = nn.value_and_grad(model, partial(causal_loss_fn, train_on_completions=True))
    todo = LazyItems(used[start_micro:], a.cand, offset=start_micro)
    batches = iterate_batches(VisionDataset(todo, cfg, processor, None, True), 1, C.MAX_SEQ, train=False)
    watcher = SkipWatcher()
    logging.getLogger().addHandler(watcher)
    model.train()
    update, micro = start_update, start_micro
    t_up = time.perf_counter()
    sec_hist = collections.deque(maxlen=20)
    last_ck = "-"
    while micro < n_micro:
        if update % C.MEM_EVERY == 0:
            pauses = wait_for_memory(C.MEM_FLOOR, log=log, poll_s=60)
            if pauses:
                log(f"memory guard: paused {pauses} x 60 s before update {update + 1}")
                append_jsonl(logd / "memory_pauses.jsonl", {"before_update": update + 1, "pauses_60s": pauses})
                t_up = time.perf_counter()
        acc, losses = None, []
        for _ in range(min(C.ACCUM, n_micro - micro)):
            batch = next(batches)
            if watcher.message:
                raise guards.TokenLengthViolation(watcher.message)
            loss, grad = loss_and_grad(model, batch)
            acc = grad if acc is None else tree_map(lambda x, y: x + y, acc, grad)
            mx.eval(loss, acc)
            mx.clear_cache()
            losses.append(float(loss))
            micro += 1
        n_acc = len(losses)
        acc = tree_map(lambda g, n=n_acc: g / n, acc)
        acc, gnorm = optim.clip_grad_norm(acc, C.CLIP)
        optimizer.update(model, acc)
        mx.eval(model.parameters(), optimizer.state)
        update += 1
        ul = float(np.mean(losses))
        loss_hist.append(ul)
        hist.append(ul)
        sec = time.perf_counter() - t_up
        t_up = time.perf_counter()
        sec_hist.append(sec)
        rec = {"update": update, "micro": micro, "loss": round(ul, 5), "lr": float(schedule(update - 1)), "gnorm": round(float(gnorm), 4),
               "sec": round(sec, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        append_jsonl(logd / "updates.jsonl", rec)
        if update % 10 == 0 or update <= 3 or update in milestones:
            log(f"update {update}/{total} loss {np.mean(list(hist)[-10:]):.4f} lr {rec['lr']:.2e} gnorm {float(gnorm):.2f} "
                f"{sec:.1f}s/update peak {rec['peak_gb']} GB")
            write_status(logd, a.tag, state="training", update=update, total=total, micro=micro,
                         loss=round(float(np.mean(list(hist)[-10:])), 4), sec_per_update=round(float(np.mean(sec_hist)), 2),
                         peak_gb=rec["peak_gb"], last_checkpoint=last_ck)
        if not math.isfinite(ul) or (update > C.STOP_AFTER and np.mean(list(hist)) > C.STOP_LOSS):
            log(f"ABORT by the stopping rule at update {update}: loss {ul}, last-{C.STOP_WINDOW} mean {np.mean(list(hist)):.3f}")
            write_status(logd, a.tag, state="ABORTED by the pre-registered stopping rule", update=update, total=total)
            sys.exit(3)
        if update in milestones or update % C.RESUME_EVERY == 0 or micro >= n_micro:
            run_dir.mkdir(parents=True, exist_ok=True)
            if update in milestones:
                pct = milestones[update]
                ck = run_dir / f"ckpt_{pct:03d}.safetensors"
                save_adapter(model, ck)
                digest = guards.sha256_file(ck)
                manifest["checkpoints"][str(pct)] = {"update": update, "micro": micro, "file": ck.name, "sha256": digest,
                                                     "saved": now_iso(), "code_commit": git["commit"]}
                manifest_f.write_text(json.dumps(manifest, indent=1))
                last_ck = f"{pct}% (update {update}) sha256 {digest[:16]}..."
                log(f"checkpoint {pct}% saved: sha256 {digest}")
            mx.save_safetensors(str(run_dir / ".tmp_resume_adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            (run_dir / ".tmp_resume_adapter.safetensors").rename(run_dir / "resume_adapter.safetensors")
            mx.save_safetensors(str(run_dir / ".tmp_resume_opt.safetensors"), dict(tree_flatten(optimizer.state)))
            (run_dir / ".tmp_resume_opt.safetensors").rename(run_dir / "resume_opt.safetensors")
            (run_dir / ".tmp_resume.json").write_text(json.dumps({"update": update, "next_micro": micro, "loss_hist": loss_hist}))
            (run_dir / ".tmp_resume.json").rename(res_json)
        if a.stop_after_update and update >= a.stop_after_update:
            log(f"testing aid: exiting after update {update}")
            sys.exit(9)
    log(f"DONE: {update} updates, {micro} samples; last-{C.STOP_WINDOW} mean loss {np.mean(list(hist)):.4f}")
    (logd / "loss_curve.json").write_text(json.dumps(loss_hist))
    write_status(logd, a.tag, state="training finished", update=update, total=total, micro=micro,
                 sec_per_update=round(float(np.mean(sec_hist)), 2) if sec_hist else None, last_checkpoint=last_ck)
    (run_dir / "DONE").write_text("done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=list(C.CANDIDATES))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="pipeline check only: the first N samples of the seeded stream")
    ap.add_argument("--allow-dirty", action="store_true", help="tests only: permit uncommitted code")
    ap.add_argument("--simulate-probe-failure", action="store_true")
    ap.add_argument("--stop-after-update", type=int, default=None)
    ap.add_argument("--results-dir", default=None, help="aggregates/logs directory (default results/stepc/runs/<tag>)")
    a = ap.parse_args()
    a.tag = a.tag or f"step3_{a.cand}"
    a.results_dir = a.results_dir or str(C.RESULTS / "runs" / a.tag)
    (C.DATA / "runs" / a.tag).mkdir(parents=True, exist_ok=True)
    logd = Path(a.results_dir)
    logd.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with open(logd / "train.log", "a") as f:
            f.write(line + "\n")

    try:
        run(a, log)
    except guards.FailClosed as e:
        log(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}")
        write_status(logd, a.tag, state=f"STOPPED: {type(e).__name__} (exit {e.exit_code})")
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
