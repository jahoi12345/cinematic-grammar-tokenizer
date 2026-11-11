"""Step B training of one candidate (docs/plans/stepb_cinescale2.md, BINDING): continue the frozen Phase 3B LoRA (sha256 4c030967...)
on data/cache/stepb/<cand>.jsonl. Built on scripts/p3b_train.py (same model, LoRA layout, loss, accumulation, clipping, image budget).
Differences fixed by the plan: initial adapter = the frozen checkpoint (hash-checked), fresh Adam, peak LR 2e-5 (3% linear warm-up,
cosine to 2e-6), checkpoints at 50% and 100%, monitoring at 50% (500 ShotQA + 200 CineScale2 val questions, never used for
selection), stopping rule (non-finite loss, or mean loss of the last 50 updates > 1.0 after update 100), memory guard every 50
updates (free memory >= 25%; PAUSES below, never crashes), resume state every 100 updates.
Exit codes: 0 finished; 3 stopping rule; 4 leak probe; 5 data root; 6 unreadable image; 7 adapter hash; 8 token length; 10 dirty
tree / plan changed; 9 testing stop; other = crash (the runner may resume once from the resume state).
Never reads ShotBench, CameraBench, gold or dev.
  nice -n 19 <venv>/bin/python scripts/stepb_train.py --cand B1 [--max-samples 64 --tag pipecheck]"""
import argparse
import collections
import json
import logging
import math
import sys
import time
from functools import partial
from pathlib import Path

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
from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import append_jsonl, now_iso, software_provenance, write_status
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize
from cgt.stepb.evaluate import as_model_item, evaluate_cs2, summarize_cs2

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
FROZEN = ROOT / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/ckpt_100.safetensors"
FROZEN_SHA = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"
DATA = ROOT / "data/cache/stepb"
PLAN = "docs/plans/stepb_cinescale2.md"
CODE_PATHS = ["scripts", "src", PLAN, "pyproject.toml", "uv.lock", "results/cinescale2", "results/stepb/data_summary.json"]
RANK, ALPHA, DROPOUT, PEAK_LR, END_LR, WARMUP_FRAC, ACCUM, CLIP, MAX_SEQ = 16, 32.0, 0.05, 2e-5, 2e-6, 0.03, 16, 1.0, 1024
STOP_AFTER, STOP_WINDOW, STOP_LOSS, MEM_EVERY, RESUME_EVERY = 100, 50, 1.0, 50, 100


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SkipWatcher(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.message = None

    def emit(self, record):
        if "Skipping example" in record.getMessage():
            self.message = record.getMessage()


def allowed_roots() -> list[Path]:
    return [(ROOT / "data/cache/p3b/images_147456_32").resolve(), (DATA / "cs2_images_147456_32").resolve()]


def require_inputs(samples: list[dict]) -> None:
    """Every fed image is a shrunk copy under the Phase 3B ShotQA cache or the Step B CineScale2 cache (fail closed, exit 5/6)."""
    roots = allowed_roots()
    for s in samples:
        p = Path(s["shrunk"]).resolve()
        if not any(r in p.parents for r in roots):
            raise guards.DataRootViolation(f"{s['id']}: image outside the allowed roots: {p}")
    guards.require_images_readable(sorted({s["shrunk"] for s in samples}))


def run(a, log) -> None:
    run_dir = DATA / "runs" / a.tag
    logd = ROOT / "results/stepb/runs" / a.tag
    git = guards.git_state(CODE_PATHS)
    if git["dirty_files"] and not a.allow_dirty:
        raise guards.DirtyTree(f"uncommitted changes under {CODE_PATHS}: {git['dirty_files'][:6]}")
    append_jsonl(logd / "provenance.jsonl", {"tag": a.tag, "cand": a.cand, "code_commit": git["commit"], "dirty_files": git["dirty_files"],
                                             "model": f"{REPO}@{REVISION}", "init_adapter_sha256": FROZEN_SHA,
                                             "software": software_provenance(), "args": vars(a), "started": now_iso()})
    log(f"code commit {git['commit']} dirty={bool(git['dirty_files'])}")
    if guards.sha256_file(FROZEN) != FROZEN_SHA:
        raise guards.AdapterHashError(f"frozen adapter {FROZEN} does not match {FROZEN_SHA}")

    summ = json.loads((ROOT / "results/stepb/data_summary.json").read_text())["files"]
    for name in (a.cand, "monitor_shotqa500", "monitor_cs2_200"):
        if guards.sha256_file(DATA / f"{name}.jsonl") != summ[name]["sha256"]:
            raise guards.DataRootViolation(f"{name}.jsonl differs from the committed results/stepb/data_summary.json")
    samples = [json.loads(x) for x in (DATA / f"{a.cand}.jsonl").read_text().splitlines()]
    if a.max_samples:
        samples = samples[: a.max_samples]
    mon_qa = [json.loads(x) for x in (DATA / "monitor_shotqa500.jsonl").read_text().splitlines()]
    mon_cs2 = [json.loads(x) for x in (DATA / "monitor_cs2_200.jsonl").read_text().splitlines()]
    if a.max_samples:
        mon_qa, mon_cs2 = mon_qa[:20], mon_cs2[:20]
    require_inputs(samples + mon_qa + mon_cs2)
    n_micro = len(samples)
    total = math.ceil(n_micro / ACCUM)
    warmup = round(WARMUP_FRAC * total)
    milestones = {int(0.5 * total + 0.5): 50, total: 100}
    log(f"tag={a.tag} cand={a.cand} samples={n_micro} updates={total} warmup={warmup} milestones={milestones}")

    mx.random.seed(0)
    model, processor = load(pinned_model_path(REPO, REVISION))
    cfg = model.config.__dict__
    tl = guards.require_token_lengths(processor, samples + mon_qa + mon_cs2, MAX_SEQ, 144)
    log(f"token lengths: max {tl['max_tokens']} of {MAX_SEQ}, over limit 0")
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=RANK, alpha=ALPHA, dropout=DROPOUT, verbose=False)
    model.load_weights(list(mx.load(str(FROZEN)).items()), strict=False)
    log(f"initial adapter loaded: {FROZEN.name} sha256 {FROZEN_SHA}")

    schedule = optim.join_schedules([optim.linear_schedule(0.0, PEAK_LR, max(warmup, 1)),
                                     optim.cosine_decay(PEAK_LR, max(total - warmup, 1), END_LR)], [warmup])
    optimizer = optim.Adam(learning_rate=schedule, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    manifest_f = logd / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {"checkpoints": {}, "init_adapter_sha256": FROZEN_SHA}
    start_update, start_micro, loss_hist = 0, 0, []
    res_json = run_dir / "resume.json"
    if res_json.exists():
        guards.verify_manifest_files(manifest, run_dir)
        st = json.loads(res_json.read_text())
        model.load_weights(list(mx.load(str(run_dir / "resume_adapter.safetensors")).items()), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(run_dir / "resume_opt.safetensors")).items()))
        start_update, start_micro, loss_hist = st["update"], st["next_micro"], st["loss_hist"]
        mx.random.seed(10_000 + start_update)
        log(f"RESUMED from update {start_update} (next sample {start_micro})")
    hist = collections.deque(loss_hist[-STOP_WINDOW:], maxlen=STOP_WINDOW)

    probe_item = as_model_item(samples[0])
    leak = leakage_probe(model, VisionDataset(ListView([probe_item]), cfg, processor, None, True).process(probe_item))
    append_jsonl(logd / "leak_probes.jsonl", {"tag": a.tag, "resumed_at_update": start_update, **leak})
    log(f"leak probe: {leak}")
    guards.require_zero_leak(leak)

    eval_ds = VisionDataset(ListView([]), cfg, processor, None, True)
    for module in model.children().values():
        if hasattr(module, "layers"):
            grad_checkpoint(module.layers[0])
    loss_and_grad = nn.value_and_grad(model, partial(causal_loss_fn, train_on_completions=True))
    todo = [as_model_item(samples[i]) for i in range(start_micro, n_micro)]
    batches = iterate_batches(VisionDataset(ListView(todo), cfg, processor, None, True), 1, MAX_SEQ, train=False)
    watcher = SkipWatcher()
    logging.getLogger().addHandler(watcher)
    model.train()
    update, micro = start_update, start_micro
    t_up = time.perf_counter()
    while micro < n_micro:
        if update % MEM_EVERY == 0:
            pauses = wait_for_memory(25, log=log, poll_s=60)
            if pauses:
                log(f"memory guard: paused {pauses} x 60 s before update {update + 1}")
                t_up = time.perf_counter()
        acc, losses = None, []
        for _ in range(min(ACCUM, n_micro - micro)):
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
        acc, gnorm = optim.clip_grad_norm(acc, CLIP)
        optimizer.update(model, acc)
        mx.eval(model.parameters(), optimizer.state)
        update += 1
        ul = float(np.mean(losses))
        loss_hist.append(ul)
        hist.append(ul)
        sec = time.perf_counter() - t_up
        t_up = time.perf_counter()
        rec = {"update": update, "micro": micro, "loss": round(ul, 5), "lr": float(schedule(update - 1)), "gnorm": round(float(gnorm), 4),
               "sec": round(sec, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        append_jsonl(logd / "updates.jsonl", rec)
        if update % 10 == 0 or update <= 3 or update in milestones:
            log(f"update {update}/{total} loss {np.mean(list(hist)[-10:]):.4f} lr {rec['lr']:.2e} gnorm {float(gnorm):.2f} {sec:.1f}s/update "
                f"peak {rec['peak_gb']} GB")
            write_status(logd, a.tag, state="training", update=update, total=total, micro=micro, loss=round(float(np.mean(list(hist)[-10:])), 4))
        if not math.isfinite(ul) or (update > STOP_AFTER and np.mean(list(hist)) > STOP_LOSS):
            log(f"ABORT by the stopping rule at update {update}: loss {ul}, last-{STOP_WINDOW} mean {np.mean(list(hist)):.3f}")
            write_status(logd, a.tag, state="ABORTED by the pre-registered stopping rule", update=update, total=total)
            sys.exit(3)
        if update in milestones or update % RESUME_EVERY == 0 or micro >= n_micro:
            if update in milestones:
                pct = milestones[update]
                ck = run_dir / f"ckpt_{pct:03d}.safetensors"
                save_adapter(model, ck)
                digest = guards.sha256_file(ck)
                manifest["checkpoints"][str(pct)] = {"update": update, "micro": micro, "file": ck.name, "sha256": digest,
                                                     "saved": now_iso(), "code_commit": git["commit"]}
                manifest_f.write_text(json.dumps(manifest, indent=1))
                log(f"checkpoint {pct}% saved: sha256 {digest}")
                if pct < 100:
                    model.eval()
                    rq = summarize(evaluate_samples(model, eval_ds, [{**s, "images": [s["shrunk"]]} for s in mon_qa], ids4))
                    rc = summarize_cs2(evaluate_cs2(model, eval_ds, mon_cs2, ids4))
                    model.train()
                    manifest["checkpoints"][str(pct)]["monitor"] = {"shotqa500_mean": rq["mean_acc"], "cs2_200_acc": rc["acc"]}
                    manifest_f.write_text(json.dumps(manifest, indent=1))
                    log(f"monitor (NOT used for selection): ShotQA {len(mon_qa)} mean {rq['mean_acc']:.4f}; CineScale2 {len(mon_cs2)} acc {rc['acc']:.1f}")
            run_dir.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(run_dir / ".tmp_resume_adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            (run_dir / ".tmp_resume_adapter.safetensors").rename(run_dir / "resume_adapter.safetensors")
            mx.save_safetensors(str(run_dir / ".tmp_resume_opt.safetensors"), dict(tree_flatten(optimizer.state)))
            (run_dir / ".tmp_resume_opt.safetensors").rename(run_dir / "resume_opt.safetensors")
            (run_dir / ".tmp_resume.json").write_text(json.dumps({"update": update, "next_micro": micro, "loss_hist": loss_hist}))
            (run_dir / ".tmp_resume.json").rename(res_json)
        if a.stop_after_update and update >= a.stop_after_update:
            log(f"testing aid: exiting after update {update}")
            sys.exit(9)
    log(f"DONE: {update} updates, {micro} samples; last-{STOP_WINDOW} mean loss {np.mean(list(hist)):.4f}")
    (logd / "loss_curve.json").write_text(json.dumps(loss_hist))
    write_status(logd, a.tag, state="training finished", update=update, total=total)
    (run_dir / "DONE").write_text("done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["B0", "B1", "B2"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="pipeline check only (first N samples of the stream)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--stop-after-update", type=int, default=None)
    a = ap.parse_args()
    a.tag = a.tag or f"stepb_{a.cand}"
    (DATA / "runs" / a.tag).mkdir(parents=True, exist_ok=True)
    logd = ROOT / "results/stepb/runs" / a.tag
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
