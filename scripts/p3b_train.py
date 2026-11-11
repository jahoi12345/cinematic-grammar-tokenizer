"""Phase 3B LoRA training of Qwen3-VL-2B on ShotQA film-split TRAIN samples, as pre-registered in docs/phase3b_plan.md.
Resumable (seeded sample order, optimizer state in the resume file), accumulation 16, warm-up + cosine schedule, global-norm clipping,
causal-safe loss, mandatory leak probe, named checkpoints at 25/50/75/100% of the epoch with a 500-question monitoring evaluation.
Exit codes: 0 finished; 3 stopping rule (divergence); 4 leak probe not exactly 0.0; 5 image outside the ShotQA root; 6 unreadable image;
7 adapter hash missing/mismatch; 8 sample too long (would be truncated/skipped) or skipped by mlx-vlm; 10 uncommitted code or modified
pre-registered plan text; anything else = crash (the supervisor resumes). Codes 3-10 are never restarted (docs/phase3b_plan.md amendments).
Usage: uv run python scripts/p3b_train.py --tag <tag> [--n-samples 5000] [--resume-every 200] [--allow-dirty]"""
import argparse
import collections
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import shrunk_path
from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import (
    append_jsonl,
    now_iso,
    plan_original_unchanged,
    software_provenance,
    write_status,
)
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
IMG_CACHE = ROOT / "data/cache/p3b/images_147456_32"
CODE_PATHS = ["scripts", "src", "docs/phase3b_plan.md", "pyproject.toml", "uv.lock"]
# fixed hyper-parameters (docs/phase3b_plan.md section 4)
RANK, ALPHA, DROPOUT, PEAK_LR, END_LR, WARMUP_FRAC, ACCUM, CLIP, MAX_SEQ = 16, 32.0, 0.05, 1e-4, 1e-5, 0.03, 16, 1.0, 1024


class ListView:
    """Minimal list-backed dataset for mlx-vlm's VisionDataset (indexing + len)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SkipWatcher(logging.Handler):
    """mlx-vlm's batch iterator silently drops over-long samples with a warning; any such warning aborts the run (exit 8)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.message = None

    def emit(self, record):
        if "Skipping example" in record.getMessage():
            self.message = record.getMessage()


def _shrink(src: str) -> str:
    """Worker: shrink one image into the cache; returns '' on success or an error string."""
    try:
        shrunk_path(src, IMG_CACHE)
        return ""
    except (OSError, ValueError) as e:
        return f"{src}: {type(e).__name__}: {e}"


def prepare_images(sources: list[str]) -> None:
    """Shrink all needed images in parallel (idempotent) and verify the result; raises UnreadableImage listing failures."""
    uniq = sorted(set(sources))
    with ProcessPoolExecutor(8) as ex:
        errs = [e for e in ex.map(_shrink, uniq, chunksize=64) if e]
    if errs:
        raise guards.UnreadableImage(f"{len(errs)} images could not be prepared, first: {errs[:5]}")
    guards.require_images_readable([shrunk_path(s, IMG_CACHE) for s in uniq])


def as_item(s: dict) -> dict:
    return {"images": [shrunk_path(s["images"][0], IMG_CACHE)], "messages": s["messages"]}


def atomic_save(path: Path, writer) -> None:
    tmp = path.with_name(".tmp_" + path.name)
    writer(tmp)
    tmp.rename(path)


def run(a, log) -> None:
    run_dir = ROOT / "data/cache/p3b/runs" / a.tag
    logd = ROOT / "results/phase3b/runs" / a.tag
    train_jsonl = Path(a.train_jsonl)
    val_jsonl = Path(a.val_jsonl)

    # ---- code provenance and integrity (fail closed)
    git = guards.git_state(CODE_PATHS)
    if git["dirty_files"] and not a.allow_dirty:
        raise guards.DirtyTree(f"uncommitted changes under {CODE_PATHS}: {git['dirty_files'][:6]}")
    plan = plan_original_unchanged()
    if not plan["unchanged"]:
        raise guards.DirtyTree("the pre-registered plan text (before '## Amendments') differs from commit " + plan["plan_commit"])
    prov = {"tag": a.tag, "code_commit": git["commit"], "dirty_files": git["dirty_files"], "plan": plan, "model": f"{REPO}@{REVISION}",
            "software": software_provenance(), "args": vars(a), "started": now_iso()}
    append_jsonl(logd / "provenance.jsonl", prov)
    log(f"code commit {git['commit']} dirty={bool(git['dirty_files'])}; plan commit {plan['plan_commit']} original text unchanged")

    # ---- data guards (fail closed)
    samples = [json.loads(x) for x in train_jsonl.read_text().splitlines()]
    expected = json.loads((train_jsonl.parent / "summary.json").read_text())["train_samples"] if (train_jsonl.parent / "summary.json").exists() else None
    if expected is not None and train_jsonl.name == "train.jsonl" and len(samples) != expected:
        raise guards.DataRootViolation(f"train.jsonl has {len(samples)} samples but summary.json says {expected}")
    guards.require_samples_under_root(samples)
    order = np.random.default_rng(0).permutation(len(samples))
    if a.n_samples:
        order = order[: a.n_samples]
    used = [samples[i] for i in order]
    val = [json.loads(x) for x in val_jsonl.read_text().splitlines()]
    guards.require_samples_under_root(val)
    sub = [val[i] for i in sorted(np.random.default_rng(0).permutation(len(val))[: a.eval_n])]
    t0 = time.perf_counter()
    prepare_images([s["images"][0] for s in used] + [s["images"][0] for s in sub])
    log(f"images prepared and readable ({len(used)} train samples + {len(sub)} monitor samples) in {time.perf_counter() - t0:.0f}s")

    n_micro = len(order)
    total_updates = math.ceil(n_micro / ACCUM)
    warmup = round(WARMUP_FRAC * total_updates)
    milestones = {int(f * total_updates + 0.5): int(f * 100) for f in (0.25, 0.5, 0.75)}  # round half up: 799, 1598, 2396, 3195 for 3,195 updates (plan section 5)
    milestones[total_updates] = 100
    log(f"tag={a.tag} samples={n_micro} updates={total_updates} warmup={warmup} milestones={milestones} model={REPO}@{REVISION}")

    mx.random.seed(0)
    path = pinned_model_path(REPO, REVISION)
    model, processor = load(path)
    cfg = model.config.__dict__
    tl = guards.require_token_lengths(processor, used, MAX_SEQ, 144)
    log(f"token lengths (prompt + answer turn + at most 144 image tokens): max {tl['max_tokens']} of {MAX_SEQ}, over limit {tl['n_over_limit']}")
    ids4 = answer_token_ids(processor)
    log(f"answer token ids (read from the chat template): {ids4}")
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=RANK, alpha=ALPHA, dropout=DROPOUT, verbose=False)

    # ---- resume state
    start_update, start_micro, loss_hist, hist_updates = 0, 0, [], collections.deque(maxlen=100)
    res_json = run_dir / "resume.json"
    schedule = optim.join_schedules([optim.linear_schedule(0.0, PEAK_LR, max(warmup, 1)),
                                     optim.cosine_decay(PEAK_LR, max(total_updates - warmup, 1), END_LR)], [warmup])
    optimizer = optim.Adam(learning_rate=schedule, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    manifest_f = logd / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {"checkpoints": {}}
    if res_json.exists():
        n_ck = guards.verify_manifest_files(manifest, run_dir)  # every recorded checkpoint must still match its sha256
        st = json.loads(res_json.read_text())
        model.load_weights(list(mx.load(str(run_dir / "resume_adapter.safetensors")).items()), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(run_dir / "resume_opt.safetensors")).items()))
        start_update, start_micro, loss_hist = st["update"], st["next_micro"], st["loss_hist"]
        hist_updates.extend(loss_hist[-100:])
        mx.random.seed(10_000 + start_update)  # mlx 0.32 cannot restore the RNG state: dropout is reseeded deterministically (amendment 1)
        log(f"RESUMED from update {start_update} (next sample index {start_micro}); {n_ck} manifest checkpoints re-verified; dropout RNG reseeded")

    # ---- leak probe (mandatory, every (re)start, fail closed)
    probe_item = as_item(used[0])
    leak = leakage_probe(model, VisionDataset(ListView([probe_item]), cfg, processor, None, True).process(probe_item))
    if a.simulate_probe_failure:
        leak = {**leak, "without_mask": 0.125}
    append_jsonl(logd / "leak_probes.jsonl", {"tag": a.tag, "resumed_at_update": start_update, **leak})
    log(f"leak probe: {leak}")
    guards.require_zero_leak(leak)

    sub_items = [{**s, "images": [shrunk_path(s["images"][0], IMG_CACHE)]} for s in sub]
    eval_ds = VisionDataset(ListView([]), cfg, processor, None, True)
    for module in model.children().values():
        if hasattr(module, "layers"):
            grad_checkpoint(module.layers[0])
    loss_and_grad = nn.value_and_grad(model, partial(causal_loss_fn, train_on_completions=True))
    todo = [as_item(used[i]) for i in range(start_micro, n_micro)]
    batches = iterate_batches(VisionDataset(ListView(todo), cfg, processor, None, True), 1, MAX_SEQ, train=False)
    watcher = SkipWatcher()
    logging.getLogger().addHandler(watcher)

    model.train()
    update, micro = start_update, start_micro
    t_up = time.perf_counter()
    t_status = 0.0
    sec_hist = collections.deque(maxlen=20)
    last_ck = "-"
    while micro < n_micro:
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
        hist_updates.append(ul)
        lr = float(schedule(update - 1))
        sec = time.perf_counter() - t_up
        t_up = time.perf_counter()
        sec_hist.append(sec)
        rec = {"update": update, "micro": micro, "loss": round(ul, 5), "lr": lr, "gnorm": round(float(gnorm), 4),
               "sec": round(sec, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        with open(logd / "updates.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if update % 10 == 0 or update <= 3 or update in milestones:
            log(f"update {update}/{total_updates} loss {np.mean(list(hist_updates)[-10:]):.4f} lr {lr:.2e} gnorm {float(gnorm):.2f} "
                f"{sec:.1f}s/update peak {rec['peak_gb']} GB")
        if time.monotonic() - t_status >= 1800 or update in milestones:
            t_status = time.monotonic()
            write_status(logd, a.tag, state="training", update=update, total=total_updates, micro=micro,
                         loss=round(float(np.mean(list(hist_updates)[-10:])), 4), sec_per_update=round(float(np.mean(sec_hist)), 2),
                         peak_gb=rec["peak_gb"], last_checkpoint=last_ck)
        if not math.isfinite(ul) or (update > 200 and np.mean(list(hist_updates)) > 1.5):
            log(f"ABORT by stopping rule at update {update}: loss {ul}, last-100-update mean {np.mean(list(hist_updates)):.3f}")
            write_status(logd, a.tag, state="ABORTED by the pre-registered stopping rule (divergence)", update=update, total=total_updates,
                         micro=micro, loss=round(ul, 4), last_checkpoint=last_ck,
                         note="Stopping rule (plan section 8): non-finite loss, or mean loss of the last 100 updates > 1.5 after update 200. Not restarted.")
            sys.exit(3)
        if update in milestones or update % a.resume_every == 0 or micro >= n_micro:
            if update in milestones:
                pct = milestones[update]
                ck = run_dir / f"ckpt_{pct:03d}.safetensors"
                save_adapter(model, ck)
                digest = guards.sha256_file(ck)  # raises AdapterHashError if the file is missing or empty
                manifest["checkpoints"][str(pct)] = {"update": update, "micro": micro, "file": ck.name, "sha256": digest,
                                                     "saved": now_iso(), "code_commit": git["commit"]}
                manifest_f.write_text(json.dumps(manifest, indent=1))
                last_ck = f"{pct}% (update {update}) sha256 {digest[:16]}..."
                log(f"checkpoint {pct}% saved: sha256 {digest}")
                if pct < 100 or a.n_samples:
                    model.eval()
                    r = summarize(evaluate_samples(model, eval_ds, sub_items, ids4))
                    model.train()
                    manifest["checkpoints"][str(pct)]["monitor_500"] = {"mean_acc": r["mean_acc"], "n": r["n"],
                                                                         "per_attribute": {k: v["acc"] for k, v in r["per_attribute"].items()}}
                    manifest_f.write_text(json.dumps(manifest, indent=1))
                    log(f"monitor (500 val questions, NOT used for selection): mean acc {r['mean_acc']:.4f}")
            atomic_save(run_dir / "resume_adapter.safetensors", lambda p: mx.save_safetensors(str(p), dict(tree_flatten(model.trainable_parameters()))))
            atomic_save(run_dir / "resume_opt.safetensors", lambda p: mx.save_safetensors(str(p), dict(tree_flatten(optimizer.state))))
            tmp = run_dir / ".tmp_resume.json"
            tmp.write_text(json.dumps({"update": update, "next_micro": micro, "loss_hist": loss_hist}))
            tmp.rename(res_json)
        if a.stop_after_update and update >= a.stop_after_update:
            log(f"testing aid: exiting after update {update}")
            sys.exit(9)
    log(f"DONE: {update} updates, {micro} samples; final 100-update mean loss {np.mean(list(hist_updates)):.4f}")
    (logd / "loss_curve.json").write_text(json.dumps(loss_hist))
    write_status(logd, a.tag, state="training finished", update=update, total=total_updates, micro=micro,
                 loss=round(float(np.mean(list(hist_updates)[-10:])), 4), sec_per_update=round(float(np.mean(sec_hist)), 2),
                 last_checkpoint=last_ck)
    (run_dir / "DONE").write_text("done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-jsonl", default=str(ROOT / "data/cache/p3b/train.jsonl"))
    ap.add_argument("--val-jsonl", default=str(ROOT / "data/cache/p3b/val.jsonl"))
    ap.add_argument("--n-samples", type=int, default=None, help="use only the first N of the seeded order (pipeline check)")
    ap.add_argument("--resume-every", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--allow-dirty", action="store_true", help="pipeline checks only: permit uncommitted code")
    ap.add_argument("--simulate-probe-failure", action="store_true", help="test aid: force the leak probe to report a leak (can only make the run fail)")
    ap.add_argument("--stop-after-update", type=int, default=None, help="testing aid: exit (code 9) after this update")
    a = ap.parse_args()
    (ROOT / "data/cache/p3b/runs" / a.tag).mkdir(parents=True, exist_ok=True)
    logd = ROOT / "results/phase3b/runs" / a.tag
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
        write_status(logd, a.tag, state=f"STOPPED: {type(e).__name__} (exit {e.exit_code})", note=f"{e}\n\nNot restarted.")
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
