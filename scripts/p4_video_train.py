"""Phase 4 video-LoRA training stage (docs/phase4_video_plan.md Amendment 2): one cross-fit fold or the final fit. Resumable, per-epoch checkpoints and out-of-fold option scoring.
K frames per clip as K images, base Qwen3-VL-2B (pinned, no earlier adapter), causal-safe loss, leak probe at EVERY start (exit 4), accumulation 8, 3 epochs, warm-up 10 updates then
cosine 1e-4 -> 1e-5, clip 1.0. Train labels only (locked MovementLabelStore; a held-out clip of a fold is a TRAIN-source clip). Never touches validation labels, ShotBench movement or CameraBench.
Exit codes: 0 finished; 3 stopping rule (non-finite loss, or mean loss > 5.0 over 50 micro-steps after the first 100); 4 leak probe not exactly 0.0; 6 unreadable frame; 8 sample skipped by mlx-vlm;
10 uncommitted code or changed plan original text; anything else = crash (the supervisor resumes). Usage: uv run python scripts/p4_video_train.py --stage fold0 --frames 8 --tokens 64 --tag v1"""
import argparse
import collections
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# smaller Metal command buffers: with the default size the Metal GPU watchdog (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) killed 8x64 steps in the pipeline check
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2")  # 8 until fold1 (Amendment 4), 4 for fold2 (Amendment 6), 2 from the relaunch of Amendment 7: scheduling only

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
from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import (
    append_jsonl,
    now_iso,
    plan_original_unchanged,
    software_provenance,
    write_status,
)
from cgt.p3b_vlm.score import answer_token_ids, letter_logits
from cgt.p4_camera_move import crossfit as X
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.labels import MovementLabelStore

mx.set_cache_limit(3 * 1024**3)  # cap the MLX buffer cache: unbounded caching grew the process to 24 GB during scoring and made the 32 GB machine swap (no computed value changes)

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
FRAMES = ROOT / "data/cache/p4/video_frames"
PLAN_FILE, PLAN_COMMIT = "docs/phase4_video_plan.md", "49e248a"  # the Amendment 2 commit: the original text above `## Amendments` must equal it
CODE_PATHS = ["scripts", "src", PLAN_FILE, "pyproject.toml", "uv.lock"]
RANK, ALPHA, DROPOUT, CLIP, MAX_SEQ = 16, 32.0, 0.05, 1.0, 2048
LETTERS = ("A", "B", "C", "D")


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


def _frames(args) -> str:
    clip, k, t = args
    try:
        V.extract_frames(QA / clip, k, t, FRAMES)
        return ""
    except (OSError, ValueError) as e:
        return f"{clip}: {type(e).__name__}: {e}"


def prepare_frames(clips: list[str], k: int, t: int) -> None:
    with ProcessPoolExecutor(6) as ex:
        errs = [e for e in ex.map(_frames, [(c, k, t) for c in clips], chunksize=8) if e]
    if errs:
        raise guards.UnreadableImage(f"{len(errs)} clips could not be prepared, first: {errs[:3]}")


def atomic_save(path: Path, writer) -> None:
    tmp = path.with_name(".tmp_" + path.name)
    writer(tmp)
    tmp.rename(path)


def score_rows(model, ds, samples: list[dict], rows: list[dict], ids4: dict[str, int]) -> list[dict]:
    """Letter logits of the four options for each held-out sample (eval mode, no gradient). The gold token is checked against the template's letter token."""
    out = []
    for s, r in zip(samples, rows, strict=True):
        item = ds.process({"images": s["images"], "messages": s["messages"]})
        four, gold_tok = letter_logits(model, item, ids4)
        if gold_tok != ids4[r["gold_letter"]]:
            raise ValueError(f"{r['clip']}: the tokenised gold answer is not the chat template's token for its letter")
        out.append({"clip": r["clip"], "source": r["source"], "gold": r["gold_letter"], "pred": LETTERS[int(np.argmax(four))], "logits": [round(float(x), 4) for x in four]})
        mx.clear_cache()
    return out


def run(a, log) -> None:
    stage = a.stage
    run_dir = ROOT / "data/cache/p4/video_runs" / a.tag / stage
    logd = ROOT / "results/phase4/runs" / f"video_{a.tag}" / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    logd.mkdir(parents=True, exist_ok=True)
    # ---- provenance and integrity (fail closed)
    git = guards.git_state(CODE_PATHS)
    if git["dirty_files"] and not a.allow_dirty:
        raise guards.DirtyTree(f"uncommitted changes under {CODE_PATHS}: {git['dirty_files'][:6]}")
    plan = plan_original_unchanged(plan_commit=PLAN_COMMIT, plan_file=PLAN_FILE)
    if not plan["unchanged"]:
        raise guards.DirtyTree(f"the original text of {PLAN_FILE} differs from commit {PLAN_COMMIT}")
    prov = {"tag": a.tag, "stage": stage, "code_commit": git["commit"], "dirty_files": git["dirty_files"], "plan": plan, "plan_commit": PLAN_COMMIT, "frames": a.frames, "tokens": a.tokens,
            "model": "Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203 (base, no adapter)", "software": software_provenance(), "args": vars(a), "started": now_iso()}
    append_jsonl(logd / "provenance.jsonl", prov)
    log(f"stage {stage}: code {git['commit'][:12]} dirty={bool(git['dirty_files'])}; plan original text unchanged vs {PLAN_COMMIT}")
    # ---- data: TRAIN labels only (the store raises on a validation clip)
    man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    store = MovementLabelStore(man["clips"]["val"])
    rows = [r for r in store.rows(man["clips"]["train"]) if r["atoms"]]
    if a.n_clips:
        rows = rows[: a.n_clips]
    tr_rows, ho_rows = X.stage_split(rows, stage)
    log(f"{len(rows)} train-source clips with a schema atom; stage trains on {len(tr_rows)}, holds out {len(ho_rows)}")
    t0 = time.perf_counter()
    prepare_frames([r["clip"] for r in tr_rows + ho_rows], a.frames, a.tokens)
    tr_s = [V.build_sample(r, V.extract_frames(QA / r["clip"], a.frames, a.tokens, FRAMES)) for r in tr_rows]
    ho_s = [V.build_sample(r, V.extract_frames(QA / r["clip"], a.frames, a.tokens, FRAMES)) for r in ho_rows]
    guards.require_images_readable([p for s in tr_s[:20] + ho_s[:20] for p in s["images"]])
    log(f"frames ready ({len(tr_s) + len(ho_s)} clips x {a.frames}) in {time.perf_counter() - t0:.0f}s")
    n = len(tr_s)
    total = X.total_updates(n, a.epochs)
    # ---- model
    mx.random.seed(0)
    model, processor = load(pinned_model_path())
    V.configure_processor(processor)
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=RANK, alpha=ALPHA, dropout=DROPOUT, verbose=False)
    optimizer = optim.Adam(learning_rate=X.PEAK_LR, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    res_json = run_dir / "resume.json"
    manifest_f = logd / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {"checkpoints": {}}
    st = {"epoch": 1, "micro": 0, "update": 0, "loss_hist": []}
    if res_json.exists():
        n_ck = guards.verify_manifest_files(manifest, run_dir)
        st = json.loads(res_json.read_text())
        model.load_weights(list(mx.load(str(run_dir / "resume_adapter.safetensors")).items()), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(run_dir / "resume_opt.safetensors")).items()))
        mx.random.seed(10_000 + st["update"])  # mlx cannot restore the RNG state: the dropout stream is reseeded deterministically on a resume
        log(f"RESUMED at epoch {st['epoch']} micro {st['micro']} update {st['update']}; {n_ck} checkpoints re-verified; dropout RNG reseeded")
    ds_eval = VisionDataset(ListView([]), cfg, processor, None, True)
    # ---- leak probe at EVERY start (fail closed)
    probe_item = {"images": tr_s[0]["images"], "messages": tr_s[0]["messages"]}
    leak = leakage_probe(model, VisionDataset(ListView([probe_item]), cfg, processor, None, True).process(probe_item))
    if a.simulate_probe_failure:
        leak = {**leak, "without_mask": 0.125}
    append_jsonl(logd / "leak_probes.jsonl", {"tag": a.tag, "stage": stage, "resumed_at_update": st["update"], **leak})
    log(f"leak probe: {leak}")
    guards.require_zero_leak(leak)
    for module in model.children().values():
        if hasattr(module, "layers"):
            grad_checkpoint(module.layers[0])
    loss_and_grad = nn.value_and_grad(model, lambda m, b: causal_loss_fn(m, b, train_on_completions=True))
    watcher = SkipWatcher()
    logging.getLogger().addHandler(watcher)

    def ensure_oof(e: int) -> None:
        """Out-of-fold scoring of the held-out clips with the epoch-e checkpoint (written atomically; recomputed from the checkpoint if a crash lost it)."""
        f = logd / f"oof_e{e}.json"
        if not ho_s or f.exists():
            return
        cur = dict(tree_flatten(model.trainable_parameters()))
        model.load_weights(list(mx.load(str(run_dir / f"ckpt_e{e}.safetensors")).items()), strict=False)
        model.eval()
        recs = score_rows(model, ds_eval, ho_s, ho_rows, ids4)
        model.train()
        model.load_weights(list(cur.items()), strict=False)
        atomic_save(f, lambda p: p.write_text(json.dumps({"stage": stage, "epoch": e, "n": len(recs), "acc": float(np.mean([r["pred"] == r["gold"] for r in recs])), "records": recs})))
        mx.clear_cache()
        log(f"OOF epoch {e}: option accuracy {np.mean([r['pred'] == r['gold'] for r in recs]):.4f} on {len(recs)} held-out clips (TRAIN sources)")

    def save_resume(epoch: int, micro: int, update: int, loss_hist: list) -> None:
        atomic_save(run_dir / "resume_adapter.safetensors", lambda p: mx.save_safetensors(str(p), dict(tree_flatten(model.trainable_parameters()))))
        atomic_save(run_dir / "resume_opt.safetensors", lambda p: mx.save_safetensors(str(p), dict(tree_flatten(optimizer.state))))
        tmp = run_dir / ".tmp_resume.json"
        tmp.write_text(json.dumps({"epoch": epoch, "micro": micro, "update": update, "loss_hist": loss_hist}))
        tmp.rename(res_json)

    resumed_at = st["update"]
    prior_loss: dict[tuple[int, int], float] = {}  # (resumed_at, update) -> loss logged by an EARLIER process that started from the same resume state
    uf = logd / "updates.jsonl"
    if uf.exists():
        for line in uf.read_text().splitlines():
            rec = json.loads(line)
            if "resumed_at" in rec:
                prior_loss[(rec["resumed_at"], rec["update"])] = rec["loss"]
    for e in range(1, st["epoch"]):
        ensure_oof(e)  # epochs finished before a crash whose OOF file is missing
    model.train()
    update, loss_hist = st["update"], st["loss_hist"]
    micro_win = collections.deque(maxlen=50)
    n_micro_total = (st["epoch"] - 1) * n + st["micro"]  # stage-total micro-steps, retained across restarts (Amendment 8: the stopping rule's grace does not restart)
    sec_hist = collections.deque(maxlen=20)
    t_up, t_status, last_ck = time.perf_counter(), 0.0, "-"
    for epoch in range(st["epoch"], a.epochs + 1):
        order = X.epoch_order(n, stage, epoch)
        start = st["micro"] if epoch == st["epoch"] else 0
        todo = [{"images": tr_s[i]["images"], "messages": tr_s[i]["messages"]} for i in order[start:]]
        batches = iterate_batches(VisionDataset(ListView(todo), cfg, processor, None, True), 1, MAX_SEQ, train=False)
        micro = start
        while micro < n:
            acc, losses = None, []
            for _ in range(min(X.ACCUM, n - micro)):
                batch = next(batches)
                if watcher.message:
                    raise guards.TokenLengthViolation(watcher.message)
                loss, grad = loss_and_grad(model, batch)
                acc = grad if acc is None else tree_map(lambda x, y: x + y, acc, grad)
                mx.eval(loss, acc)
                mx.clear_cache()
                losses.append(float(loss))
                micro_win.append(float(loss))
                micro += 1
                n_micro_total += 1
            n_acc = len(losses)
            acc = tree_map(lambda g, k=n_acc: g / k, acc)
            acc, gnorm = optim.clip_grad_norm(acc, CLIP)
            update += 1
            lr = X.lr_at(update, total)
            optimizer.learning_rate = lr
            optimizer.update(model, acc)
            mx.eval(model.parameters(), optimizer.state)
            ul = float(np.mean(losses))
            loss_hist.append(ul)
            sec = time.perf_counter() - t_up
            t_up = time.perf_counter()
            sec_hist.append(sec)
            prev = prior_loss.get((resumed_at, update))
            if prev is not None:  # Amendments 7/8: EVERY update that an earlier process started from the same resume state already logged must be reproduced to a relative 1e-3
                ok = abs(ul - prev) <= 1e-3 * max(abs(prev), 1e-8)
                append_jsonl(logd / "resume_checks.jsonl", {"resumed_at": resumed_at, "update": update, "loss_now": round(ul, 5), "loss_before": round(prev, 5), "ok": ok})
                log(f"resume check update {update}: loss {ul:.5f} vs earlier process {prev:.5f}: {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    write_status(logd, f"{a.tag}/{stage}", state="STOPPED: a resume did not reproduce the earlier loss (exit 13); NOT restarted", update=update, total=total, micro=micro, last_checkpoint=last_ck)
                    sys.exit(13)
            append_jsonl(logd / "updates.jsonl", {"update": update, "epoch": epoch, "micro": micro, "resumed_at": resumed_at, "loss": round(ul, 5), "lr": lr, "gnorm": round(float(gnorm), 4), "sec": round(sec, 2),
                                                  "peak_gb": round(mx.get_peak_memory() / 1e9, 2)})
            if update % 10 == 0 or update <= 3:
                log(f"epoch {epoch} update {update}/{total} loss {np.mean(loss_hist[-10:]):.4f} lr {lr:.2e} gnorm {float(gnorm):.2f} {sec:.1f}s/update peak {mx.get_peak_memory() / 1e9:.1f} GB")
            if time.monotonic() - t_status >= 1800:
                t_status = time.monotonic()
                write_status(logd, f"{a.tag}/{stage}", state=f"training epoch {epoch}/{a.epochs}", update=update, total=total, micro=micro, loss=round(float(np.mean(loss_hist[-10:])), 4),
                             sec_per_update=round(float(np.mean(sec_hist)), 2), peak_gb=round(mx.get_peak_memory() / 1e9, 2), last_checkpoint=last_ck)
            if not math.isfinite(ul) or (n_micro_total >= 100 and len(micro_win) == 50 and float(np.mean(micro_win)) > 5.0):
                log(f"ABORT by the stopping rule at update {update}: loss {ul}, 50-micro-step mean {np.mean(micro_win):.3f}")
                write_status(logd, f"{a.tag}/{stage}", state="ABORTED by the pre-registered stopping rule", update=update, total=total, micro=micro, last_checkpoint=last_ck,
                             note="Non-finite loss, or mean loss > 5.0 over 50 micro-steps after the first 100. Not restarted.")
                sys.exit(3)
            if update % a.resume_every == 0 and micro < n:
                save_resume(epoch, micro, update, loss_hist)
            if a.stop_after_update and update >= a.stop_after_update:
                log(f"testing aid: exiting after update {update}")
                sys.exit(9)
        # ---- end of epoch: checkpoint, resume state at the next epoch, out-of-fold scoring
        ck = run_dir / f"ckpt_e{epoch}.safetensors"
        save_adapter(model, ck)
        mx.clear_cache()
        digest = guards.sha256_file(ck)
        manifest["checkpoints"][str(epoch)] = {"epoch": epoch, "update": update, "file": ck.name, "sha256": digest, "saved": now_iso(), "code_commit": git["commit"]}
        manifest_f.write_text(json.dumps(manifest, indent=1))
        last_ck = f"epoch {epoch} sha256 {digest[:16]}..."
        log(f"checkpoint epoch {epoch} saved: sha256 {digest}")
        save_resume(epoch + 1, 0, update, loss_hist)
        ensure_oof(epoch)
        st["micro"] = 0
    (logd / "loss_curve.json").write_text(json.dumps(loss_hist))
    write_status(logd, f"{a.tag}/{stage}", state="stage finished", update=update, total=total, micro=n, loss=round(float(np.mean(loss_hist[-10:])), 4),
                 sec_per_update=round(float(np.mean(sec_hist)) if sec_hist else 0.0, 2), last_checkpoint=last_ck)
    (run_dir / "DONE").write_text("done")
    log(f"DONE stage {stage}: {update} updates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fold0", "fold1", "fold2", "final"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=X.EPOCHS)
    ap.add_argument("--resume-every", type=int, default=25)
    ap.add_argument("--n-clips", type=int, default=None, help="pipeline check only: use the first N train-source clips")
    ap.add_argument("--allow-dirty", action="store_true", help="pipeline checks only")
    ap.add_argument("--simulate-probe-failure", action="store_true", help="test aid: can only make the leak probe fail")
    ap.add_argument("--stop-after-update", type=int, default=None, help="test aid: exit with code 9 after this update")
    a = ap.parse_args()
    logd = ROOT / "results/phase4/runs" / f"video_{a.tag}" / a.stage
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
        write_status(logd, f"{a.tag}/{a.stage}", state=f"STOPPED: {type(e).__name__} (exit {e.exit_code})", note=f"{e}\n\nNot restarted.")
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
