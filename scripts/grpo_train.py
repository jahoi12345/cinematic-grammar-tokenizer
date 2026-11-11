"""GRPO pilot training. Built from scripts/step3_train.py
(not changed): same model, LoRA layout, 1x images, memory guard, resume state every 25 updates, MLX buffer env logged.

Per update: Q_PER_UPDATE seeded ShotQA TRAIN questions (cgt.grpo.data.pilot_questions; never val/test). For each question:
  1. reference pass (frozen start adapter aba56883..., swapped in, no gradient): log-probabilities at the answer position;
  2. policy pass (current LoRA, gradient): logits at the answer position (the prompt only is fed, so the answer is never visible);
     G answers are sampled from softmax(logits / T) with key mx.random.key(SAMPLE_SEED + stream position); reward = 1 if the sampled
     token parses to the gold option letter (cgt.grpo.reward, official option matcher) else 0; GRPO loss cgt.grpo.loss.grpo_loss
     (group-normalised advantages, clipped ratio, beta x k3 KL to the start adapter, completion-only mask).
Gradients are averaged over the update's questions, clipped to global norm 1.0, Adam (peak LR, linear warm-up, then constant).
LoRA dropout is OFF (model.eval()) so the sampling and the loss see the same policy. One final adapter `final.safetensors`
(registered in the manifest). Never reads ShotBench, CameraBench, gold, dev or val.
Exit codes: 0 finished; 3 non-finite loss; 5 data; 6 image; 7 adapter hash; 8 length; 9 testing stop; 10 dirty tree.
  HF_HUB_OFFLINE=1 nice -n 19 <venv>/bin/python scripts/grpo_train.py [--n-updates 2 --tag grpo_pc --gen-probe 2]"""
import argparse
import json
import math
import os
import sys
import time
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
from mlx_vlm.trainer.utils import (
    find_all_linear_names,
    get_peft_model,
    grad_checkpoint,
    save_adapter,
)
from mlx_vlm.utils import load

from cgt.cinescale2.semantic import wait_for_memory
from cgt.grpo import config as C
from cgt.grpo.data import pilot_questions
from cgt.grpo.loss import group_stats, grpo_loss
from cgt.grpo.reward import LETTERS, letter_table
from cgt.p3b_vlm import guards
from cgt.p3b_vlm.adapter_load import load_adapter_checked
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.runstatus import append_jsonl, now_iso, software_provenance, write_status
from cgt.p3b_vlm.score import answer_token_ids

_SKIP = ("input_ids", "pixel_values", "attention_mask", "completion_mask")
REASON_SUFFIX = "\nThink briefly about the visual evidence, then give the letter."   # generation-cost probe only (not trained)


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class NonFinite(guards.FailClosed):
    exit_code = 3


def prompt_inputs(vds, sample: dict, ids4: dict) -> tuple:
    """(prompt ids (1, p), pixel values, extra kwargs, gold letter index). Only positions BEFORE the first supervised token are fed."""
    item = vds.process({"images": sample["images"], "messages": sample["messages"]})
    ids = np.array(item["input_ids"]).reshape(1, -1)
    cm = np.array(item["completion_mask"]).reshape(-1)
    p = int(np.flatnonzero(cm)[0])
    if int(ids[0, p]) != ids4[sample["answer"]]:
        raise guards.DataRootViolation(f"{sample['id']}: tokenised gold answer is not the template's letter token")
    kw = {k: mx.array(np.array(v)) for k, v in item.items() if k not in _SKIP}
    return mx.array(ids[:, :p]), mx.array(np.array(item["pixel_values"])), kw, LETTERS.index(sample["answer"])


def last_logp(model, ids, pv, kw):
    logits = model(ids, pv, None, **kw).logits[0, -1].astype(mx.float32)
    return logits - mx.logsumexp(logits)


def question_loss(model, ids, pv, kw, ref_logp, key, table, gold_idx):
    """GRPO loss of one question (B = 1, L = 1) + diagnostics. Sampling uses the detached policy log-probabilities."""
    logp = last_logp(model, ids, pv, kw)
    a = mx.random.categorical(mx.stop_gradient(logp) / C.TEMPERATURE, num_samples=C.G, key=key)
    lp_new = logp[a]
    lp_ref = ref_logp[a]
    r = (table[a] == gold_idx).astype(mx.float32)
    shape = (1, C.G, 1)
    loss = grpo_loss(mx, lp_new.reshape(shape), mx.stop_gradient(lp_new).reshape(shape), lp_ref.reshape(shape), r.reshape(1, C.G),
                     mx.ones(shape), C.CLIP_EPS, C.BETA, C.ADV_EPS)
    p = mx.exp(mx.stop_gradient(logp))
    exact_kl = mx.sum(p * (mx.stop_gradient(logp) - ref_logp))
    valid = (table[a] >= 0).astype(mx.float32).mean()
    return loss, r, exact_kl, valid


def gen_probe(model, processor, samples: list[dict], n_q: int, log) -> dict:
    """Timing of short-reasoning generation (64 new tokens max, temperature 1, 2 samples per question). Text is never stored."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    out = []
    for s in samples[:n_q]:
        text = s["messages"][0]["content"]
        text = text if isinstance(text, str) else " ".join(c.get("text", "") for c in text if isinstance(c, dict))
        prompt = apply_chat_template(processor, model.config, text + REASON_SUFFIX, num_images=1)
        for _ in range(2):
            t0 = time.perf_counter()
            try:
                g = generate(model, processor, prompt, image=s["images"], max_tokens=64, temperature=1.0, verbose=False)
                out.append({"seconds": round(time.perf_counter() - t0, 2), "generation_tokens": g.generation_tokens,
                            "prompt_tokens": g.prompt_tokens, "generation_tps": round(g.generation_tps, 2),
                            "prompt_tps": round(g.prompt_tps, 2)})
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:   # a probe only: the error type is recorded, the pipeline check continues
                out.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
    ok = [x for x in out if "error" not in x]
    tps = float(np.mean([x["generation_tps"] for x in ok])) if ok else None
    res = {"runs": out, "mean_generation_tps": tps,
           "proj_sec_per_question_64tok_G": round(C.G * (64 / tps + float(np.mean([x["prompt_tokens"] / max(x["prompt_tps"], 1e-6)
                                                                                 for x in ok]))), 1) if ok and tps else None}
    log(f"generation probe: {res}")
    return res


def run(a, log) -> None:
    run_dir = C.DATA / "runs" / a.tag
    logd = Path(a.results_dir)
    git = guards.git_state(C.CODE_PATHS)
    if git["dirty_files"] and not a.allow_dirty:
        raise guards.DirtyTree(f"uncommitted changes under {C.CODE_PATHS}: {git['dirty_files'][:6]}")
    mlx_env = {k: os.environ.get(k) for k in C.MLX_ENV}
    append_jsonl(logd / "provenance.jsonl", {"tag": a.tag, "code_commit": git["commit"], "dirty_files": git["dirty_files"],
                                             "model": f"{C.REPO}@{C.REVISION}", "software": software_provenance(), "args": vars(a),
                                             "init_adapter_sha256": C.INIT_SHA, "train_sha256": C.TRAIN_SHA, "mlx_env_seen": mlx_env,
                                             "grpo": {"G": C.G, "T": C.TEMPERATURE, "Q_PER_UPDATE": C.Q_PER_UPDATE, "BETA": C.BETA,
                                                      "CLIP_EPS": C.CLIP_EPS, "PEAK_LR": C.PEAK_LR, "WARMUP_FRAC": C.WARMUP_FRAC,
                                                      "GRAD_CLIP": C.GRAD_CLIP, "ANSWER_MODE": C.ANSWER_MODE},
                                             "started": now_iso()})
    log(f"code commit {git['commit']} dirty={bool(git['dirty_files'])}; MLX buffer env seen: {mlx_env}")
    if guards.sha256_file(C.INIT_ADAPTER) != C.INIT_SHA:
        raise guards.AdapterHashError(f"initial adapter {C.INIT_ADAPTER} does not match {C.INIT_SHA}")
    total = a.n_updates
    qs = pilot_questions(total * C.Q_PER_UPDATE)
    guards.require_images_readable(sorted({s["images"][0] for s in qs}))
    warmup = max(round(C.WARMUP_FRAC * C.N_UPDATES), 1)
    log(f"tag={a.tag} questions={len(qs)} updates={total} warmup={warmup} G={C.G} beta={C.BETA} clip={C.CLIP_EPS} lr={C.PEAK_LR}")

    mx.random.seed(C.MX_SEED)
    model, processor = load(pinned_model_path(C.REPO, C.REVISION))
    cfg = model.config.__dict__
    tl = guards.require_token_lengths(processor, qs, C.MAX_SEQ, C.IMAGE_TOKENS)
    log(f"token lengths: max {tl['max_tokens']} of {C.MAX_SEQ}, over limit {tl['n_over_limit']}")
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=C.RANK, alpha=C.ALPHA, dropout=C.DROPOUT,
                           verbose=False)
    n_init = load_adapter_checked(model, C.INIT_ADAPTER)
    ref = dict(mx.load(str(C.INIT_ADAPTER)))
    trainable = {k for k, _ in tree_flatten(model.trainable_parameters())}
    if not set(ref) <= trainable:   # the swap below restores trainable tensors only, so the reference must hold nothing else
        raise guards.AdapterHashError(f"start adapter holds non-trainable tensors, e.g. {sorted(set(ref) - trainable)[:3]}")
    log(f"start adapter loaded: {n_init} tensors matched (also kept as the frozen KL reference)")
    model.eval()   # LoRA dropout off: sampling and loss see the same policy (gradients still flow)

    vds = VisionDataset(ListView([]), cfg, processor, None, True)
    t_tab = time.perf_counter()
    vocab = int(last_logp(model, *prompt_inputs(vds, qs[0], ids4)[:3]).shape[0])
    table_np = letter_table(lambda t: processor.tokenizer.decode(t), vocab)
    for i, L in enumerate(LETTERS):
        if table_np[ids4[L]] != i:
            raise guards.DataRootViolation(f"letter table maps the template token of {L} to {table_np[ids4[L]]}")
    table = mx.array(table_np)
    log(f"letter table: vocab {vocab}, {int((table_np >= 0).sum())} tokens parse to A-D ({time.perf_counter() - t_tab:.1f} s)")

    schedule = optim.join_schedules([optim.linear_schedule(0.0, C.PEAK_LR, warmup), lambda _: C.PEAK_LR], [warmup])
    optimizer = optim.Adam(learning_rate=schedule, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    manifest_f = logd / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {"checkpoints": {}, "model": f"{C.REPO}@{C.REVISION}"}
    start = 0
    res_json = run_dir / "resume.json"
    if res_json.exists():
        st = json.loads(res_json.read_text())
        model.load_weights(list(mx.load(str(run_dir / "resume_adapter.safetensors")).items()), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(run_dir / "resume_opt.safetensors")).items()))
        start = st["update"]
        log(f"RESUMED from update {start}")
        append_jsonl(logd / "restarts.jsonl", {"resumed_at_update": start, "time": now_iso()})

    for module in model.children().values():
        if hasattr(module, "layers"):
            grad_checkpoint(module.layers[0])
    loss_and_grad = nn.value_and_grad(model, question_loss)
    for update in range(start, total):
        if update % C.MEM_EVERY == 0:
            pauses = wait_for_memory(C.MEM_FLOOR, log=log, poll_s=60)
            if pauses:
                append_jsonl(logd / "memory_pauses.jsonl", {"before_update": update + 1, "pauses_60s": pauses})
        t0 = time.perf_counter()
        batch = list(range(update * C.Q_PER_UPDATE, (update + 1) * C.Q_PER_UPDATE))
        inputs = [prompt_inputs(vds, qs[k], ids4) for k in batch]
        cur = tree_flatten(model.trainable_parameters())
        model.load_weights(list(ref.items()), strict=False)          # frozen start adapter
        refs = []
        for ids, pv, kw, _ in inputs:
            lp = mx.stop_gradient(last_logp(model, ids, pv, kw))
            mx.eval(lp)
            refs.append(lp)
        model.load_weights(cur, strict=False)                        # current policy back
        mx.clear_cache()
        t_ref = time.perf_counter() - t0
        acc, losses, rs, kls, valids = None, [], [], [], []
        for k, (ids, pv, kw, gold), ref_lp in zip(batch, inputs, refs, strict=True):
            (loss, r, ekl, valid), grad = loss_and_grad(model, ids, pv, kw, ref_lp, mx.random.key(C.SAMPLE_SEED + k), table,
                                                     mx.array(gold))
            acc = grad if acc is None else tree_map(lambda x, y: x + y, acc, grad)
            mx.eval(loss, acc, r, ekl, valid)
            mx.clear_cache()
            losses.append(float(loss))
            rs.append(np.array(r))
            kls.append(float(ekl))
            valids.append(float(valid))
        nq = len(batch)
        acc = tree_map(lambda g, n=nq: g / n, acc)
        acc, gnorm = optim.clip_grad_norm(acc, C.GRAD_CLIP)
        optimizer.update(model, acc)
        mx.eval(model.parameters(), optimizer.state)
        sec = time.perf_counter() - t0
        ul = float(np.mean(losses))
        rec = {"update": update + 1, "loss": round(ul, 6), **{k: round(v, 4) for k, v in group_stats(np.stack(rs)).items()},
               "kl_exact_mean": round(float(np.mean(kls)), 6), "frac_valid_letter": round(float(np.mean(valids)), 4),
               "lr": float(schedule(update)), "gnorm": round(float(gnorm), 5), "sec": round(sec, 2), "sec_ref": round(t_ref, 2),
               "sec_policy": round(sec - t_ref, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        append_jsonl(logd / "updates.jsonl", rec)
        if (update + 1) % 10 == 0 or update < 3:
            log(f"update {update + 1}/{total} reward {rec['reward_mean']:.3f} informative {rec['frac_groups_informative']:.2f} "
                f"kl {rec['kl_exact_mean']:.5f} loss {ul:.4f} gnorm {rec['gnorm']:.3f} {sec:.1f}s/update peak {rec['peak_gb']} GB")
            write_status(logd, a.tag, state="training", update=update + 1, total=total, loss=round(ul, 5), sec_per_update=round(sec, 2),
                         peak_gb=rec["peak_gb"], last_checkpoint="-")
        if not math.isfinite(ul):
            raise NonFinite(f"non-finite loss at update {update + 1}")
        done = update + 1
        if done % C.RESUME_EVERY == 0 or done == total:
            run_dir.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(run_dir / ".tmp_resume_adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            (run_dir / ".tmp_resume_adapter.safetensors").rename(run_dir / "resume_adapter.safetensors")
            mx.save_safetensors(str(run_dir / ".tmp_resume_opt.safetensors"), dict(tree_flatten(optimizer.state)))
            (run_dir / ".tmp_resume_opt.safetensors").rename(run_dir / "resume_opt.safetensors")
            (run_dir / ".tmp_resume.json").write_text(json.dumps({"update": done}))
            (run_dir / ".tmp_resume.json").rename(res_json)
            manifest["checkpoints"]["resume"] = {"update": done, "file": "resume_adapter.safetensors",
                                                 "sha256": guards.sha256_file(run_dir / "resume_adapter.safetensors"), "saved": now_iso()}
            manifest_f.write_text(json.dumps(manifest, indent=1))
        if a.stop_after_update and done >= a.stop_after_update:
            log(f"testing aid: exiting after update {done}")
            sys.exit(9)
    ck = run_dir / "final.safetensors"
    save_adapter(model, ck)
    digest = guards.sha256_file(ck)
    manifest["checkpoints"]["final"] = {"update": total, "file": ck.name, "sha256": digest, "saved": now_iso(), "code_commit": git["commit"]}
    manifest_f.write_text(json.dumps(manifest, indent=1))
    log(f"DONE: {total} updates; final adapter sha256 {digest}")
    if a.gen_probe:
        (logd / "gen_probe.json").write_text(json.dumps(gen_probe(model, processor, qs, a.gen_probe, log), indent=1) + "\n")
    write_status(logd, a.tag, state="training finished", update=total, total=total, last_checkpoint=f"final sha256 {digest[:16]}...")
    (run_dir / "DONE").write_text("done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=C.TAG)
    ap.add_argument("--n-updates", type=int, default=C.N_UPDATES, help="pipeline check only: fewer updates (first questions of the order)")
    ap.add_argument("--gen-probe", type=int, default=0, help="pipeline check only: time short-reasoning generation on N questions")
    ap.add_argument("--allow-dirty", action="store_true", help="tests only: permit uncommitted code")
    ap.add_argument("--stop-after-update", type=int, default=None)
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
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
