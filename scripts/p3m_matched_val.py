"""Matched protocol on the ShotQA film-held-out VALIDATION questions, for OUR frozen model only (docs/p3m_matched_protocol.md section 4).
Never reads ShotBench, CameraBench, gold or dev. The same questions as Phase 3B validation (data/cache/p3b/val.jsonl, 5,826) or a deterministic
subset (--n: the same seeded permutation as scripts/p3b_eval_val.py). Labels are used only to score, as scripts/p3b_eval_val.py does.
Refusals before any model load: tag must be matched_v1 and the adapter's recomputed SHA-256 must equal the frozen manifest (exit 7); memory free
>= 25% (exit 12); the GPU lock /tmp/cgt_gpu.lock must be free (taken atomically, never waits, never steals; exit 11).
Resumable: per-question outputs (text, parsed letter, method; no gold) are appended to <cache-dir>/<variant>.jsonl and skipped on restart.
At the end the aggregate (and the letter-logit comparison on the SAME questions) goes to results/p3m_matched/val_stage1_<variant>.json.
Usage: uv run python scripts/p3m_matched_val.py --variant reasoning [--n 1000] [--data-root PATH]"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3m_matched import registry
from cgt.p3m_matched.protocol import SYSTEM_PROMPTS, protocol_sha256, question_from_shotqa_turn

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_PERCENT = 25
LETTER_LOGIT_PRED_SHA256 = "cebea813002eb24971d6ae652b38a90a0f7fc7105ceed92d9a0da3c64770ece6"  # frozen_manifest predictions_sha256["100"]
LETTER_LOGIT_PRED_REL = "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/val_predictions/ckpt_100.predictions.jsonl"


class Refused(guards.FailClosed):
    pass


class MemoryLow(Refused):
    exit_code = 12


class GpuBusy(Refused):
    exit_code = 11


def free_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.split(":")[1].strip().rstrip("%"))
    return 0


def memory_guard(reader=free_percent) -> int:
    free = reader()
    if free < MIN_FREE_PERCENT:
        raise MemoryLow(f"memory guard: {free}% free < {MIN_FREE_PERCENT}%; nothing started")
    return free


def require_val_tag(tag: str, data_root: Path) -> dict:
    """Only our frozen model; the adapter hash is recomputed (AdapterHashError, exit 7)."""
    if tag != "matched_v1":
        raise guards.TagNotBound(f"validation mode is for our frozen model (matched_v1) only, not {tag!r}")
    e = registry.require_identity(tag, data_root)
    if e["identity_sha256"] != registry.FROZEN_ADAPTER_SHA256:
        raise guards.AdapterHashError("registry identity is not the frozen Phase 3B adapter")
    return e


def load_val_questions(data_root: Path, n: int | None, shotqa_root: Path) -> list[dict]:
    """Questions in val.jsonl order. `key` = "<line index>:<id>": val.jsonl has 64 ids that occur twice (duplicate sft.json entries of the same image),
    so the id alone is NOT unique; every per-question join (resume, letter-logit comparison) uses the key."""
    val = [json.loads(x) for x in (Path(data_root) / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    pos = list(range(len(val)))
    if n:
        pos = sorted(np.random.default_rng(0).permutation(len(val))[:n])
    val = [(i, val[i]) for i in pos]
    guards.require_samples_under_root([s for _, s in val], shotqa_root)
    return [{"key": f"{i}:{s['id']}", "id": s["id"], "attribute": s["attribute"], "image": s["images"][0], "question": question_from_shotqa_turn(s["messages"][0]["content"]),
             "options": {k: s["options"][k] for k in sorted(s["options"])}, "answer": s["answer"]} for i, s in val]


def letter_logit_preds(data_root: Path) -> dict[str, str]:
    p = Path(data_root) / LETTER_LOGIT_PRED_REL
    if hashlib.sha256(p.read_bytes()).hexdigest() != LETTER_LOGIT_PRED_SHA256:
        raise guards.AdapterHashError("letter-logit validation predictions do not match the frozen manifest hash")
    val = [json.loads(x)["id"] for x in (Path(data_root) / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    preds = [json.loads(x) for x in p.read_text().splitlines()]
    if len(preds) != len(val) or any(r["id"] != i for r, i in zip(preds, val, strict=True)):
        raise guards.FailClosed("letter-logit predictions are not aligned with val.jsonl line order")
    return {f"{n}:{r['id']}": r["pred"] for n, r in enumerate(preds)}  # file is in val.jsonl line order (checked above)


def compare(results: list[dict], ll: dict[str, str]) -> dict:
    """Letter-logit accuracy on the SAME questions and per-question agreement (matched letter == letter-logit letter)."""
    from cgt.p3m_matched.evaluate import ATTRS
    out = {}
    for a in ATTRS:
        r = [x for x in results if x["attribute"] == a]
        if not r:
            continue
        out[a] = {"n": len(r), "letter_logit_acc": float(np.mean([ll[x["key"]] == x["gold"] for x in r])),
                  "agreement": float(np.mean([x["letter"] == ll[x["key"]] for x in r]))}
    vals = list(out.values())
    return {"per_attribute": out, "letter_logit_mean_acc": float(np.mean([v["letter_logit_acc"] for v in vals])),
            "agreement_all": float(np.mean([x["letter"] == ll[x["key"]] for x in results]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="matched_v1")
    ap.add_argument("--variant", required=True, choices=sorted(SYSTEM_PROMPTS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--data-root", default=str(ROOT), help="checkout holding data/cache/p3b (val.jsonl, image cache, run dir)")
    ap.add_argument("--shotqa-root", default=str(ROOT / "data/external/shotqa"))
    ap.add_argument("--cache-dir", default=None, help="default <data-root>/data/cache/p3m_matched/val")
    ap.add_argument("--out-dir", default=str(ROOT / "results/p3m_matched"))
    ap.add_argument("--lock", default=None, help="GPU lock path (default /tmp/cgt_gpu.lock)")
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


def run(a, engine_factory=None, mem_reader=free_percent) -> dict:
    data_root = Path(a.data_root)
    e = require_val_tag(a.tag, data_root)  # before any data or model access
    free = memory_guard(mem_reader)
    from cgt.pipeline.gpulock import GpuLock, LockHeld
    try:
        lock = GpuLock(f"p3m_matched_val {a.variant}", Path(a.lock) if a.lock else None).acquire()
    except LockHeld as exc:
        raise GpuBusy(str(exc)) from exc
    try:
        return _run_locked(a, e, data_root, free, engine_factory)
    finally:
        lock.release()


def _run_locked(a, e, data_root: Path, free: int, engine_factory) -> dict:
    from cgt.p3b_vlm.images import shrunk_path
    from cgt.p3m_matched.evaluate import run_questions, summarize
    qs = load_val_questions(data_root, a.n, Path(a.shotqa_root))
    cache = Path(a.cache_dir or data_root / "data/cache/p3m_matched/val")
    try:
        for q in qs:
            q["image"] = shrunk_path(q["image"], data_root / "data/cache/p3b/images_147456_32")
    except (OSError, ValueError) as exc:
        raise guards.UnreadableImage(str(exc)) from exc
    guards.require_images_readable([q["image"] for q in qs])
    raw = cache / f"{a.variant}{'' if not a.n else f'_n{a.n}'}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if "key" not in r:
                raise guards.FailClosed(f"{raw} has lines without a unique key (written before the duplicate-id fix): move it away")
            done[r["key"]] = r
    todo = [q for q in qs if q["key"] not in done]
    print(f"[{now_iso()}] {len(qs)} questions, {len(done)} already done, variant {a.variant}, protocol {protocol_sha256(a.variant)[:16]}, "
          f"memory free {free}%", flush=True)
    t0 = time.perf_counter()
    if todo:
        if engine_factory is None:
            from cgt.p3m_matched.engine import MlxEngine, load_mlx_model

            def engine_factory(entry):
                return MlxEngine(*load_mlx_model(entry))
        engine = engine_factory(e)
        with open(raw, "a") as f:
            def progress(i, total):
                if i % 100 == 0 or i == total:
                    print(f"[{now_iso()}] progress {i}/{total} ({(time.perf_counter() - t0) / i:.2f} s/question)", flush=True)
            for q in todo:  # one at a time so every answer is on disk before the next (resume after a Metal watchdog kill)
                r = run_questions(engine, [q], a.variant)[0]
                f.write(json.dumps({"key": q["key"], **{k: r[k] for k in ("id", "attribute", "text", "letter", "method", "n_choices", "generation_tokens")}}) + "\n")
                f.flush()
                done[q["key"]] = r
                progress(len(done) - (len(qs) - len(todo)), len(todo))
    res = [{**done[q["key"]], "key": q["key"], "gold": q["answer"], "correct": done[q["key"]]["letter"] == q["answer"]} for q in qs]
    summ = summarize(res)
    cmp_ = compare(res, letter_logit_preds(data_root))
    out = {"time": now_iso(), "tag": a.tag, "variant": a.variant, "protocol_sha256": protocol_sha256(a.variant), "adapter_sha256": e["identity_sha256"],
           "questions": len(res), "subset_n": a.n, "seconds_this_session": time.perf_counter() - t0, "matched": summ, "letter_logit_same_questions": cmp_,
           "raw_outputs_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
           "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()}
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"val_stage1_{a.variant}{'' if not a.n else f'_n{a.n}'}.json").write_text(json.dumps(out, indent=1))
    print(f"matched mean {100 * summ['mean_acc']:.2f}% (invalid {100 * summ['invalid_rate']:.2f}%); letter-logit same questions "
          f"{100 * cmp_['letter_logit_mean_acc']:.2f}%; agreement {100 * cmp_['agreement_all']:.2f}%", flush=True)
    return out


if __name__ == "__main__":
    sys.exit(main())
