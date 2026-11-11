"""Step A-4B GPU views child (docs/plans/stepa_4b.md section 4). For each requested view, the frozen Step C 4B adapter's 4 letter logits per
dedup_v2 validation question are appended to <out-dir>/<view>.jsonl ({key, logits}; NO gold, no prediction), one flushed line per
question, so a crash loses at most one question and a rerun skips done keys. Views, prompts and images are Step A's
(cgt.p3s_stepa.views, unchanged); the model load is Step C's (scripts/stepc_eval.py). Labels are never read: the assistant turn fed to
the tokenizer is a fixed "A" (only positions BEFORE the answer token are run).
Does NOT take the GPU lock (the runner scripts/stepa_4b_run.py holds it). Refusals before any model load: adapter sha256 != frozen
(AdapterHashError), images outside ShotQA. Stops starting new questions at --stop-at (ISO with offset) and exits 0.
Per-view timing (aggregates) to <out-dir>/<view>.timing.json.
  HF_HUB_OFFLINE=1 nice -n 19 <venv>/bin/python scripts/stepa_4b_views.py --views base,shift1 [--n 20 | --keys K1,K2] [--out-dir D] [--stop-at ISO]"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.p3b_vlm import guards
from cgt.p3s_stepa.views import RUN_ORDER, VIEWS, build_user_text, prepare_image
from cgt.stepa_4b import config as K
from cgt.stepa_4b import core
from cgt.stepc import config as SC
from cgt.stepc.data import cached_image


def require_frozen_adapter(path: Path = K.ADAPTER) -> str:
    got = guards.sha256_file(path)
    if got != K.ADAPTER_SHA:
        raise guards.AdapterHashError(f"adapter {path} has sha256 {got[:16]}..., the frozen Step C 4B is {K.ADAPTER_SHA[:16]}...")
    return got


def load_questions(n: int | None, keys: list[str] | None = None) -> list[dict]:
    """dedup_v2 questions WITHOUT the answer field (the GPU loop never sees labels). Halves are not needed here."""
    rows = core.load_val_rows()
    qs = core.v2_split(rows, core.shotqa_film_of)["questions"]
    guards.require_samples_under_root(qs, SC.SHOTQA_ROOT)
    qs = [{k: q[k] for k in ("key", "id", "img_id", "attribute", "images", "messages", "options")} for q in qs]
    for q in qs:
        q["messages"] = [q["messages"][0]]  # user turn only; the answer turn is dropped here
    if keys:
        want = set(keys)
        qs = [q for q in qs if q["key"] in want]
        if len(qs) != len(want):
            raise SystemExit("unknown --keys")
    return qs[:n] if n else qs


def view_image(q: dict, view: str) -> str:
    v = VIEWS[view]
    src = q["images"][0]
    if v["image"] == "plain" and v["budget"] == 147_456:
        return cached_image(src)  # the exact Phase 3B budget-shrunk image Step C validated on
    sub = f"{v['image']}_{v['budget']}"
    stem = Path(src).stem + ".jpg"
    reuse = K.STEPA_IMG / sub / stem  # Step A's prepared copy (same recipe), read only
    if reuse.is_file() and reuse.stat().st_size > 0:
        return str(reuse)
    return str(prepare_image(src, K.DATA / "images" / sub / stem, v["image"], v["budget"]))


def view_sample(q: dict, view: str) -> dict:
    text = build_user_text(q["messages"][0]["content"], q["options"], VIEWS[view]["shift"])
    return {"images": [view_image(q, view)], "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "A"}]}


def load_model(sha: str):
    import mlx.core as mx
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.model_path import pinned_model_path
    from cgt.p3b_vlm.score import answer_token_ids, letter_logits

    model, processor = load(pinned_model_path(SC.REPO, SC.REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=SC.RANK, alpha=SC.ALPHA, dropout=SC.DROPOUT,
                           verbose=False)
    model.load_weights(list(mx.load(str(K.ADAPTER)).items()), strict=False)
    if guards.sha256_file(K.ADAPTER) != sha:
        raise guards.AdapterHashError("adapter changed between the check and the load")
    model.eval()

    class _Empty:
        def __len__(self):
            return 0

    return model, VisionDataset(_Empty(), cfg, processor, None, True), ids4, letter_logits, mx


def run(a) -> dict:
    views = sorted([v for v in a.views.split(",") if v], key=RUN_ORDER.index)
    bad = [v for v in views if v not in K.VIEWS]
    if bad:
        raise SystemExit(f"views not in the Step A-4B view set: {bad}")
    sha = require_frozen_adapter()
    qs = load_questions(a.n, a.keys.split(",") if a.keys else None)
    stop = dt.datetime.fromisoformat(a.stop_at) if a.stop_at else None
    out_dir = Path(a.out_dir or K.DATA / "views")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    model, ds, ids4, letter_logits, mx = load_model(sha)
    load_s = time.perf_counter() - t_load
    print(f"adapter {sha[:16]} verified; {len(qs)} questions; views {views}; load {load_s:.1f} s; "
          f"mlx env {[os.environ.get(k) for k in K.MLX_ENV]}", flush=True)
    status = {}
    for view in views:
        f_out = out_dir / f"{view}.jsonl"
        work = core.todo(qs, view, core.done_keys(f_out))
        print(f"view {view}: {len(work)} to do", flush=True)
        t0, n_done, stopped = time.perf_counter(), 0, False
        with open(f_out, "a") as f:
            for i, q in enumerate(work, 1):
                if stop and dt.datetime.now().astimezone() >= stop:
                    stopped = True
                    break
                four, _ = letter_logits(model, ds.process(view_sample(q, view)), ids4)
                f.write(json.dumps({"key": q["key"], "logits": [round(float(x), 4) for x in four]}) + "\n")
                f.flush()
                n_done = i
                if i % 250 == 0 or i == len(work):
                    print(f"{view} {i}/{len(work)} ({(time.perf_counter() - t0) / i:.3f} s/q)", flush=True)
        secs = time.perf_counter() - t0
        st = {"view": view, "n": n_done, "seconds": round(secs, 1), "sec_per_question": round(secs / max(n_done, 1), 4),
              "load_seconds": round(load_s, 1), "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "stopped_at_deadline": stopped,
              "mlx_env_seen": {k: os.environ.get(k) for k in K.MLX_ENV}}
        (out_dir / f"{view}.timing.json").write_text(json.dumps(st, indent=1) + "\n")
        status[view] = st
        if stopped:
            print(f"stop-at reached during {view}; exiting cleanly", flush=True)
            break
    if guards.sha256_file(K.ADAPTER) != sha:
        raise guards.AdapterHashError("adapter changed during the views run")
    print(json.dumps(status), flush=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--stop-at", default=None, help="ISO 8601 with offset")
    ap.add_argument("--n", type=int, default=None, help="first n dedup_v2 questions only (pipeline check)")
    ap.add_argument("--keys", default=None, help="comma-separated question keys only (pipeline check)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


if __name__ == "__main__":
    main()
