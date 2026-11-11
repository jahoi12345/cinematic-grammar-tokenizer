"""Step A static track: GPU views (docs/acc_step_a_static_plan.md section 4). For each requested view, the frozen Phase 3B model's 4 letter
logits per deduplicated validation question are appended to data/cache/p3s_stepa/views/<view>.jsonl ({key, logits}; NO gold, no prediction).
Labels are never read here: the assistant turn fed to the tokenizer is a fixed "A" (only positions BEFORE the answer token are run).
Refusals before any model load: adapter SHA-256 != frozen (exit 7), memory free < 25% (exit 12), GPU lock held (exit 11), an image outside
ShotQA (exit 5). Resumable (done keys skipped); stops starting new questions at --stop-at (local time, ISO) and exits 0.
Usage: caffeinate -i uv run python scripts/p3s_stepa_views.py --views base,shift1,... [--stop-at 2026-10-17T08:15] [--n 40]"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3s_stepa.split import dedup_first
from cgt.p3s_stepa.views import RUN_ORDER, VIEWS, applies, build_user_text, prepare_image

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ADAPTER_SHA256 = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"
ADAPTER_REL = "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/ckpt_100.safetensors"
BASE_IMG = "data/cache/p3b/images_147456_32"
MIN_FREE_PERCENT = 25


class MemoryLow(guards.FailClosed):
    exit_code = 12


class GpuBusy(guards.FailClosed):
    exit_code = 11


def free_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.split(":")[1].strip().rstrip("%"))
    return 0


def require_frozen_adapter(path: Path) -> str:
    got = guards.sha256_file(path)
    if got != FROZEN_ADAPTER_SHA256:
        raise guards.AdapterHashError(f"adapter {path} has SHA-256 {got}, the frozen model is {FROZEN_ADAPTER_SHA256}")
    return got


def load_questions(root: Path, n: int | None) -> list[dict]:
    """Deduplicated validation questions WITHOUT the answer field (the GPU loop never sees labels)."""
    raw = [json.loads(x) for x in (root / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    qs = dedup_first(raw)
    guards.require_samples_under_root(qs)
    qs = [{k: q[k] for k in ("key", "id", "img_id", "attribute", "images", "messages", "options")} for q in qs]
    for q in qs:
        q["messages"] = [q["messages"][0]]  # user turn only; the answer turn is dropped here
    return qs[:n] if n else qs


def view_image(root: Path, q: dict, view: str) -> str:
    v = VIEWS[view]
    src = q["images"][0]
    if v["image"] == "plain" and v["budget"] == 147_456:
        p = root / BASE_IMG / (Path(src).stem + ".jpg")  # the exact cached image Phase 3B validated on
        if not p.exists():
            raise guards.UnreadableImage(f"{q['id']}: base cached image missing")
        return str(p)
    return str(prepare_image(src, root / "data/cache/p3s_stepa/images" / f"{v['image']}_{v['budget']}" / (Path(src).stem + ".jpg"),
                             v["image"], v["budget"]))


def view_sample(root: Path, q: dict, view: str) -> dict:
    text = build_user_text(q["messages"][0]["content"], q["options"], VIEWS[view]["shift"])
    return {"images": [view_image(root, q, view)],
            "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "A"}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--n", type=int, default=None, help="first n questions only (smoke)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lock", default=None)
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


def run(a, mem_reader=free_percent, model_loader=None, root: Path = ROOT) -> dict:
    views = [v for v in a.views.split(",") if v]
    bad = [v for v in views if v not in VIEWS]
    if bad:
        raise SystemExit(f"unknown views {bad}")
    views = sorted(views, key=RUN_ORDER.index)
    sha = require_frozen_adapter(root / ADAPTER_REL)  # before any data or model access
    free = mem_reader()
    if free < MIN_FREE_PERCENT:
        raise MemoryLow(f"memory guard: {free}% free < {MIN_FREE_PERCENT}%; nothing started")
    qs = load_questions(root, a.n)
    stop = datetime.fromisoformat(a.stop_at).astimezone() if a.stop_at else None  # naive = local time
    from cgt.pipeline.gpulock import GpuLock, LockHeld
    try:
        lock = GpuLock(f"p3s_stepa_views {','.join(views)}", Path(a.lock) if a.lock else None).acquire()
    except LockHeld as exc:
        raise GpuBusy(str(exc)) from exc
    out_dir = Path(a.out_dir or root / "data/cache/p3s_stepa/views")
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    try:
        print(f"[{now_iso()}] adapter {sha[:16]} verified; memory free {free}%; {len(qs)} questions; views {views}", flush=True)
        model, ds, ids4, letter_logits = (model_loader or _load_model)(root, sha)
        for view in views:
            f_out = out_dir / f"{view}{'' if not a.n else f'_n{a.n}'}.jsonl"
            done = {json.loads(x)["key"] for x in f_out.read_text().splitlines()} if f_out.exists() else set()
            todo = [q for q in qs if applies(view, q["attribute"]) and q["key"] not in done]
            print(f"[{now_iso()}] view {view}: {len(done)} done, {len(todo)} to do", flush=True)
            t0 = time.perf_counter()
            stopped = False
            with open(f_out, "a") as f:
                for i, q in enumerate(todo, 1):
                    if stop and datetime.now().astimezone() >= stop:
                        stopped = True
                        break
                    item = ds.process(view_sample(root, q, view))
                    four, _ = letter_logits(model, item, ids4)
                    f.write(json.dumps({"key": q["key"], "logits": [round(float(x), 4) for x in four]}) + "\n")
                    f.flush()
                    if i % 200 == 0 or i == len(todo):
                        print(f"[{now_iso()}] {view} {i}/{len(todo)} ({(time.perf_counter() - t0) / i:.3f} s/q)", flush=True)
            status[view] = {"stopped_at_deadline": stopped, "seconds": round(time.perf_counter() - t0, 1)}
            if stopped:
                print(f"[{now_iso()}] stop-at reached during {view}; exiting cleanly", flush=True)
                break
    finally:
        lock.release()
    print(json.dumps(status), flush=True)
    return status


def _load_model(root: Path, sha: str):
    import mlx.core as mx
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.model_path import pinned_model_path
    from cgt.p3b_vlm.score import answer_token_ids, letter_logits

    model, processor = load(pinned_model_path())
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
    model.load_weights(list(mx.load(str(root / ADAPTER_REL)).items()), strict=False)
    if guards.sha256_file(root / ADAPTER_REL) != sha:
        raise guards.AdapterHashError("adapter changed between the check and the load")
    model.eval()

    class _Empty:
        def __len__(self):
            return 0

    return model, VisionDataset(_Empty(), cfg, processor, None, True), ids4, letter_logits


if __name__ == "__main__":
    sys.exit(main())
