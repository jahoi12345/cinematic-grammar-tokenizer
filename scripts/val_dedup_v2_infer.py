"""Inference on the 60 ShotQA validation rows that dedup v2 adds (docs/plans/val_dedup_v2.md, Amendment 1). Descriptive only.

(1) Step A views (base, shift1-3, hflip, crop80) with the frozen adapter, through the unchanged Step A code (scripts/p3s_stepa_views.py
    `_load_model` / `view_sample`; label-free prompts with a fixed "A" turn). (2) Step B B0 100% and B1 100% adapters through the
    stepb_eval code path (same model load / LoRA config / cgt.p3b_vlm.score.evaluate_samples), on the same budget-shrunk images.
    Plus a 5-question parity check per Step B adapter on already-scored rows.
Outputs (gitignored, per question): data/cache/val_dedup_v2/{views_<view>.jsonl, stepb_<cand>.jsonl, stepb_<cand>_parity.jsonl, run.json}.
Guards: GPU only under /tmp/cgt_gpu.lock (GpuLock; waits while held, never removes it), memory free >= 25% before start and before every
model load, a 45-minute wall-clock cap checked before every question. Never reads ShotBench, CameraBench, gold or dev.
Usage: nice -n 19 caffeinate -i <venv>/bin/python scripts/val_dedup_v2_infer.py"""
import gc
import json
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.eval.val_dedup import dedup_v2_indices
from cgt.p3b_vlm import guards
from cgt.p3s_stepa.split import dedup_first
from cgt.p3s_stepa.views import RUN_ORDER, applies
from cgt.pipeline.gpulock import GpuLock, LockHeld

CAP_S = 45 * 60
MIN_FREE = 25
LOCK_WAIT_S = 30 * 60
VAL = ROOT / "data/cache/p3b/val.jsonl"
OUT = ROOT / "data/cache/val_dedup_v2"
STEPA_VIEWS = ["base", "shift1", "shift2", "shift3", "hflip", "crop80"]
STEPB = {"B0": ROOT / "data/cache/stepb/runs/stepb_B0/ckpt_100.safetensors", "B1": ROOT / "data/cache/stepb/runs/stepb_B1/ckpt_100.safetensors"}
T0 = time.monotonic()


class CapReached(guards.FailClosed):
    exit_code = 14


def load_script(name: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_cap() -> None:
    if time.monotonic() - T0 > CAP_S:
        raise CapReached(f"45-minute cap reached after {time.monotonic() - T0:.0f} s")


def check_mem(free_percent) -> int:
    f = free_percent()
    if f < MIN_FREE:
        raise guards.FailClosed(f"memory guard: {f}% free < {MIN_FREE}%")
    return f


def added_rows() -> tuple[list[dict], list[dict], dict[int, int]]:
    rows = [json.loads(x) for x in VAL.read_text().splitlines()]
    first = {r["line"] for r in dedup_first(rows)}
    added = [n for n in dedup_v2_indices(rows) if n not in first]
    assert len(added) == 60
    first_line = {}
    for n, r in enumerate(rows):
        first_line.setdefault(r["id"], n)
    return rows, [{**rows[n], "line": n, "key": f"{n}:{rows[n]['id']}"} for n in added], {n: first_line[rows[n]["id"]] for n in added}


def main() -> None:
    views_mod = load_script("p3s_stepa_views")
    stepb_mod = load_script("stepb_eval")
    rows, add, partner = added_rows()
    guards.require_samples_under_root(add)
    sha_frozen = views_mod.require_frozen_adapter(ROOT / views_mod.ADAPTER_REL)
    for cand, p in STEPB.items():
        guards.require_registered_hash(ROOT / f"results/stepb/runs/stepb_{cand}/manifest.json", p)
    log = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "added_rows": len(add)}
    waited = 0
    while True:
        try:
            lock = GpuLock("val_dedup_v2_infer").acquire()
            break
        except LockHeld as e:
            if waited >= LOCK_WAIT_S:
                raise SystemExit(f"GPU lock still held after {waited} s: {e}") from e
            print(f"lock held, waiting: {e}", flush=True)
            time.sleep(60)
            waited += 60
    t_gpu = time.monotonic()
    try:
        log["mem_free_start"] = check_mem(views_mod.free_percent)
        # (1) Step A views, frozen adapter
        model, ds, ids4, letter_logits = views_mod._load_model(ROOT, sha_frozen)
        qs = [{k: q[k] for k in ("key", "id", "img_id", "attribute", "images", "messages", "options")} for q in add]
        for q in qs:
            q["messages"] = [q["messages"][0]]          # label-free, exactly as the Step A runner
        for view in sorted(STEPA_VIEWS, key=RUN_ORDER.index):
            out = []
            for q in qs:
                if not applies(view, q["attribute"]):
                    continue
                check_cap()
                four, _ = letter_logits(model, ds.process(views_mod.view_sample(ROOT, q, view)), ids4)
                out.append({"key": q["key"], "logits": [round(float(x), 4) for x in four]})
            (OUT / f"views_{view}.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
            print(f"view {view}: {len(out)} rows ({time.monotonic() - t_gpu:.0f} s)", flush=True)
        del model, ds
        gc.collect()
        # (2) Step B adapters, stepb_eval code path
        import mlx.core as mx
        from mlx_vlm.trainer.datasets import VisionDataset
        from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
        from mlx_vlm.utils import load

        from cgt.p3b_vlm.model_path import pinned_model_path
        from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples
        val_dedup = {s["id"]: s for s in stepb_mod.load_set("shotqa_val_dedup")}   # sha-checked against the data summary
        samples = [{**q, "images": [val_dedup[q["id"]]["shrunk"]]} for q in add]  # same image as the first occurrence
        for q in add:
            assert val_dedup[q["id"]]["img_id"] == q["img_id"]
        parity_ids = [rows[partner[q["line"]]]["id"] for q in add[:5]]
        for cand, p in STEPB.items():
            mx.clear_cache()
            log[f"mem_free_before_{cand}"] = check_mem(views_mod.free_percent)
            check_cap()
            sha = guards.require_registered_hash(ROOT / f"results/stepb/runs/stepb_{cand}/manifest.json", p)
            model, processor = load(pinned_model_path(stepb_mod.REPO, stepb_mod.REVISION))
            cfg = model.config.__dict__
            ids4 = answer_token_ids(processor)
            model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
            model.load_weights(list(mx.load(str(p)).items()), strict=False)
            model.eval()
            ds = VisionDataset(stepb_mod.ListView([]), cfg, processor, None, True)
            res = []
            for s in samples:
                check_cap()
                res += evaluate_samples(model, ds, [s], ids4)
            par = evaluate_samples(model, ds, [{**val_dedup[i], "images": [val_dedup[i]["shrunk"]]} for i in parity_ids], ids4)
            if guards.sha256_file(p) != sha:
                raise guards.AdapterHashError("adapter changed during evaluation")
            (OUT / f"stepb_{cand}.jsonl").write_text("\n".join(json.dumps({**r, "line": q["line"]}) for r, q in zip(res, add, strict=True)) + "\n")
            (OUT / f"stepb_{cand}_parity.jsonl").write_text("\n".join(json.dumps(r) for r in par) + "\n")
            log[f"{cand}_adapter_sha256"] = sha
            print(f"{cand}: {len(res)} rows ({time.monotonic() - t_gpu:.0f} s)", flush=True)
            del model, ds
            gc.collect()
    finally:
        lock.release()
        log["gpu_seconds"] = round(time.monotonic() - t_gpu, 1)
        log["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (OUT / "run.json").write_text(json.dumps(log, indent=1) + "\n")
        print(json.dumps(log), flush=True)


if __name__ == "__main__":
    try:
        main()
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e
