"""Phase 3C step 2: score the candidate full-vocabulary methods on the ShotQA film-split VALIDATION subset (inference only; ShotQA validation only).
Resumable: results are appended to data/cache/p3c/select/<method>.jsonl. Usage:
  uv run python scripts/p3c_select_method.py run [--methods A2m,...] [--limit N]
  uv run python scripts/p3c_select_method.py report"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import shrunk_path
from cgt.p3c_tokens import resume, selection
from cgt.p3c_tokens.methods import METHODS, score_attribute

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cache/p3c/select"
IMG = ROOT / "data/cache/p3c/images_val"
ADAPTER = ROOT / "data/cache/p3c/ckpt_100.safetensors"


def load_done(path: Path) -> dict:
    """(id, attribute) -> record (see cgt.p3c_tokens.resume)."""
    return resume.load_done(path)


def cmd_run(a) -> None:
    from cgt.p3c_tokens.engine import Engine
    guards.require_clean_tree(["src/cgt/p3c_tokens", "scripts/p3c_select_method.py", "docs/phase3c_plan.md"])
    samples = selection.validation_subset()  # positions below are positions in this full registered subset (stable across --limit runs)
    indexed = list(enumerate(samples))
    if a.limit:
        indexed = indexed[:: max(1, len(indexed) // a.limit)][: a.limit]
    guards.require_samples_under_root([s for _, s in indexed])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subset_ids.json").write_text(json.dumps([s["id"] for s in samples]))
    eng = Engine(ADAPTER)
    methods = ["native"] + (a.methods.split(",") if a.methods else METHODS)
    done = {m: load_done(OUT / f"{m}.jsonl") for m in methods if m != "native"}
    native_done = resume.native_done_positions(OUT / "native.jsonl", samples)
    by_img: dict[str, list[tuple[int, dict]]] = {}
    for j, s in indexed:
        by_img.setdefault(s["img_id"], []).append((j, s))
    files = {m: open(OUT / f"{m}.jsonl", "a") for m in methods}  # noqa: SIM115
    t0, n_img = time.perf_counter(), 0
    for items in by_img.values():
        todo = [(m, j, s) for m in methods for j, s in items
                if (j not in native_done if m == "native" else (s["id"], s["attribute"]) not in done[m])]
        if not todo:
            continue
        st = eng.image_state(shrunk_path(items[0][1]["images"][0], IMG))
        for m, j, s in todo:
            p0 = eng.passes
            if m == "native":
                text = s["messages"][0]["content"]
                lg = eng.letter_logits(st, text, 4)
                rec = {"id": s["id"], "attr": s["attribute"], "pos": j, "q": resume.question_hash(text), "logits": [float(x) for x in lg]}
                native_done.add(j)
            else:
                sc, _ = score_attribute(eng, st, s["attribute"], m)
                rec = {"id": s["id"], "attr": s["attribute"], "s": [float(x) for x in sc]}
                done[m][(s["id"], s["attribute"])] = rec
            rec["passes"] = eng.passes - p0
            files[m].write(json.dumps(rec) + "\n")
            files[m].flush()
        n_img += 1
        if n_img % 25 == 0:
            print(f"{n_img}/{len(by_img)} images, {time.perf_counter() - t0:.0f}s, passes {eng.passes}", flush=True)
    print(f"done: {time.perf_counter() - t0:.0f}s, passes {eng.passes}", flush=True)


def cmd_equivalence(a) -> None:
    """Cached-prefix scoring vs the stock full forward pass on the same prompts (native 4-option questions)."""
    from cgt.p3c_tokens.engine import Engine
    samples = selection.validation_subset()[:: 1498 // a.n][: a.n]
    eng = Engine(ADAPTER)
    diffs, agree = [], 0
    for s in samples:
        path = shrunk_path(s["images"][0], IMG)
        st = eng.image_state(path)
        text = s["messages"][0]["content"]
        new = eng.letter_logits(st, text, 4)
        old = eng.stock_letter_logits(path, text)
        diffs.append(float(np.abs(new - old).max()))
        agree += int(np.argmax(new) == np.argmax(old))
    rep = {"n": len(samples), "max_abs_logit_diff": max(diffs), "mean_abs_logit_diff": float(np.mean(diffs)), "argmax_agree": agree}
    print(json.dumps(rep))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "equivalence.json").write_text(json.dumps(rep, indent=1))


def cmd_native_fix(a) -> None:
    """The 4 subset ids that sft.json asks twice (same image id and turn) share a native record key; recompute their native 4-option logits by subset position."""
    from collections import Counter

    from cgt.p3c_tokens.engine import Engine
    samples = selection.validation_subset()
    dup = {i for i, c in Counter(s["id"] for s in samples).items() if c > 1}
    eng = Engine(ADAPTER)
    with open(OUT / "native_collisions.jsonl", "w") as f:
        for j, s in enumerate(samples):
            if s["id"] in dup:
                st = eng.image_state(shrunk_path(s["images"][0], IMG))
                lg = eng.letter_logits(st, s["messages"][0]["content"], 4)
                f.write(json.dumps({"pos": j, "id": s["id"], "attr": s["attribute"], "logits": [float(x) for x in lg]}) + "\n")
    print("native collision records written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--methods", default=None)
    r.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("equivalence")
    e.add_argument("--n", type=int, default=40)
    sub.add_parser("native-fix")
    a = ap.parse_args()
    {"run": cmd_run, "equivalence": cmd_equivalence, "native-fix": cmd_native_fix}[a.cmd](a)
