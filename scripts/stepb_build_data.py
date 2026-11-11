"""Step B data build + gate G2 (docs/plans/stepb_cinescale2.md sections 1 and 3). CPU only; no model.

Writes data/cache/stepb/{B0,B1,B2}.jsonl (training streams), shotqa_val_dedup.jsonl, cs2_val.jsonl, monitor_shotqa500.jsonl,
monitor_cs2_200.jsonl, the budget-shrunk CineScale2 images (data/cache/stepb/cs2_images_147456_32/) and the committed summary
results/stepb/data_summary.json (sha256 of every file). Every sample carries `shrunk` = the image actually fed to the model.
G2 asserts (fail closed, exit 5): CineScale2 images come only from split_v3 train/val and from under the unpacked root; no excluded
path; no CineScale2 val image in any training stream; no ShotQA validation image in any training stream; every ShotQA shrunk image
already exists in the Phase 3B cache (nothing is written there).
  PYTHONPATH= nice -n 19 <venv>/bin/python -I scripts/stepb_build_data.py"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cgt.cinescale2.labels import parse_relpath
from cgt.p3b_vlm.images import BUDGET_PX, MULTIPLE, shrink_to_budget
from cgt.stepb import data as D

SPLIT = ROOT / "results/cinescale2/split_v3.json"
CS2_ROOT = ROOT / "data/external/cinescale2/unpacked"
P3B = ROOT / "data/cache/p3b"
P3B_IMG = P3B / "images_147456_32"
OUT = ROOT / "data/cache/stepb"
CS2_IMG = OUT / "cs2_images_147456_32"
SUMMARY = ROOT / "results/stepb/data_summary.json"


def fail(msg: str) -> None:
    print(f"G2 FAIL: {msg}", flush=True)
    sys.exit(5)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cs2_shrunk(relpath: str) -> str:
    it = parse_relpath(relpath)
    dst = CS2_IMG / f"{it.task}__{it.label}__{Path(relpath).name.replace('.', '_')}.jpg"
    return str(shrink_to_budget(str(CS2_ROOT / relpath), dst, BUDGET_PX, MULTIPLE))


def shotqa_shrunk(s: dict) -> str:
    p = P3B_IMG / (Path(s["images"][0]).stem + ".jpg")
    if not p.exists():
        fail(f"ShotQA shrunk image missing from the Phase 3B cache: {p}")
    return str(p)


def write(name: str, rows: list[dict]) -> dict:
    p = OUT / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {"file": name, "n": len(rows), "sha256": sha(p)}


def main() -> None:
    split = json.loads(SPLIT.read_text())
    excluded = set().union(*map(set, split["excluded"].values()))
    cs2_val = list(split["val"])
    all_rel = [p.relative_to(CS2_ROOT).as_posix() for p in sorted(CS2_ROOT.rglob("*")) if p.is_file()]
    items = {r: parse_relpath(r) for r in all_rel}
    train_rel = sorted(r for r, it in items.items() if it and r not in excluded and r not in set(cs2_val))
    ang_train = [r for r in train_rel if items[r].task == "angle"]
    ang_val = [r for r in cs2_val if items[r].task == "angle"]
    if set(ang_train) & set(ang_val) or (set(ang_train) | set(ang_val)) & excluded:
        fail("CineScale2 train/val overlap or excluded image in train/val")
    by_class = {}
    for r in ang_train:
        by_class.setdefault(items[r].label, []).append(r)

    shotqa_train = [json.loads(x) for x in (P3B / "train.jsonl").read_text().splitlines()]
    shotqa_val_all = [json.loads(x) for x in (P3B / "val.jsonl").read_text().splitlines()]
    shotqa_val = D.dedup_first(shotqa_val_all)
    if {s["img_id"] for s in shotqa_train} & {s["img_id"] for s in shotqa_val_all}:
        fail("ShotQA validation image in the ShotQA training stream")
    OUT.mkdir(parents=True, exist_ok=True)

    def cs2_sample(r, k):
        s = D.cs2_question(r, str(CS2_ROOT / r), items[r].label, k)
        s["shrunk"] = cs2_shrunk(r)
        return s

    info = {"split": {"file": str(SPLIT.relative_to(ROOT)), "sha256": sha(SPLIT)},
            "cs2_angle_train": len(ang_train), "cs2_angle_train_per_class": {c: len(v) for c, v in sorted(by_class.items())},
            "cs2_angle_val": len(ang_val), "shotqa_train_stream": len(shotqa_train),
            "shotqa_val_lines": len(shotqa_val_all), "shotqa_val_dedup": len(shotqa_val), "files": {}}
    for cand in D.RUN_ORDER:
        n_cs2, n_qa = D.CANDIDATES[cand]
        qa = [{**shotqa_train[i], "source": "shotqa", "shrunk": shotqa_shrunk(shotqa_train[i])}
              for i in D.replay_subset(len(shotqa_train), n_qa, seed=1)]
        cs = [cs2_sample(r, k) for r, k in D.balanced_draws(by_class, n_cs2, seed=2)] if n_cs2 else []
        if {s["relpath"] for s in cs} & set(cs2_val):
            fail(f"{cand}: CineScale2 val image in the training stream")
        stream = D.mix(qa + cs, seed=0)
        info["files"][cand] = write(f"{cand}.jsonl", stream)
        info["files"][cand]["cs2_correct_class_counts"] = {c: sum(1 for s in cs if s["cs2_class"] == c) for c in D.CS2_CLASSES}
        info["files"][cand]["shotqa"] = len(qa)
        info["files"][cand]["cs2"] = len(cs)
    val_cs2 = [cs2_sample(r, 0) for r in ang_val]
    val_qa = [{**s, "source": "shotqa", "shrunk": shotqa_shrunk(s)} for s in shotqa_val]
    info["files"]["shotqa_val_dedup"] = write("shotqa_val_dedup.jsonl", val_qa)
    info["files"]["cs2_val"] = write("cs2_val.jsonl", val_cs2)
    info["files"]["cs2_val"]["correct_class_counts"] = {c: sum(1 for s in val_cs2 if s["cs2_class"] == c) for c in D.CS2_CLASSES}
    mon_qa = [val_qa[i] for i in sorted(np.random.default_rng(0).permutation(len(val_qa))[:500])]
    mon_cs2 = [val_cs2[i] for i in sorted(np.random.default_rng(0).permutation(len(val_cs2))[:200])]
    info["files"]["monitor_shotqa500"] = write("monitor_shotqa500.jsonl", mon_qa)
    info["files"]["monitor_cs2_200"] = write("monitor_cs2_200.jsonl", mon_cs2)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    print("G2 PASS")


if __name__ == "__main__":
    main()
