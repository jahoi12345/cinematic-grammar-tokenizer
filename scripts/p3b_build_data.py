"""Write Phase 3B chat JSONL (ShotQA film-split train / val) to data/cache/p3b/ (gitignored; ShotQA is CC-BY-NC-ND).
Usage: uv run python scripts/p3b_build_data.py [--bench-n 400]"""
import argparse
import collections
import json
from pathlib import Path

from cgt.p3b_vlm import data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-n", type=int, default=400)
    a = ap.parse_args()
    out = ROOT / "data/cache/p3b"
    train, val, info = data.film_split_ids()
    tr, va = data.load_chat_samples("train"), data.load_chat_samples("val")
    data.write_jsonl(tr, out / "train.jsonl")
    data.write_jsonl(va, out / "val.jsonl")
    data.write_jsonl(data.subset(tr, a.bench_n, 0), out / "bench_subset.jsonl")
    summ = {"film_split": {k: v for k, v in info.items() if k != "val_films"}, "train_images": len(train), "val_images": len(val),
            "train_samples": len(tr), "val_samples": len(va),
            "train_by_attribute": dict(collections.Counter(s["attribute"] for s in tr)),
            "no_overlap_train_val_images": not (train & val), "bench_subset": a.bench_n}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
