"""Pre-shrink the images of a chat JSONL to the 147,456-pixel budget (CPU only, parallel). Usage:
uv run python scripts/p3b_prep_images.py --jsonl data/cache/p3b/train.jsonl [--n 5000]"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cgt.p3b_vlm.images import shrunk_path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/p3b/images_147456_32"


def _one(src: str) -> str:
    return shrunk_path(src, CACHE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.jsonl).read_text().splitlines()]
    if a.n:
        import numpy as np
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[: a.n]]
    srcs = sorted({r["images"][0] for r in rows})
    with ProcessPoolExecutor(a.workers) as ex:
        done = list(ex.map(_one, srcs, chunksize=64))
    print(f"shrunk/cached {len(done)} images into {CACHE}")


if __name__ == "__main__":
    main()
