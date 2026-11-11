"""Shrink every ShotQA train+val image of the Phase 3B jsonl files to the Qwen2.5-VL equal-token budget (112,896 px, sides multiples
of 28; cgt.p3b_vlm.images.shrink_to_budget, same recipe as Phase 3B). CPU only. Output: data/cache/base_gen/images_112896_28 (gitignored).
  nice -n 10 <venv>/bin/python scripts/base_gen_prep_images.py"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.base_gen import config as C
from cgt.p3b_vlm.images import shrunk_path
from cgt.stepc import config as SC


def _one(src: str) -> str:
    return shrunk_path(src, C.IMG_CACHE_3B, C.BUDGET_PX_3B, C.MULTIPLE_3B)


def main() -> None:
    srcs = sorted({json.loads(x)["images"][0] for f in (SC.TRAIN_JSONL, SC.VAL_JSONL) for x in Path(f).read_text().splitlines()})
    with ProcessPoolExecutor(8) as ex:
        n = len(list(ex.map(_one, srcs, chunksize=64)))
    print(f"shrunk/cached {n} images into {C.IMG_CACHE_3B}", flush=True)


if __name__ == "__main__":
    main()
