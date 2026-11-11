"""Step C data access: read-only, hash-checked, fail closed. Images are the Phase 3B budget-shrunk copies, which must already exist
(nothing is shrunk or written into the Phase 3B cache)."""
import json
from pathlib import Path

import numpy as np

from cgt.eval.val_dedup import dedup_v2_indices
from cgt.p3b_vlm import guards
from cgt.stepc import config as C


def cached_image(src: str, cache: Path = C.IMG_CACHE) -> str:
    p = Path(cache) / (Path(src).stem + ".jpg")
    if not p.is_file() or p.stat().st_size == 0:
        raise guards.UnreadableImage(f"shrunk image missing from the Phase 3B cache: {p}")
    return str(p)


def _rows(path: Path, sha: str) -> list[dict]:
    if guards.sha256_file(path) != sha:
        raise guards.DataRootViolation(f"{path} does not match its recorded sha256 {sha[:12]}...")
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def train_stream(path: Path = C.TRAIN_JSONL, sha: str = C.TRAIN_SHA) -> list[dict]:
    """The Phase 3B train stream in the Phase 3B order: np.random.default_rng(0).permutation over train.jsonl lines."""
    rows = _rows(path, sha)
    if len(rows) != C.N_TRAIN:
        raise guards.DataRootViolation(f"train.jsonl has {len(rows)} samples, expected {C.N_TRAIN}")
    guards.require_samples_under_root(rows, C.SHOTQA_ROOT)
    order = np.random.default_rng(C.ORDER_SEED).permutation(len(rows))
    return [rows[i] for i in order]


def val_dedup_v2(path: Path = C.VAL_JSONL, sha: str = C.VAL_SHA) -> tuple[list[int], list[dict]]:
    """(line indices, rows) of the dedup_v2 ShotQA film-held-out validation set (5,822 questions)."""
    rows = _rows(path, sha)
    idx = dedup_v2_indices(rows)
    if len(idx) != C.N_VAL_V2:
        raise guards.DataRootViolation(f"dedup_v2 has {len(idx)} questions, expected {C.N_VAL_V2}")
    keep = [rows[i] for i in idx]
    guards.require_samples_under_root(keep, C.SHOTQA_ROOT)
    return idx, keep
