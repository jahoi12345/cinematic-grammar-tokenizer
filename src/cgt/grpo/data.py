"""GRPO pilot data: the seeded ShotQA TRAIN question subset (never val/test), read-only and hash-checked; images are the Phase 3B 1x
budget copies (cgt.stepc.data.cached_image, must exist). Validation reuses cgt.stepc.data.val_dedup_v2 / cgt.step3.data.val_image."""
import numpy as np

from cgt.grpo import config as C
from cgt.p3b_vlm import guards
from cgt.stepc.data import _rows, cached_image


def subset_indices(n_train: int = C.N_TRAIN, n: int = C.N_QUESTIONS, seed: int = C.SUBSET_SEED) -> np.ndarray:
    """train.jsonl line indices in pilot order: the first N_UPDATES * Q_PER_UPDATE of default_rng(26).permutation(51111) (prefix-stable:
    a shorter run uses the first n questions of the same order)."""
    return np.random.default_rng(seed).permutation(n_train)[:n]


def pilot_questions(n: int = C.N_QUESTIONS) -> list[dict]:
    rows = _rows(C.TRAIN_JSONL, C.TRAIN_SHA)
    if len(rows) != C.N_TRAIN:
        raise guards.DataRootViolation(f"train.jsonl has {len(rows)} samples, expected {C.N_TRAIN}")
    out = [rows[i] for i in subset_indices(len(rows), n)]
    guards.require_samples_under_root(out, C.SHOTQA_ROOT)
    return [{**r, "images": [cached_image(r["images"][0])]} for r in out]
