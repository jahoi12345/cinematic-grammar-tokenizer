"""Step 3 data: the seeded replay subset of the ShotQA TRAIN stream, per-candidate image paths, the dedup_v2 validation set and the
fixed robustness-proxy subset. Read-only on the Phase 3B cache and ShotQA originals; generated images go to data/cache/step3/."""
from pathlib import Path

import numpy as np
from PIL import Image

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import shrink_to_budget
from cgt.step3 import augment
from cgt.step3 import config as C
from cgt.stepb.select import ATTRS
from cgt.stepc.data import _rows, cached_image, val_dedup_v2

__all__ = ["proxy_image", "proxy_subset", "replay_indices", "replay_stream", "train_image", "val_dedup_v2", "val_image"]


def replay_indices(n_train: int = C.N_TRAIN, n: int = C.N_SAMPLES, seed: int = C.REPLAY_SEED) -> np.ndarray:
    """Line indices of train.jsonl used, in training order: default_rng(3).choice(51111, 9600, replace=False)."""
    return np.random.default_rng(seed).choice(n_train, n, replace=False)


def replay_stream(path: Path = C.TRAIN_JSONL, sha: str = C.TRAIN_SHA) -> list[dict]:
    rows = _rows(path, sha)
    if len(rows) != C.N_TRAIN:
        raise guards.DataRootViolation(f"train.jsonl has {len(rows)} samples, expected {C.N_TRAIN}")
    out = [rows[i] for i in replay_indices(len(rows))]
    guards.require_samples_under_root(out, C.SHOTQA_ROOT)
    return out


def budget_image(src: str, budget_px: int, data: Path = C.DATA) -> str:
    """1x budget: the Phase 3B shrunk copy (must exist). 2x budget: shrink the ShotQA original into data/cache/step3/."""
    if budget_px == C.BUDGET_1X:
        return cached_image(src)
    if budget_px != C.BUDGET_2X:
        raise ValueError(f"unregistered image budget {budget_px}")
    return str(shrink_to_budget(src, Path(data) / f"images_{budget_px}_{C.MULTIPLE}" / (Path(src).stem + ".jpg"), budget_px,
                                C.MULTIPLE))


def train_image(sample: dict, micro: int, cand: str, data: Path = C.DATA) -> str:
    """Image fed for training sample number `micro` of candidate `cand`. G1: augmented copy keyed by the stream position."""
    spec = C.CANDIDATES[cand]
    base = budget_image(sample["images"][0], spec["budget_px"], data)
    if not spec["augment"]:
        return base
    dst = Path(data) / "aug" / cand / f"{micro:05d}.png"
    augment.augment_file(base, dst, C.AUG_SEED, f"{micro}:{sample['id']}", sample["attribute"])
    return str(dst)


def val_image(sample: dict, cand: str, data: Path = C.DATA) -> str:
    """Validation images: the candidate's TRAINING budget (G0/G1 and the baseline 1x, G2 2x); never augmented."""
    return budget_image(sample["images"][0], C.CANDIDATES[cand]["budget_px"] if cand in C.CANDIDATES else C.BUDGET_1X, data)


def proxy_subset(rows: list[dict], per_attr: int = C.PROXY_PER_ATTR, seed: int = C.PROXY_SEED) -> list[int]:
    """Positions (into the dedup_v2 list) of the proxy subset: for each attribute in ATTRS order, default_rng(seed + k).choice of
    `per_attr` positions without replacement; returned sorted."""
    out = []
    for k, a in enumerate(ATTRS):
        pos = [i for i, r in enumerate(rows) if r["attribute"] == a]
        out += [int(x) for x in np.random.default_rng(seed + k).choice(pos, min(per_attr, len(pos)), replace=False)]
    return sorted(out)


def proxy_image(sample: dict, cand: str, data: Path = C.DATA) -> str:
    src = val_image(sample, cand, data)
    budget = C.CANDIDATES[cand]["budget_px"] if cand in C.CANDIDATES else C.BUDGET_1X
    dst = Path(data) / f"proxy_{budget}" / (Path(src).stem + ".png")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            augment.proxy_transform(im, C.PROXY_SCALE, C.PROXY_JPEG_Q).save(dst, format="PNG")
    return str(dst)
