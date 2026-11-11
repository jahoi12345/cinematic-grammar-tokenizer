"""Cross-fitting bookkeeping for the video LoRA (docs/phase4_video_plan.md Amendment 2, docs/phase4_interface.md section 8). Pure CPU, no labels read here.
Fold assignment, per-epoch sample order, the epoch rule and the learning-rate schedule are FIXED by Amendment 2; this module only implements them."""
import math

import numpy as np

N_FOLDS, SEED, EPOCHS, ACCUM, WARMUP, PEAK_LR, END_LR = 3, 0, 3, 8, 10, 1e-4, 1e-5


def fold_assignment(sources, n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    """{source: fold}: the sorted unique sources are permuted once with `default_rng(seed)` and the source at permutation rank r goes to fold r mod n_folds."""
    uniq = sorted(set(sources))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[int(i)]: r % n_folds for r, i in enumerate(perm)}


def stage_split(rows: list[dict], stage: str) -> tuple[list[dict], list[dict]]:
    """(train rows, held-out rows) of a stage: `fold0|fold1|fold2` train on the other folds' clips and hold out their own fold; `final` trains on every row, holds out nothing.
    Rows need `source`; the fold of a clip is the fold of its source."""
    if stage == "final":
        return list(rows), []
    k = int(stage.removeprefix("fold"))
    fa = fold_assignment([r["source"] for r in rows])
    return [r for r in rows if fa[r["source"]] != k], [r for r in rows if fa[r["source"]] == k]


def stage_seed(stage: str) -> int:
    """Sample-order seed s of Amendment 2: 0 for the final fit, k + 1 for fold k."""
    return 0 if stage == "final" else int(stage.removeprefix("fold")) + 1


def epoch_order(n: int, stage: str, epoch: int) -> np.ndarray:
    """Order of the n training samples in epoch e (1-based epoch number is used as given): `default_rng([s, e]).permutation(n)`."""
    return np.random.default_rng([stage_seed(stage), epoch]).permutation(n)


def updates_per_epoch(n: int, accum: int = ACCUM) -> int:
    return math.ceil(n / accum)


def total_updates(n: int, epochs: int = EPOCHS, accum: int = ACCUM) -> int:
    return epochs * updates_per_epoch(n, accum)


def lr_at(update: int, total: int, warmup: int = WARMUP, peak: float = PEAK_LR, end: float = END_LR) -> float:
    """Learning rate used for the 1-based `update`: linear warm-up from 0 to the peak over `warmup` updates (reaching the peak at update `warmup`), then cosine decay to `end`
    over the remaining updates (the value at the last update equals `end`)."""
    if update <= warmup:
        return peak * update / warmup
    t = (update - warmup) / max(total - warmup, 1)
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * min(t, 1.0)))


def choose_epoch(oof_correct: dict[int, list[bool]]) -> tuple[int, dict[int, float]]:
    """E = the epoch in {1, 2, 3} with the highest out-of-fold mean option accuracy pooled over all held-out predictions; ties go to the smaller E.
    `oof_correct[e]` = list of hit booleans of every OOF prediction at epoch e (all folds pooled). Returns (E, {epoch: accuracy})."""
    acc = {e: float(np.mean(v)) for e, v in oof_correct.items()}
    best = max(acc.values())
    return min(e for e, a in acc.items() if a == best), acc

