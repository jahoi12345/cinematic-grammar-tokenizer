"""Step B training streams and validation sets (docs/plans/stepb_cinescale2.md sections 3-4). Pure Python/numpy (CPU tests).

- ShotQA replay: a seeded subset (numpy default_rng(1), without replacement) of the Phase 3B training samples (data/cache/p3b/train.jsonl).
- CineScale2 angle questions: one 4-option question per drawn image, worded exactly like ShotQA's angle turn; the options are the
  correct class plus 3 of the 4 other CineScale2-distinguishable classes (overhead, high, eye-level, low, dutch; never aerial);
  distractor choice and letter order come from a per-sample seed = int(sha256(f"{relpath}:{pass_index}")[:8], 16).
- Class balance: each CineScale2 draw picks a class uniformly (default_rng(seed)), then an image of that class uniformly
  (with replacement inside the class); the pass index of an image is the number of times it was drawn before.
- Mixed stream: concatenation, then one default_rng(0) permutation.
- ShotQA validation: data/cache/p3b/val.jsonl with the 64 duplicated ids deduplicated (first occurrence in line order kept)."""
import hashlib
from collections.abc import Sequence

import numpy as np

from cgt.cinescale2.labels import ANGLE_TO_SCHEMA, CLASSES, SCHEMA_ANGLE_TEXT

LETTERS = ("A", "B", "C", "D")
CS2_CLASSES = tuple(ANGLE_TO_SCHEMA[c] for c in CLASSES["angle"])   # overhead, high, eye-level, low, dutch
QUESTION = "Question: What's the camera angle of this shot?\nOptions:\n{opts}\nPlease select the most likely answer from the options above. \n"


def sample_seed(relpath: str, pass_index: int) -> int:
    return int(hashlib.sha256(f"{relpath}:{pass_index}".encode()).hexdigest()[:8], 16)


def cs2_question(relpath: str, abs_path: str, cs2_label: str, pass_index: int) -> dict:
    """One CineScale2 angle question in the Phase 3B chat-sample format (+ `source`, `cs2_class`, `relpath`)."""
    correct = ANGLE_TO_SCHEMA[cs2_label]
    rng = np.random.default_rng(sample_seed(relpath, pass_index))
    others = [c for c in CS2_CLASSES if c != correct]
    picked = [others[i] for i in sorted(rng.choice(len(others), size=3, replace=False))]
    opts = [correct, *picked]
    order = rng.permutation(4)
    opts = [opts[i] for i in order]
    options = {LETTERS[k]: SCHEMA_ANGLE_TEXT[v] for k, v in enumerate(opts)}
    answer = LETTERS[opts.index(correct)]
    text = QUESTION.format(opts="\n".join(f"{k}. {v}" for k, v in options.items()))
    return {"id": f"cs2:{relpath}#{pass_index}", "img_id": relpath, "attribute": "angle", "images": [abs_path],
            "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": answer}],
            "options": options, "answer": answer, "source": "cs2", "cs2_class": correct, "relpath": relpath}


def balanced_draws(by_class: dict[str, Sequence[str]], n: int, seed: int) -> list[tuple[str, int]]:
    """n (relpath, pass_index) draws: class uniform over the non-empty classes, image uniform within the class (replacement)."""
    rng = np.random.default_rng(seed)
    classes = [c for c in sorted(by_class) if by_class[c]]
    seen: dict[str, int] = {}
    out = []
    for _ in range(n):
        c = classes[int(rng.integers(len(classes)))]
        imgs = by_class[c]
        r = imgs[int(rng.integers(len(imgs)))]
        out.append((r, seen.get(r, 0)))
        seen[r] = seen.get(r, 0) + 1
    return out


def replay_subset(n_total: int, n: int, seed: int = 1) -> list[int]:
    """Indices (sorted) of the ShotQA replay subset: default_rng(seed).choice without replacement."""
    if n > n_total:
        raise ValueError("replay subset larger than the ShotQA training stream")
    return sorted(int(i) for i in np.random.default_rng(seed).choice(n_total, size=n, replace=False))


def mix(samples: list[dict], seed: int = 0) -> list[dict]:
    return [samples[i] for i in np.random.default_rng(seed).permutation(len(samples))]


def dedup_first(samples: list[dict]) -> list[dict]:
    """Keep the first occurrence of each `id` in order."""
    seen, out = set(), []
    for s in samples:
        if s["id"] not in seen:
            seen.add(s["id"])
            out.append(s)
    return out


CANDIDATES = {
    # id: (number of CineScale2 samples, number of ShotQA replay samples); total 13,734 = 859 updates of 16
    "B0": (0, 13_734),
    "B1": (6_709, 13_734 - 6_709),
    "B2": (3_434, 13_734 - 3_434),
}
RUN_ORDER = ("B0", "B1", "B2")
