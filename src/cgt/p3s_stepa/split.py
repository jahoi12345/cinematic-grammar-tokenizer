"""Deduplicated validation set and its film-level calibration / selection split (plan sections 1-2)."""
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

SPLIT_SEED = 0
ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]


def dedup_first(rows: list[dict]) -> list[dict]:
    """Keep the FIRST occurrence (val.jsonl line order) of every question id; each kept row gets `line` (its val.jsonl line index)
    and `key` = "<line>:<id>". val.jsonl has 64 ids that occur twice; a recorded decision counts each id once."""
    seen, out = set(), []
    for n, r in enumerate(rows):
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append({**r, "line": n, "key": f"{n}:{r['id']}"})
    return out


def split_films_half(films: list[str], seed: int = SPLIT_SEED) -> tuple[set[str], set[str]]:
    """(calibration films, selection films): sorted distinct film keys, permuted with np.random.default_rng(seed); the first
    ceil(n/2) are calibration, the rest selection. Disjoint and covering."""
    fs = sorted(set(films))
    perm = np.random.default_rng(seed).permutation(len(fs))
    n_cal = (len(fs) + 1) // 2
    cal = {fs[i] for i in perm[:n_cal]}
    return cal, set(fs) - cal


def load_split(val_path: Path, film_of: Callable[[set[str]], dict[str, str]], seed: int = SPLIT_SEED) -> dict:
    """{questions: deduplicated rows with `film` and `half` ('cal'/'sel'), info: counts}. `film_of(img_ids)` -> {img_id: film key}
    (ShotQA meta.jsonl titles via cgt.p3_static_attrs.data.image_films); an image without a film raises ValueError."""
    raw = [json.loads(x) for x in Path(val_path).read_text().splitlines()]
    qs = dedup_first(raw)
    fmap = film_of({q["img_id"] for q in qs})
    for q in qs:
        f = fmap.get(q["img_id"])
        if not f:
            raise ValueError(f"{q['id']}: no film key")
        q["film"] = f
    cal, sel = split_films_half([q["film"] for q in qs], seed)
    for q in qs:
        q["half"] = "cal" if q["film"] in cal else "sel"
    info = {"val_lines": len(raw), "distinct_ids": len(qs), "dropped_duplicate_lines": len(raw) - len(qs), "films": len(cal) + len(sel),
            "seed": seed}
    for h, fs in (("cal", cal), ("sel", sel)):
        hq = [q for q in qs if q["half"] == h]
        info[h] = {"films": len(fs), "images": len({q["img_id"] for q in hq}), "questions": len(hq),
                   "per_attribute": {a: sum(q["attribute"] == a for q in hq) for a in ATTRS}}
    return {"questions": qs, "info": info}


def shotqa_film_of(img_ids: set[str]) -> dict[str, str]:
    from cgt.p3_static_attrs.data import SHOTQA_ROOT, assert_shotqa_path, image_films
    return image_films(assert_shotqa_path(SHOTQA_ROOT / "meta.jsonl"), img_ids)[0]
