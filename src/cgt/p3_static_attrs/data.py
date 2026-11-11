"""ShotQA-only data access and the image-level train/val split (docs/phase3_interface.md sections 0 and 2)."""
import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SHOTQA_ROOT = ROOT / "data/external/shotqa"


def film_key(title: str) -> str:
    """Film identity from a meta.jsonl title: drop TV suffixes ('/season/1/episode/2'), lower-case, '&' -> 'and', keep [a-z0-9].
    Series and a same-titled film share a key on purpose (conservative: they are held out together)."""
    t = title.split("/season")[0].split("/episode")[0]
    return re.sub(r"[^a-z0-9]+", "", t.lower().replace("&", "and"))


def image_films(meta_path: Path, img_ids: Iterable[str]) -> tuple[dict[str, str | None], dict]:
    """{img_id: film_key or None}. Joins `img_ids` (exact basename incl. extension) to meta.jsonl `img_id`. An image with no
    meta row (the .png/.JPG/.jpeg images) or an empty title gets None (untitled). An id whose rows disagree on the film
    raises ValueError (does not occur in ShotQA). `meta_path` must be a guarded ShotQA path."""
    by: dict[str, set[str]] = {}
    for line in Path(meta_path).read_text().splitlines():
        m = json.loads(line)
        title = (m.get("title") or "").strip()
        if title:
            by.setdefault(m["img_id"], set()).add(film_key(title))
    out: dict[str, str | None] = {}
    for i in img_ids:
        keys = by.get(i, set())
        if len(keys) > 1:
            raise ValueError(f"{i}: conflicting films {sorted(keys)}")
        out[i] = next(iter(keys)) if keys else None
    return out, {"images": len(out), "untitled": sum(v is None for v in out.values())}


def split_films(img_film: dict[str, str | None], val_frac: float = 0.1, seed: int = 0,
                exclude: Iterable[str] = ()) -> tuple[list[str], list[str], dict]:
    """Film-held-out split. Untitled images (film None) and `exclude` ids are dropped from BOTH sides (an untitled image may
    belong to a held-out film, so it cannot safely be trained on). Films = sorted distinct keys of the remaining images;
    val films = `round(n_films * val_frac)` films from `np.random.default_rng(seed).permutation`. Returns
    (train_ids, val_ids, info): both id lists sorted, no film in both, info has counts and the held-out film keys."""
    ex = set(exclude)
    titled = {i: f for i, f in img_film.items() if f is not None and i not in ex}
    films = sorted(set(titled.values()))
    perm = np.random.default_rng(seed).permutation(len(films))
    val_films = {films[i] for i in perm[: round(len(films) * val_frac)]}
    val = sorted(i for i, f in titled.items() if f in val_films)
    train = sorted(i for i, f in titled.items() if f not in val_films)
    info = {"n_films": len(films), "n_val_films": len(val_films), "train_images": len(train), "val_images": len(val),
            "val_image_fraction": round(len(val) / max(len(titled), 1), 4),
            "untitled_dropped": sum(1 for i, f in img_film.items() if f is None and i not in ex),
            "excluded_dropped": sum(1 for i in img_film if i in ex), "val_films": sorted(val_films)}
    return train, val, info


def image_dir() -> Path:
    """Directory holding ShotQA images. The released data.tar.gz extracts FLAT into data/external/shotqa/ (the
    `data/<id>` paths in sft.json do not exist as a folder); a `data/` subfolder is honoured if someone creates one."""
    return SHOTQA_ROOT / "data" if (SHOTQA_ROOT / "data").is_dir() else SHOTQA_ROOT


def image_path(img_id: str) -> Path:
    """Guarded path of a ShotQA image by basename (`ABC.jpg`; extensions vary: .jpg .png .JPG .jpeg). May not exist."""
    return assert_shotqa_path(image_dir() / img_id)


def load_leak_ids(results_dir: Path | None = None) -> set[str]:
    """Union of ShotQA ids excluded as ShotBench leakage: `leak_ids.json` (exact ids + hamming <= 4) and
    `leak_ids_pixel.json` (pixel-verified same-frame pairs within hamming <= 8). Missing files count as empty."""
    import json
    d = Path(results_dir) if results_dir else ROOT / "results/phase3"
    out: set[str] = set()
    for name in ("leak_ids.json", "leak_ids_pixel.json"):
        if (d / name).exists():
            out |= set(json.loads((d / name).read_text()))
    return out


def assert_shotqa_path(path) -> Path:
    """Raise unless `path` resolves inside data/external/shotqa/ (training code may read nothing else)."""
    p = Path(path).resolve()
    root = SHOTQA_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"training data must live under {root}, got {p}")
    return p


def split_images(img_ids, val_frac: float = 0.1, seed: int = 0, exclude=()) -> tuple[list[str], list[str]]:
    """Deterministic image-level split. `exclude` (e.g. ShotBench leakage ids) is removed from both sides."""
    ids = sorted(set(img_ids) - set(exclude))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = round(len(ids) * val_frac)
    val = sorted(ids[i] for i in perm[:n_val])
    train = sorted(ids[i] for i in perm[n_val:])
    return train, val
