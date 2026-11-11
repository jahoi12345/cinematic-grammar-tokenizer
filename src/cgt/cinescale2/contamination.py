"""Contamination checks for CineScale2: normalised title matching and perceptual-hash near-duplicates.
Hashes and titles only: no labels or answers of ShotBench are read and no accuracy is computed.

Two 64-bit dHashes per image (the Phase 3 `cgt.p3_static_attrs.leakage.dhash` definition: grey, 9x8 bilinear, bit = pixel > left):
- `full`: the whole image;
- `trim`: after cropping near-black letterbox / pillarbox borders (rows/cols whose mean grey < `BORDER_LEVEL`),
  so the same frame with and without black bars still matches.
A pair is a hash CANDIDATE if any full/trim hash combination is within the distance threshold. Hash candidates are verified at
pixel level as in Phase 3 (results/phase3/leakage_verified.md): MAD = mean absolute difference of 32x32 grey thumbnails in [0, 1]
(min over full and trimmed thumbnails); 'same frame' = MAD < `MAD_SAME` AND Pearson correlation of the thumbnails >= `CORR_SAME`
(max over full and trimmed). The correlation term is new here: on dark, low-contrast frames MAD alone is small for unrelated
images (seen on CineScale2 vs ShotBench pairs), while correlation separates them. Near-uniform (flat) frames collide in the hash."""
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

BORDER_LEVEL = 16
MAD_SAME = 0.05
CORR_SAME = 0.9
THUMB = 32


def norm_title(title: str) -> str:
    """Strict key, identical to cgt.p3_static_attrs.data.film_key: drop TV suffixes, lower, '&' -> 'and', keep [a-z0-9]."""
    t = title.split("/season")[0].split("/episode")[0]
    return re.sub(r"[^a-z0-9]+", "", t.lower().replace("&", "and"))


def loose_title(title: str) -> str:
    """Looser key: also drops a trailing '(YYYY)' / ' YYYY', a leading article and 'marvels'-style possessive apostrophes."""
    t = title.split("/season")[0].split("/episode")[0].lower().replace("&", "and").replace("'", "").replace("’", "")
    t = re.sub(r"[\s(]*(19|20)\d\d\)?\s*$", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t.strip())
    return re.sub(r"[^a-z0-9]+", "", t)


def title_overlap(a: Iterable[str], b: Iterable[str]) -> list[tuple[str, str]]:
    """Sorted (title_a, title_b) pairs whose strict OR loose keys are equal (empty keys never match)."""
    bs = list(b)
    idx: dict[str, set[str]] = {}
    for t in bs:
        for k in (norm_title(t), loose_title(t)):
            if k:
                idx.setdefault(k, set()).add(t)
    out = set()
    for t in a:
        for k in (norm_title(t), loose_title(t)):
            for m in idx.get(k, ()) if k else ():
                out.add((t, m))
    return sorted(out)


def _dhash_arr(a: np.ndarray) -> int:
    im = Image.fromarray(a).resize((9, 8), Image.BILINEAR)
    g = np.asarray(im, dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if x else "0" for x in bits), 2)


def trim_borders(a: np.ndarray, level: int = BORDER_LEVEL) -> np.ndarray:
    """Crop rows/cols at the edges whose mean grey value is below `level`; returns `a` unchanged if all would go."""
    rows = np.nonzero(a.mean(axis=1) >= level)[0]
    cols = np.nonzero(a.mean(axis=0) >= level)[0]
    if len(rows) < 2 or len(cols) < 2:
        return a
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _thumb(a: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)


def hashes(path: Path) -> tuple[int, int, tuple[int, int], np.ndarray]:
    """(full dHash, border-trimmed dHash, (width, height), uint8[2, 32, 32] full and trimmed grey thumbnails) of one image.
    Raises OSError if unreadable."""
    with Image.open(path) as im:
        size = im.size
        g = np.asarray(im.convert("L"), dtype=np.uint8)
    t = trim_borders(g)
    return _dhash_arr(g), _dhash_arr(t), size, np.stack([_thumb(g), _thumb(t)])


def _safe(path):
    try:
        return hashes(Path(path))
    except (OSError, ValueError):
        return None


def hash_many(paths: Sequence[Path], workers: int = 1) -> tuple[list[int], np.ndarray, np.ndarray, list, np.ndarray]:
    """(indices of readable inputs, full uint64[n], trim uint64[n], sizes, thumbs uint8[n, 2, 32, 32]) in input order;
    unreadable files are skipped."""
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_safe, [str(p) for p in paths], chunksize=64))
    else:
        res = [_safe(p) for p in paths]
    ok = [i for i, r in enumerate(res) if r is not None]
    full = np.array([res[i][0] for i in ok], dtype=np.uint64)
    trim = np.array([res[i][1] for i in ok], dtype=np.uint64)
    thumbs = np.array([res[i][3] for i in ok], dtype=np.uint8).reshape(-1, 2, THUMB, THUMB)
    return ok, full, trim, [res[i][2] for i in ok], thumbs


def min_distances(a_full: np.ndarray, a_trim: np.ndarray, b_full: np.ndarray, b_trim: np.ndarray,
                  chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each row of A: (minimum hamming distance to any row of B over both hash kinds, argmin index into B).
    The distance of a pair is min(d(full_a, full_b), d(trim_a, trim_b), d(full_a, trim_b), d(trim_a, full_b))."""
    n = len(a_full)
    best = np.full(n, 65, dtype=np.int16)
    arg = np.full(n, -1, dtype=np.int64)
    if len(b_full) == 0:
        return best, arg
    for s in range(0, n, chunk):
        d = None
        for x in (a_full[s:s + chunk], a_trim[s:s + chunk]):
            for y in (b_full, b_trim):
                dd = np.bitwise_count(np.bitwise_xor(x[:, None], y[None, :])).astype(np.int16)
                d = dd if d is None else np.minimum(d, dd)
        best[s:s + chunk] = d.min(axis=1)
        arg[s:s + chunk] = d.argmin(axis=1)
    return best, arg


def distance_histogram(dist: np.ndarray, max_distance: int = 10) -> list[dict]:
    """Rows d = 0..max_distance: images whose minimum distance is exactly d, and cumulative <= d."""
    rows, cum = [], 0
    for d in range(max_distance + 1):
        k = int((dist == d).sum())
        cum += k
        rows.append({"distance": d, "exactly": k, "cumulative": cum})
    return rows


def candidate_pairs(a_full: np.ndarray, a_trim: np.ndarray, b_full: np.ndarray, b_trim: np.ndarray, max_distance: int,
                    chunk: int = 1024, same_set: bool = False) -> np.ndarray:
    """int64[k, 3] rows (i into A, j into B, distance) for every pair within `max_distance` (min over the 4 hash combinations).
    `same_set=True` (A is B) keeps only i < j."""
    out = []
    for s in range(0, len(a_full), chunk):
        d = None
        for x in (a_full[s:s + chunk], a_trim[s:s + chunk]):
            for y in (b_full, b_trim):
                dd = np.bitwise_count(np.bitwise_xor(x[:, None], y[None, :])).astype(np.int16)
                d = dd if d is None else np.minimum(d, dd)
        r, c = np.nonzero(d <= max_distance)
        keep = (r + s < c) if same_set else np.ones(len(r), bool)
        out.append(np.stack([r[keep] + s, c[keep], d[r[keep], c[keep]]], axis=1).astype(np.int64))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


def pair_mad(a_thumbs: np.ndarray, b_thumbs: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """MAD in [0, 1] per candidate pair: min over (full, full) and (trim, trim) thumbnails."""
    if len(pairs) == 0:
        return np.zeros(0, np.float32)
    a = a_thumbs[pairs[:, 0]].astype(np.float32) / 255.0
    b = b_thumbs[pairs[:, 1]].astype(np.float32) / 255.0
    return np.abs(a - b).mean(axis=(2, 3)).min(axis=1)


def is_flat(thumbs: np.ndarray, level: float = 0.05) -> np.ndarray:
    """Near-uniform frames (std of the full grey thumbnail < level), which collide in the hash."""
    return (thumbs[:, 0].astype(np.float32) / 255.0).std(axis=(1, 2)) < level


def pair_corr(a_thumbs: np.ndarray, b_thumbs: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Pearson correlation per candidate pair: max over (full, full) and (trim, trim) thumbnails; 0 for a constant thumbnail."""
    if len(pairs) == 0:
        return np.zeros(0, np.float32)
    a = a_thumbs[pairs[:, 0]].astype(np.float32).reshape(len(pairs), 2, -1)
    b = b_thumbs[pairs[:, 1]].astype(np.float32).reshape(len(pairs), 2, -1)
    a = a - a.mean(axis=2, keepdims=True)
    b = b - b.mean(axis=2, keepdims=True)
    den = np.sqrt((a * a).sum(axis=2) * (b * b).sum(axis=2))
    r = np.where(den > 0, (a * b).sum(axis=2) / np.where(den > 0, den, 1), 0.0)
    return r.max(axis=1)


def same_frame(a_thumbs: np.ndarray, b_thumbs: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(same-frame mask, MAD, correlation) for candidate pairs."""
    mad = pair_mad(a_thumbs, b_thumbs, pairs)
    corr = pair_corr(a_thumbs, b_thumbs, pairs)
    return (mad < MAD_SAME) & (corr >= CORR_SAME), mad, corr
