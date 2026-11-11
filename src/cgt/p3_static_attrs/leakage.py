"""Leakage checks between ShotQA (training source) and ShotBench (test): exact ids and 64-bit dHash near-duplicates."""
import ast
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def dhash(path: Path) -> int:
    """64-bit difference hash: grey, resize to 9x8 (bilinear), bit = pixel > left neighbour, row-major. Raises OSError if unreadable."""
    im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _safe_dhash(path: Path) -> int | None:
    try:
        return dhash(Path(path))
    except OSError:
        return None


def hash_paths(paths: Sequence[Path], workers: int = 1) -> tuple[list[str], np.ndarray]:
    """(basenames, uint64 hashes) of the readable images, in input order; unreadable files are skipped silently.
    `workers > 1` hashes in a process pool (same results, order preserved)."""
    paths = [Path(p) for p in paths]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_safe_dhash, paths, chunksize=64))
    else:
        res = [_safe_dhash(p) for p in paths]
    ids = [p.name for p, h in zip(paths, res, strict=True) if h is not None]
    return ids, np.array([h for h in res if h is not None], dtype=np.uint64)


def shotbench_image_ids(test_tsv: Path) -> set[str]:
    """Basenames of every image referenced by image-type rows of ShotBench test.tsv (column `path` is a Python-literal list)."""
    sb = pd.read_csv(test_tsv, sep="\t")
    out: set[str] = set()
    for p, t in zip(sb.path, sb.type, strict=True):
        if "image" in t:
            out |= {Path(q).name for q in ast.literal_eval(p)}
    return out


def exact_overlap(qa_ids: set[str], sb_ids: set[str]) -> set[str]:
    return set(qa_ids) & set(sb_ids)


def near_duplicates(qa_ids: Sequence[str], qa_hashes: np.ndarray, sb_ids: Sequence[str], sb_hashes: np.ndarray,
                    max_distance: int = 8, chunk: int = 2048) -> dict[str, list[tuple[str, int]]]:
    """ShotQA id -> [(ShotBench id, hamming distance), ...] for every pair with distance <= max_distance (0 = identical hash)."""
    out: dict[str, list[tuple[str, int]]] = {}
    for s in range(0, len(qa_hashes), chunk):
        d = np.bitwise_count(np.bitwise_xor(qa_hashes[s:s + chunk, None], sb_hashes[None, :]))
        for r, c in zip(*np.nonzero(d <= max_distance), strict=True):
            out.setdefault(qa_ids[s + r], []).append((sb_ids[c], int(d[r, c])))
    return out


def distance_counts(near: dict[str, list[tuple[str, int]]], max_distance: int = 8) -> list[dict]:
    """One row per d in 0..max_distance: `exactly` = ShotQA images whose MINIMUM distance to any ShotBench image is d,
    `cumulative` = images with minimum distance <= d."""
    mins = [min(dd for _, dd in v) for v in near.values()]
    rows, cum = [], 0
    for d in range(max_distance + 1):
        n = sum(1 for m in mins if m == d)
        cum += n
        rows.append({"distance": d, "exactly": n, "cumulative": cum})
    return rows


def exclusion_ids(exact: set[str], near: dict[str, list[tuple[str, int]]], max_hamming: int = 4) -> set[str]:
    """Ids to drop from training/validation: exact id matches plus ShotQA images within `max_hamming` of a ShotBench image."""
    close = {i for i, v in near.items() if min(dd for _, dd in v) <= max_hamming}
    return set(exact) | close
