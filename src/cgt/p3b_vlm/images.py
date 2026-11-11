"""Aspect-preserving image shrinking to a pixel budget (reduced image size for the Phase 3B benchmark)."""
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image


def budget_size(w: int, h: int, budget_px: int, multiple: int) -> tuple[int, int]:
    """(W, H): same aspect ratio as (w, h), area <= budget_px, both sides multiples of `multiple` (at least one multiple)."""
    scale = math.sqrt(budget_px / (w * h))
    nw = max(multiple, int(w * scale) // multiple * multiple)
    nh = max(multiple, int(h * scale) // multiple * multiple)
    while nw * nh > budget_px and (nw > multiple or nh > multiple):
        if nw >= nh and nw > multiple:
            nw -= multiple
        elif nh > multiple:
            nh -= multiple
        else:
            nw -= multiple
    return nw, nh


def shrink_to_budget(src: str, dst: Path, budget_px: int, multiple: int) -> Path:
    """Write a resized RGB JPEG copy of `src` to `dst` (skipped if it exists) and return `dst`."""
    dst = Path(dst)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src).convert("RGB")
        im.resize(budget_size(*im.size, budget_px, multiple), Image.BICUBIC).save(dst, quality=92)
    return dst


BUDGET_PX = 147_456  # 384 x 384 area: at most 144 Qwen3-VL image tokens (32-pixel cells); same value for train, validation, ShotBench
MULTIPLE = 32


def shrunk_path(src: str, cache_root: Path, budget_px: int = BUDGET_PX, multiple: int = MULTIPLE) -> str:
    """Path of the budget-shrunk copy of `src` under `cache_root` (created on first use, safe to call from worker processes)."""
    return str(shrink_to_budget(src, Path(cache_root) / (Path(src).stem + ".jpg"), budget_px, multiple))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


RECIPE = "shrink_to_budget-v1: aspect-preserving, bicubic, RGB JPEG quality 92, sides floored to multiples"  # bump if the preprocessing ever changes


def prepare_eval_cache(sources: list[str], cache_root: Path, budget_px: int = BUDGET_PX, multiple: int = MULTIPLE) -> dict[str, str]:
    """Evaluation image cache with collision-resistant keys and a provenance manifest (`<cache_root>/provenance.json`).
    Key = `<source file name WITH extension>.<first 16 hex of the source content SHA-256>.jpg`; two different source files that map to the same key
    raise ValueError. The cache must be fresh (empty) or verified: every cached file must be listed in the manifest with the recorded output SHA-256
    (a stray, missing-from-manifest or modified file raises ValueError). An existing manifest's budget, multiple and recipe must equal the current request (ValueError otherwise). Resize semantics are exactly `shrink_to_budget`. Returns {source: cached path}."""
    seen_stems: dict[str, str] = {}
    for src in dict.fromkeys(str(x) for x in sources):  # refuse BEFORE any cache entry is created; the same source repeated is fine
        stem = Path(src).stem.casefold()
        if stem in seen_stems:
            raise ValueError(f"two distinct source images share the stem '{stem}': {seen_stems[stem]} and {src}")
        seen_stems[stem] = src
    root = Path(cache_root)
    mf = root / "provenance.json"
    entries: dict = {}
    if mf.exists():
        prov = json.loads(mf.read_text())
        want = {"budget_px": budget_px, "multiple": multiple, "resize": RECIPE}
        got = {k: prov.get(k) for k in want}
        if got != want:  # never relabel old outputs under a new configuration
            raise ValueError(f"cache {root} was built with budget/multiple/recipe {got}, the request is {want}")
        entries = prov["entries"]
    present = {p.name for p in root.glob("*.jpg")} if root.exists() else set()
    if present - set(entries):
        raise ValueError(f"cache {root} holds {len(present - set(entries))} files that are not in its provenance manifest (stale or foreign cache)")
    for key, e in entries.items():
        if key in present and _sha256(root / key) != e["output_sha256"]:
            raise ValueError(f"cached file {key} no longer matches its recorded SHA-256")
    out, owner = {}, {}
    for src in sources:
        sp = Path(src)
        digest = _sha256(sp)
        key = f"{sp.name}.{digest[:16]}.jpg"
        if key in owner and owner[key] != str(sp):
            raise ValueError(f"two different ShotBench images map to the same cache key {key}: {owner[key]} and {sp}")
        owner[key] = str(sp)
        if key in entries and (root / key).exists():
            if entries[key]["source_sha256"] != digest:
                raise ValueError(f"cache key {key} was built from different source content")
        else:
            (root / key).unlink(missing_ok=True)
            shrink_to_budget(src, root / key, budget_px, multiple)
            entries[key] = {"source": str(sp), "source_sha256": digest, "output_sha256": _sha256(root / key)}
        out[src] = str(root / key)
    tmp = root / ".tmp_provenance.json"
    tmp.write_text(json.dumps({"budget_px": budget_px, "multiple": multiple, "resize": RECIPE,
                               "entries": entries}, indent=1))
    os.replace(tmp, mf)
    return out
