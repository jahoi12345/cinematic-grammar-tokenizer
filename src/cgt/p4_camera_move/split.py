"""Source-grouped train/validation split for the ShotQA movement clips (docs/phase4_interface.md section 2). Deterministic, no source on both sides."""
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "results/phase4/split_manifest.json"


def source_grouped_split(sources: Sequence[str], val_frac: float = 0.15, seed: int = 0) -> dict[str, str]:
    """{source: "train" | "val"}. Sorted unique sources, one `default_rng(seed).permutation`, the first round(val_frac * n) (at least 1) are validation."""
    uniq = sorted(set(sources))
    n_val = max(1, round(val_frac * len(uniq)))
    if not 0 < n_val < len(uniq):
        raise ValueError(f"cannot split {len(uniq)} sources with val_frac={val_frac}")
    perm = np.random.default_rng(seed).permutation(len(uniq))
    val = {uniq[i] for i in perm[:n_val]}
    out = {s: ("val" if s in val else "train") for s in uniq}
    assert set(out) == set(uniq) and sum(v == "val" for v in out.values()) == n_val
    assert 0 < n_val < len(uniq)
    return out


def write_split_manifest(rows: Sequence[dict], path: Path = MANIFEST, val_frac: float = 0.15, seed: int = 0) -> dict:
    """Write the manifest for rows with keys `clip`, `source`; asserts every clip has exactly one side and NO source appears on both sides."""
    side = source_grouped_split([r["source"] for r in rows], val_frac, seed)
    clips = {"train": sorted(r["clip"] for r in rows if side[r["source"]] == "train"), "val": sorted(r["clip"] for r in rows if side[r["source"]] == "val")}
    train_src = {r["source"] for r in rows if side[r["source"]] == "train"}
    val_src = {r["source"] for r in rows if side[r["source"]] == "val"}
    if train_src & val_src:
        raise AssertionError(f"sources on both sides: {sorted(train_src & val_src)[:5]}")
    if len(clips["train"]) + len(clips["val"]) != len({r["clip"] for r in rows}) or set(clips["train"]) & set(clips["val"]):
        raise AssertionError("a clip is missing or on both sides")
    body = {"seed": seed, "val_frac": val_frac, "n_sources": len(side), "n_val_sources": len(val_src), "n_clips": len(rows),
            "n_train_clips": len(clips["train"]), "n_val_clips": len(clips["val"]), "train_sources": sorted(train_src), "val_sources": sorted(val_src), "clips": clips}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body, indent=1))
    return body
