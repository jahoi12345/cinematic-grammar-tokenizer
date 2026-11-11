"""CPU tests for the ShotQA validation dedup v2 key (docs/plans/val_dedup_v2.md)."""
import hashlib
import json
from pathlib import Path

import pytest

from cgt.eval.val_dedup import V2_COUNT_REAL_VAL, dedup_key, dedup_v2, dedup_v2_indices
from cgt.p3s_stepa.split import dedup_first
from cgt.stepb import data as D

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/cache/p3b/val.jsonl"
VAL_SHA = "409e87d59faedc4753a27bf0e9151af1e7991cf53549434f53da4ec519c2637e"


def q(i, attr, opts, ans="A"):
    return {"id": i, "attribute": attr, "options": dict(zip("ABCD", opts, strict=True)), "answer": ans}


def test_genuine_duplicates_collapse_also_when_reordered():
    a = q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Ultra Wide / Fisheye"])
    same = q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Ultra Wide / Fisheye"])
    reordered = q("x#0", "lens", ["Long Lens", "Wide", "Ultra Wide / Fisheye", "Medium"], ans="B")
    out = dedup_v2([a, same, reordered])
    assert out == [a] and out[0] is a
    assert dedup_key(a) == dedup_key(reordered)


def test_same_id_with_other_attribute_or_option_set_is_kept():
    a = q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Ultra Wide / Fisheye"])
    other_attr = q("x#0", "angle", ["High angle", "Low angle", "Overhead", "Dutch angle"])
    other_set = q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Telephoto"])
    b = q("y#0", "lens", ["Wide", "Medium", "Long Lens", "Ultra Wide / Fisheye"])
    rows = [a, other_attr, other_set, b, other_attr]
    assert dedup_v2_indices(rows) == [0, 1, 2, 3]
    assert len(dedup_first(rows)) == 2 and len(D.dedup_first(rows)) == 2   # the old rule is unchanged


def test_v2_keeps_every_dedup_first_row():
    rows = [q("x#0", "lens", list("abcd")), q("x#0", "angle", list("efgh")), q("y#0", "size", list("ijkl")), q("x#0", "lens", list("dcba"))]
    first = {r["line"] for r in dedup_first(rows)}
    assert first <= set(dedup_v2_indices(rows)) and dedup_v2_indices(rows) == [0, 1, 2]


@pytest.mark.skipif(not VAL.exists(), reason="data/cache/p3b/val.jsonl not present")
def test_real_val_counts():
    if hashlib.sha256(VAL.read_bytes()).hexdigest() != VAL_SHA:
        pytest.skip("val.jsonl is not the recorded file")
    rows = [json.loads(x) for x in VAL.read_text().splitlines()]
    assert len(rows) == 5826 and len(dedup_first(rows)) == 5762
    keep = dedup_v2_indices(rows)
    assert len(keep) == V2_COUNT_REAL_VAL == 5822
    assert {r["line"] for r in dedup_first(rows)} <= set(keep)
    assert len({(r["id"], r["attribute"]) for r in rows}) == 5791
