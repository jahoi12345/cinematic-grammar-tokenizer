"""an earlier review guard tests for the val dedup v2 key and the re-score scorer (a verification record). CPU, synthetic only."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from cgt.eval.val_dedup import dedup_key, dedup_v2, dedup_v2_indices

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")


def q(i, attr, opts, ans="A"):
    return {"id": i, "attribute": attr, "options": dict(zip("ABCD", opts, strict=True)), "answer": ans}


def test_same_id_same_option_texts_different_attribute_is_kept():
    # the attribute is part of the key: identical option texts asked about a different attribute is a different question
    a = q("x#0", "light_type", ["Daylight", "Practical", "Mixed", "Firelight"])
    b = q("x#0", "light_cond", ["Daylight", "Practical", "Mixed", "Firelight"])
    assert dedup_v2_indices([a, b]) == [0, 1]
    assert dedup_key(a) != dedup_key(b)


def test_key_ignores_letter_order_but_not_option_text():
    a = q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Fisheye"])
    assert dedup_key(a) == dedup_key(q("x#0", "lens", ["Fisheye", "Long Lens", "Medium", "Wide"], ans="D"))
    assert dedup_key(a) != dedup_key(q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Telephoto"]))
    assert dedup_key(a) != dedup_key(q("y#0", "lens", ["Wide", "Medium", "Long Lens", "Fisheye"]))


def test_first_occurrence_kept_in_line_order():
    a = q("x#0", "lens", ["Wide", "Medium", "Long Lens", "Fisheye"])
    b = q("y#0", "size", list("abcd"))
    r = q("x#0", "lens", ["Long Lens", "Wide", "Fisheye", "Medium"], ans="B")
    out = dedup_v2([a, b, r])
    assert out[0] is a and out[1] is b and len(out) == 2


@pytest.fixture(scope="module")
def rescore():
    p = ROOT / "scripts/val_dedup_v2_rescore.py"
    if not p.exists():
        pytest.skip("rescore script absent")
    spec = spec_from_file_location("val_dedup_v2_rescore", p)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rescore_primary_is_unweighted_attribute_mean(rescore):
    # 7 attributes; size has 3 questions (all right), the others 1 question each (wrong) -> unweighted mean 100/7, micro mean 30
    rows, hit = [], {}
    for a in ATTRS:
        for _ in range(3 if a == "size" else 1):
            hit[len(rows)] = 1.0 if a == "size" else 0.0
            rows.append({"attribute": a})
    s = rescore.scores_from_hits(list(range(len(rows))), hit, rows)
    assert s["n"] == len(rows)
    assert s["per_attribute"]["size"] == 100.0 and s["per_attribute"]["lens"] == 0.0
    assert abs(s["primary"] - 100.0 / 7) < 1e-12
