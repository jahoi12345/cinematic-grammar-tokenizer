"""Published ShotBench Table 2 (arXiv 2506.21356v2): recompute the 7-static-column means and check the printed Avg."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/phase3/shotbench_published_table2.json"
STATIC = ["SS", "SF", "CA", "LS", "LT", "LC", "SC"]


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text())


def test_shape_and_source(table):
    assert table["columns"] == STATIC + ["CM"]
    assert "2506.21356v2" in table["source"] and "Table 2" in table["source"]
    assert "NOT reruns" in table["note"]
    rows = table["rows"]
    assert len(rows) == table["n_rows"] == 25
    assert len({r["model"] for r in rows}) == 25
    groups = [r["group"] for r in rows]
    assert groups.count("open") == 21 and groups.count("proprietary") == 3 and groups.count("shotvl") == 1
    for r in rows:
        for c in table["columns"] + ["Avg_published"]:
            assert 0 < r[c] < 100


def test_static_means_recomputed(table):
    for r in table["rows"]:
        assert r["mean7_static"] == round(sum(r[c] for c in STATIC) / 7, 2), r["model"]


def test_known_project_values(table):
    by = {r["model"]: r for r in table["rows"]}
    assert by["ShotVL-3B"]["mean7_static"] == 66.83  # a recorded decision
    assert by["GPT-4o"]["mean7_static"] == 60.84
    assert by["Qwen2.5-VL-72B-Instruct"]["Avg_published"] == 59.1  # stated in the paper's text
    assert by["GPT-4o"]["Avg_published"] == 59.3


def test_published_avg_against_8_column_mean(table):
    mismatches = []
    for r in table["rows"]:
        m8 = sum(r[c] for c in table["columns"]) / 8
        ok = abs(m8 - r["Avg_published"]) <= 0.05 + 1e-9
        assert r["avg_matches_mean8"] == ok, r["model"]
        if not ok:
            mismatches.append(r["model"])
    # flagged rows must be exactly the ones listed in the file
    assert mismatches == [m["model"] for m in table["published_avg_mismatches"]] == ["ShotVL-3B"]
