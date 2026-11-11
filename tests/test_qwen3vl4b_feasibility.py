"""an earlier review of an earlier change (Qwen3-VL-4B feasibility, cost only): helper tests that kill mutants the worker's tests miss, and a
consistency check of the committed report against its own raw per-update timings. CPU only, reads only the committed report."""
import json
import math
import statistics
from pathlib import Path

import pytest

from cgt.p3b_vlm.feasibility import n_updates, project_hours, timing_summary

REPORT = Path(__file__).resolve().parents[1] / "results/qwen3vl4b_feasibility/report.json"


def test_timing_summary_is_median_not_mean():
    # kept = [1, 2, 10]: median 2, mean 4.33 (the worker's test has median == mean, so a median->mean swap survives it)
    s = timing_summary([0.0, 0.0, 1.0, 2.0, 10.0], warmup=2)
    assert s["median"] == 2.0
    assert s["mean"] == pytest.approx(13.0 / 3)


def test_timing_summary_drops_exactly_the_first_warmup_entries():
    s = timing_summary([1000.0, 500.0, 1.0, 2.0, 3.0, 4.0], warmup=2)
    assert s["n"] == 4 and s["max"] == 4.0 and s["min"] == 1.0
    assert timing_summary([7.0, 1.0], warmup=0)["n"] == 2
    assert timing_summary([7.0, 1.0, 3.0], warmup=1) == {"n": 2, "median": 2.0, "mean": 2.0, "min": 1.0, "max": 3.0}


def test_n_updates_counts_partial_last_update():
    assert n_updates(17, 16) == 2
    assert n_updates(16, 16) == 1
    assert n_updates(1, 1) == 1


def test_project_hours_units():
    assert project_hours(3600.0, 1) == 1.0
    assert project_hours(1.0, 3600) == 1.0
    assert project_hours(0.0, 10) == 0.0


@pytest.mark.skipif(not REPORT.exists(), reason="report not committed")
def test_report_consistent_with_raw_update_timings():
    rep = json.loads(REPORT.read_text())
    assert rep["accuracy_reported"] is False
    blob = json.dumps({k: v for k, v in rep.items() if k != "accuracy_reported"}).lower()
    for bad in ('"acc', '"pred', '"correct', '"logit', "shotbench", "camerabench", "/gold", "/dev/"):
        assert bad not in blob, bad
    nu = n_updates(51_111, 16)
    assert nu == 3195
    for k, m in rep["models"].items():
        tr = m["train"]
        secs = [u["sec"] for u in tr["updates"]]  # rounded to 0.01 s in the report
        assert len(secs) == tr["warmup_updates"] + tr["measured_updates"] == 12
        kept = secs[tr["warmup_updates"]:]
        assert abs(statistics.median(kept) - tr["sec_per_update"]["median"]) < 0.006, k
        assert abs(statistics.fmean(kept) - tr["sec_per_update"]["mean"]) < 0.006, k
        assert abs(max(kept) - tr["sec_per_update"]["max"]) < 0.006 and abs(min(kept) - tr["sec_per_update"]["min"]) < 0.006, k
        assert tr["peak_gb"] == max(u["peak_gb"] for u in tr["updates"])
        assert all(u["finite"] for u in tr["updates"])
        v = m["val"]["sec_per_question"]
        assert v["n"] == 48 and v["min"] <= v["median"] <= v["max"]
        p = rep["projections"][k]
        assert p["train_updates_1epoch"] == nu and p["full_val_questions"] == 5822
        assert math.isclose(p["train_1epoch_hours"], tr["sec_per_update"]["median"] * nu / 3600, rel_tol=1e-12)
        assert math.isclose(p["full_val_hours"], v["median"] * 5822 / 3600, rel_tol=1e-12)
        assert m["lora"]["rank"] == 16 and m["lora"]["scale"] == 2.0 and m["leak_probe"]["without_mask"] == 0.0
    a, b = rep["models"]["4b"], rep["models"]["2b"]
    r = rep["ratio_4b_over_2b"]
    assert math.isclose(r["sec_per_update"], a["train"]["sec_per_update"]["median"] / b["train"]["sec_per_update"]["median"])
    assert math.isclose(r["sec_per_question"], a["val"]["sec_per_question"]["median"] / b["val"]["sec_per_question"]["median"])
    assert math.isclose(r["train_peak_gb"], a["train"]["peak_gb"] / b["train"]["peak_gb"])
