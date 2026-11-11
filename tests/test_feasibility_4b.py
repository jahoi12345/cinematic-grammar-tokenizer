import pytest

from cgt.p3b_vlm.feasibility import n_updates, project_hours, timing_summary


def test_n_updates_full_epoch():
    assert n_updates(51_111, 16) == 3195
    assert n_updates(13_734, 16) == 859  # Step B B0 size
    assert n_updates(32, 16) == 2
    assert n_updates(0, 16) == 0
    with pytest.raises(ValueError):
        n_updates(10, 0)


def test_timing_summary_drops_warmup():
    s = timing_summary([100.0, 50.0, 2.0, 4.0, 3.0], warmup=2)
    assert s == {"n": 3, "median": 3.0, "mean": 3.0, "min": 2.0, "max": 4.0}
    with pytest.raises(ValueError):
        timing_summary([1.0, 2.0], warmup=2)


def test_project_hours():
    assert project_hours(8.7, 3195) == pytest.approx(7.72125)
    assert project_hours(0.27, 5822) == pytest.approx(0.4366, abs=1e-4)
    with pytest.raises(ValueError):
        project_hours(-1.0, 3)
