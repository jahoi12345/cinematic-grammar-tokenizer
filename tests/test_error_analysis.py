import numpy as np
import pytest

from cgt.error_analysis import core as E


def test_gold_rank_ties_follow_pick_option():
    assert E.gold_rank([3, 1, 2, 0], 0) == 1
    assert E.gold_rank([3, 1, 2, 0], 2) == 2
    assert E.gold_rank([3, 1, 2, 0], 3) == 4
    assert E.gold_rank([2, 2, 1, 0], 0) == 1   # tie: lowest letter wins
    assert E.gold_rank([2, 2, 1, 0], 1) == 2


def _r(g, p, film="f1"):
    g, p = frozenset(g), frozenset(p)
    return {"gold": g, "pred": p, "gold_text": ", ".join(sorted(g)), "pred_text": ", ".join(sorted(p)), "film": film,
            "correct": g == p}


def test_component_prf_and_confusions():
    rs = [_r({"hard", "side"}, {"hard", "side"}), _r({"hard"}, {"soft"}), _r({"soft"}, {"soft"}), _r({"hard", "top"}, {"hard", "side"})]
    m = E.component_prf(rs, ["hard", "soft", "side", "top", "edge"])
    assert m["hard"] == {"support": 3, "predicted": 2, "recall": 2 / 3, "precision": 1.0}
    assert m["soft"]["recall"] == 1.0 and m["soft"]["precision"] == 0.5
    assert m["edge"] == {"support": 0, "predicted": 0, "recall": None, "precision": None}
    c = {(x["gold"], x["pred"]): x["count"] for x in E.component_confusions(rs)}
    assert c == {("hard", "soft"): 1, ("top", "side"): 1}
    o = E.option_confusions(rs)
    assert {(x["gold"], x["pred"]) for x in o} == {("hard", "soft"), ("hard, top", "hard, side")}


def test_component_confusions_none_side():
    rs = [_r({"hard", "side"}, {"hard"})]
    assert E.component_confusions(rs) == [{"gold": "side", "pred": "(none)", "count": 1}]


def test_film_concentration():
    films = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    correct = [False] * 6 + [True] * 4 + [True] * 9 + [False] + [True] * 10
    o = E.film_concentration(films, correct, min_q=10)
    assert o["errors"] == 7 and o["films_with_errors"] == 2
    assert o["top5_films_error_share"] == 1.0
    assert o["films_holding_half_of_errors"] == 1
    assert o["per_film_accuracy"]["min"] == pytest.approx(0.4) and o["per_film_accuracy"]["max"] == 1.0
    assert o["dispersion_ratio"] > 1
    even = E.film_concentration(["a"] * 10 + ["b"] * 10, ([False] * 2 + [True] * 8) * 2)
    assert even["dispersion_ratio"] == pytest.approx(0.0)


def test_auc():
    assert E.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert E.auc([1, 2, 3, 4], [True, True, False, False]) == 0.0
    assert E.auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert E.auc([1, 2], [True, True]) is None


def test_accuracy_by_quantile():
    b = E.accuracy_by_quantile(list(range(10)), [False] * 5 + [True] * 5, q=2)
    assert [x["n"] for x in b] == [5, 5] and [x["accuracy"] for x in b] == [0.0, 1.0]


def test_image_stats_synthetic():
    img = np.zeros((40, 80, 3), np.uint8)
    img[:, 60:] = 255   # one bright block on the right: gradient mass at x = 60/80
    s = E.image_stats(img)
    assert s["mean_luma"] == pytest.approx(0.25, abs=1e-6)
    assert s["dark_frac"] == pytest.approx(0.75) and s["bright_frac"] == pytest.approx(0.25)
    assert s["grad_cx"] > 0.5 and s["lr_balance"] == pytest.approx(1.0)
    assert s["aspect"] == 2.0 and s["saturation"] == 0.0 and s["warmth"] == 0.0
    flat = E.image_stats(np.full((10, 10, 3), 128, np.uint8))
    assert flat["rms_contrast"] == pytest.approx(0.0) and flat["sharpness"] == pytest.approx(0.0) and flat["grad_cx"] == 0.5 and flat["lr_balance"] == 0.0
    red = E.image_stats(np.dstack([np.full((4, 4), 200, np.uint8), np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)]))
    assert red["saturation"] == 1.0 and red["warmth"] > 0


def test_group_medians():
    assert E.group_medians([1, 3, 5, 7], ["a", "a", "b", "b"]) == {"a": {"n": 2, "median": 2.0}, "b": {"n": 2, "median": 6.0}}
