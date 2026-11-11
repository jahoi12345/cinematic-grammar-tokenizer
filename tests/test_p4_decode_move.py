"""Step A movement decoders (docs/acc_step_a_move_plan.md): synthetic probabilities only, no label store, no data file."""
import inspect

import numpy as np
import pytest

from cgt.p4_camera_move.baseline import option_scores
from cgt.p4_decode import move as M

ATOMS = ["static", "push-in", "pull-out", "boom-up", "boom-down", "pan-left", "pan-right", "tilt-up", "tilt-down", "zoom-in", "zoom-out", "move-left", "move-right", "arc", "roll"]
AI = {a: i for i, a in enumerate(ATOMS)}


def vec(**kw) -> np.ndarray:
    p = np.full(len(ATOMS), 0.1)
    for k, v in kw.items():
        p[AI[k.replace("_", "-")]] = v
    return p


def rand_P(n=400, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, size=(n, len(ATOMS)))


def test_copies_equal_film_move():
    film_move = pytest.importorskip("cgt.p4_camera_move.film_move", exc_type=ImportError)
    assert film_move.OPPOSITE_PAIRS == M.OPPOSITE_PAIRS
    for s in (["static"], ["static", "arc"], ["pan-left", "pan-right"], ["tilt-up"], ["zoom-in", "zoom-out", "roll"]):
        assert film_move.is_contradictory(s) == M.is_contradictory(s)


def test_d0_equals_film_move_predict():
    film_move = pytest.importorskip("cgt.p4_camera_move.film_move", exc_type=ImportError)
    P = rand_P()
    assert M.decode(P, ATOMS, "D0") == film_move.predict(P, ATOMS)[0]


def test_d0_boundary_is_inclusive():
    assert M.decode(np.array([vec(arc=0.5, roll=0.6)]), ATOMS, "D0") == [["arc", "roll"]]


def test_d0_rule_light():
    P = rand_P()
    for p, s in zip(P, M.decode(P, ATOMS, "D0"), strict=True):
        above = [a for a in ATOMS if p[AI[a]] >= 0.5]
        assert s == (above or [ATOMS[int(np.argmax(p))]])
    P2 = np.full((3, len(ATOMS)), 0.2)
    P2[:, 4] = 0.3
    assert M.decode(P2, ATOMS, "D0") == [["boom-down"]] * 3


@pytest.mark.parametrize("cand", ["D1", "D3"])
def test_constrained_never_contradictory_nor_empty(cand):
    P = rand_P(2000, 1)
    th = M.fit_thresholds(rand_P(300, 2), rand_P(300, 3) > 0.7)
    for s in M.decode(P, ATOMS, cand, th):
        assert s
        assert not M.is_contradictory(s)["contradictory"]


def test_d1_pair_keeps_higher_and_tie_first():
    assert M.decode(np.array([vec(pan_left=0.6, pan_right=0.9)]), ATOMS, "D1") == [["pan-right"]]
    assert M.decode(np.array([vec(pan_left=0.9, pan_right=0.6, tilt_up=0.7)]), ATOMS, "D1") == [["pan-left", "tilt-up"]]
    assert M.decode(np.array([vec(push_in=0.7, pull_out=0.7)]), ATOMS, "D1") == [["push-in"]]


def test_d1_static_rule():
    assert M.decode(np.array([vec(static=0.9, arc=0.6, roll=0.55)]), ATOMS, "D1") == [["static"]]
    assert M.decode(np.array([vec(static=0.6, arc=0.8, roll=0.55)]), ATOMS, "D1") == [["arc", "roll"]]
    # static compared after the pair resolution: the dropped pan-left (0.95) is not used
    assert M.decode(np.array([vec(static=0.9, pan_left=0.95, pan_right=0.97)]), ATOMS, "D1") == [["pan-right"]]
    assert M.decode(np.array([vec(static=0.9)]), ATOMS, "D1") == [["static"]]


def test_d1_only_removes_atoms():
    P = rand_P(500, 4)
    for a, b in zip(M.decode(P, ATOMS, "D0"), M.decode(P, ATOMS, "D1"), strict=True):
        assert set(b) <= set(a)


def test_fit_thresholds_uses_only_calibration_and_rules():
    rng = np.random.default_rng(5)
    Pc = rng.uniform(size=(200, len(ATOMS)))
    Yc = Pc > 0.8
    Yc[:, AI["roll"]] = False  # no positive -> 0.5
    t = M.fit_thresholds(Pc, Yc)
    assert t[AI["roll"]] == 0.5
    assert np.all((t >= 0.05) & (t <= 0.95))
    assert abs(t[AI["static"]] - 0.8) <= 0.051
    # the signature takes calibration arrays only; there is no selection argument and no label store
    assert list(inspect.signature(M.fit_thresholds).parameters) == ["P_cal", "Y_cal", "grid"]
    # changing selection data cannot matter; changing calibration data does
    t2 = M.fit_thresholds(Pc.copy(), Yc.copy())
    assert np.array_equal(t, t2)
    assert not np.array_equal(t, M.fit_thresholds(Pc, Pc > 0.3))


def test_fit_thresholds_tie_to_half():
    Pc = np.zeros((4, len(ATOMS)))
    Yc = np.zeros((4, len(ATOMS)), dtype=bool)
    Pc[:, 0] = [0.01, 0.02, 0.99, 0.98]
    Yc[:, 0] = [False, False, True, True]  # every t in [0.05, 0.95] is perfect
    assert M.fit_thresholds(Pc, Yc)[0] == 0.5


def test_recalibrate_maps_threshold_to_half_and_identity():
    P = rand_P(50, 6)
    t = np.full(len(ATOMS), 0.5)
    assert np.allclose(M.recalibrate(P, t), P)
    t[0] = 0.7
    Q = M.recalibrate(np.array([[0.7] * len(ATOMS)]), t)
    assert abs(Q[0, 0] - 0.5) < 1e-9
    assert Q[0, 1] == pytest.approx(0.7)
    # D2 with all thresholds 0.5 equals D0
    assert M.decode(P, ATOMS, "D2", np.full(len(ATOMS), 0.5)) == M.decode(P, ATOMS, "D0")


def test_d2_needs_thresholds():
    with pytest.raises(ValueError):
        M.decode(rand_P(2), ATOMS, "D2")
    with pytest.raises(ValueError):
        M.decode(rand_P(2), ATOMS, "D9")


def test_d2_threshold_effect():
    t = np.full(len(ATOMS), 0.5)
    t[AI["arc"]] = 0.8
    assert M.decode(np.array([vec(arc=0.7, tilt_up=0.6)]), ATOMS, "D2", t) == [["tilt-up"]]
    assert M.decode(np.array([vec(arc=0.7)]), ATOMS, "D2", t) == [["arc"]]  # fallback by q: arc q=0.37 > others


def q(opts, gold="A"):
    return {"option_atoms": opts, "gold_letter": gold}


def test_option_scores_d0_equal_phase4_and_mask():
    p = vec(static=0.9, pan_left=0.8)
    qu = q({"A": ("static",), "B": ("pan-left", "static"), "C": ("pan-left",), "D": ("pan-left", "pan-right")}, gold="B")
    s = M.option_geometry_scores(qu, p, ATOMS)
    ref = option_scores(qu, p, ATOMS)
    assert np.allclose(s, [ref[L] for L in "ABCD"])
    assert M.option_hits([qu], np.array([p]), ATOMS, "D0")[0]  # B is the most likely set
    sm = M.option_geometry_scores(qu, p, ATOMS, mask=True)
    assert np.isinf(sm[1]) and np.isinf(sm[3]) and np.isfinite(sm[0]) and np.isfinite(sm[2])
    assert not M.option_hits([qu], np.array([p]), ATOMS, "D1")[0]
    all_bad = q({"A": ("static", "arc"), "B": ("pan-left", "pan-right"), "C": ("tilt-up", "tilt-down"), "D": ("static", "roll")})
    assert np.all(np.isfinite(M.option_geometry_scores(all_bad, p, ATOMS, mask=True)))


def test_option_tie_lowest_letter():
    qu = q({"A": ("arc",), "B": ("arc",), "C": ("roll",), "D": ("static",)}, gold="B")
    assert not M.option_hits([qu], np.array([vec(arc=0.9)]), ATOMS, "D0")[0]


def test_combined_scores():
    qu = q({"A": ("arc",), "B": ("roll",), "C": ("static",), "D": ("pan-left",)}, gold="B")
    p = np.array([vec(arc=0.6, roll=0.4)])
    assert not M.option_hits([qu], p, ATOMS, "D0")[0]
    assert M.option_hits([qu], p, ATOMS, "D0", video_logits=[np.array([0, 30, 0, 0])], w_s=0.3, w_g=0.7)[0]


def test_fusion_weights_and_ties():
    sem = [np.log([0.7, 0.1, 0.1, 0.1])] * 5
    geo = [np.log([0.25, 0.25, 0.25, 0.25])] * 5
    ws, wg = M.fit_fusion_weights(sem, geo, [0] * 5)
    assert ws == 1.0 and wg == 0.0
    ws, wg = M.fit_fusion_weights(geo, geo, [0] * 5)  # flat: all equal -> smallest sum
    assert (ws, wg) == (0.0, 0.0)


def test_set_metrics():
    m = M.set_metrics([["static"], ["pan-left", "tilt-up"]], [("static",), ("pan-left",)], ATOMS)
    assert m["exact_match"] == 0.5 and m["jaccard"] == 0.75
    assert M.set_metrics([["pan-left"]], [("pan-left", "tilt-up")], ATOMS)["jaccard"] == 0.5
    assert m["n_atoms_supported"] == 2 and m["macro_f1_supported"] == 1.0
    assert m["contradictory_share"] == 0.0


def test_paired_bootstrap_and_decide():
    x = np.array([1, 1, 1, 1, 0, 1] * 10)
    y = np.array([0, 1, 0, 1, 0, 1] * 10)
    src = np.repeat(np.arange(12), 5)
    b = M.paired_bootstrap(x, y, src)
    assert b["diff"] > 0 and b["ci95"][0] > 0
    acc = {"D0": 0.7, "D1": 0.75, "D2": 0.75, "D3": 0.6}
    boot = {c: {"ci95": [0.01, 0.1]} for c in ("D1", "D2", "D3")}
    jac = {"D0": 0.5, "D1": 0.495, "D2": 0.3, "D3": 0.5}
    assert M.decide(acc, boot, jac) == {"top_ranked": "D1", "ci_excludes_zero_above": True, "jaccard_ok": True, "winner": "D1"}
    jac["D1"] = 0.48
    assert M.decide(acc, boot, jac)["winner"] == "D0"
    jac["D1"] = 0.5
    boot["D1"] = {"ci95": [0.0, 0.1]}
    assert M.decide(acc, boot, jac)["winner"] == "D0"  # interval touching 0 does not exclude it; no stepping down to D2
