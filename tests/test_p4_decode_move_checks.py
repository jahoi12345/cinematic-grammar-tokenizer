"""an earlier review extra checks for the Step A movement decoders (docs/acc_step_a_move_plan.md): gaps found by verification pass mutations. Synthetic data only."""
import numpy as np
import pytest

M = pytest.importorskip("cgt.p4_decode.move", exc_type=ImportError)

ATOMS = ["static", "push-in", "pull-out", "boom-up", "boom-down", "pan-left", "pan-right", "tilt-up", "tilt-down", "zoom-in", "zoom-out", "move-left", "move-right", "arc", "roll"]
AI = {a: i for i, a in enumerate(ATOMS)}


def vec(**kw) -> np.ndarray:
    p = np.full(len(ATOMS), 0.1)
    for k, v in kw.items():
        p[AI[k.replace("_", "-")]] = v
    return p


def test_d1_static_tie_goes_to_static():
    # plan section 2: "if p(static) >= max p over the remaining movement atoms, output {static}"
    assert M.decode(np.array([vec(static=0.7, arc=0.7)]), ATOMS, "D1") == [["static"]]


def test_d3_constraints_use_recalibrated_q_not_p():
    # p: pan-right > pan-left; q (after thresholds) reverses the order -> D3 must keep pan-left (plan: "D1's constraints using q instead of p")
    th = np.full(len(ATOMS), 0.5)
    th[AI["pan-left"]], th[AI["pan-right"]] = 0.3, 0.65
    P = np.array([vec(pan_left=0.6, pan_right=0.7)])
    assert M.decode(P, ATOMS, "D1") == [["pan-right"]]
    assert M.decode(P, ATOMS, "D3", th) == [["pan-left"]]


def test_paired_bootstrap_resamples_whole_sources():
    # two sources; per-source differences constant (1 and 0): every source resample has mean difference in {0, 0.5, 1}
    x = np.array([1.0] * 10 + [0.0] * 10)
    y = np.zeros(20)
    src = np.array(["a"] * 10 + ["b"] * 10)
    b = M.paired_bootstrap(x, y, src)
    assert b["diff"] == 0.5
    assert set(b["ci95"]) <= {0.0, 0.5, 1.0}


def test_paired_bootstrap_matches_p4_video_validate_protocol():
    rng = np.random.default_rng(7)
    src = np.array([f"s{k}" for k in rng.integers(0, 9, 120)])
    x, y = rng.integers(0, 2, 120).astype(float), rng.integers(0, 2, 120).astype(float)
    usrc = sorted(set(src))
    r = np.random.default_rng(0)
    d = []
    for _ in range(1000):
        ix = np.concatenate([np.where(src == s)[0] for s in r.choice(usrc, len(usrc), replace=True)])
        d.append(x[ix].mean() - y[ix].mean())
    ref = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    assert M.paired_bootstrap(x, y, src)["ci95"] == ref


def test_fit_thresholds_equidistant_tie_goes_to_smaller():
    # F1 = 2/3 for every t <= 0.40 and for 0.60 <= t <= 0.90, 0.4 in between; 0.40 and 0.60 are exactly equidistant from 0.5 -> the smaller (plan section 2)
    Pc = np.zeros((4, len(ATOMS)))
    Yc = np.zeros((4, len(ATOMS)), dtype=bool)
    Pc[:, 0] = [0.42, 0.9, 0.57, 0.58]
    Yc[:, 0] = [True, True, False, False]
    assert M.fit_thresholds(Pc, Yc)[0] == 0.4
