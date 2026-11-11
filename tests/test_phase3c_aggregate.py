"""CPU tests for the Phase 3C aggregation logic (synthetic inputs, no model). docs/phase3c_plan.md sections 2, 3, 5."""
import math

import numpy as np
import pytest

from cgt.p3c_tokens import aggregate as ag
from cgt.p3c_tokens.vocab import ATTRS, labels, option_prompt

SIZES = {"size": 7, "lens": 4, "angle": 5, "framing": 7, "light_type": 12, "light_cond": 10, "composition": 6}


def test_vocab_sizes_match_schema():
    assert {a: len(labels(a)) for a in ATTRS} == SIZES


def test_prompt_is_the_shotqa_template():
    got = option_prompt("lens", ["wide", "medium", "ultra-wide", "long"])
    assert got == ("Question: What's the lens size of this shot?\nOptions:\nA. Wide\nB. Medium\nC. Ultra Wide / Fisheye\nD. Long Lens\n"
                   "Please select the most likely answer from the options above. \n")


@pytest.mark.parametrize("n", sorted(set(SIZES.values())))
@pytest.mark.parametrize("r", [1, 2, 3])
def test_windows_cover_every_label(n, r):
    w = ag.make_windows(n, r)
    assert len(w) == (1 if n == 4 else math.ceil(n * r / 4))
    assert all(len(x) == 4 and len(set(x)) == 4 for x in w)
    cover = np.bincount([i for x in w for i in x], minlength=n)
    assert cover.min() >= (1 if n == 4 else max(1, min(r, len(w) - 1) if r > 1 else 1))
    if r >= 2 and n > 4:
        assert cover.min() >= 2


def test_pass_counts_in_plan():
    assert [len(ag.make_windows(n, 2)) for n in SIZES.values()] == [4, 1, 3, 4, 6, 5, 3]
    assert [len(ag.make_windows(n, 3)) for n in SIZES.values()] == [6, 1, 4, 6, 9, 8, 5]
    assert sum(len(ag.make_windows(n, 2)) for n in SIZES.values()) == 26


def _oracle_logq(true_scores):
    """A noiseless 'model': answers every window with softmax of the true per-label scores."""
    return lambda w: ag.log_softmax(np.array([true_scores[i] for i in w]))


@pytest.mark.parametrize("n", [5, 7, 10, 12])
@pytest.mark.parametrize("agg", ["mean", "bt"])
def test_aggregation_recovers_the_top_label_from_a_consistent_model(n, agg):
    rng = np.random.default_rng(n)
    for _ in range(20):
        truth = rng.normal(size=n) * 2
        wins = ag.make_windows(n, 2)
        ask = _oracle_logq(truth)
        logq = [ask(w) for w in wins]
        s = ag.agg_mean(n, wins, logq) if agg == "mean" else ag.agg_bt(n, wins, logq)
        # the dominant label is recovered when it clearly leads
        order = np.argsort(-truth)
        if truth[order[0]] - truth[order[1]] > 1.0:
            assert int(np.argmax(s)) == int(order[0])


def test_bt_recovers_scores_up_to_a_constant():
    truth = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 0.5, -0.5])
    wins = ag.make_windows(7, 3)
    s = ag.agg_bt(7, wins, [_oracle_logq(truth)(w) for w in wins], steps=2000)
    assert np.allclose(s, truth - truth.mean(), atol=0.05)


def test_mean_aggregation_by_hand():
    wins = [[0, 1, 2, 3], [2, 3, 4, 0]]
    lq = [np.log([0.4, 0.3, 0.2, 0.1]), np.log([0.5, 0.2, 0.2, 0.1])]
    s = ag.agg_mean(5, wins, lq)
    assert s[0] == pytest.approx((math.log(0.4) + math.log(0.1)) / 2)
    assert s[1] == pytest.approx(math.log(0.3))
    assert s[4] == pytest.approx(math.log(0.2))
    with pytest.raises(ValueError):
        ag.agg_mean(6, wins, lq)


def test_tournament_returns_the_dominant_label_and_counts_asks():
    # upper bounds: the final round is skipped when one label wins every round-1 window (it can be in two windows when n is not a multiple of 4)
    for n, expect_calls in [(4, 1), (5, 3), (7, 3), (12, 4)]:
        truth = np.zeros(n)
        truth[n - 2] = 5.0
        s, calls = ag.tournament(n, _oracle_logq(truth))
        assert int(np.argmax(s)) == n - 2
        assert 1 <= calls <= expect_calls
        assert s[n - 2] > -1.0 and (np.delete(s, n - 2) < 0.0).all()


def test_tournament_score_argmax_is_the_final_winner():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.choice([5, 6, 7, 10, 12]))
        truth = rng.normal(size=n)
        s, _ = ag.tournament(n, _oracle_logq(truth))
        # with a consistent model the knockout winner is the global argmax
        assert int(np.argmax(s)) == int(np.argmax(truth))


def test_label_set_and_alpha():
    s = np.log([0.5, 0.3, 0.15, 0.05])
    assert ag.label_set(s, 1.0) == [0]
    assert ag.label_set(s, 0.5) == [0, 1]
    assert ag.label_set(s, 0.25) == [0, 1, 2]
    assert ag.label_set(np.zeros(4), 1.0)[0] in range(4)  # ties keep a primary label


def test_set_f1_and_pick_alpha_tie_prefers_larger_alpha():
    assert ag.set_f1({1}, {1}) == 1.0 and ag.set_f1({1, 2}, {1}) == pytest.approx(2 / 3) and ag.set_f1({3}, {1}) == 0.0
    scores = [np.log([0.9, 0.05, 0.03, 0.02])] * 3
    a, f1 = ag.pick_alpha(scores, [{0}, {0}, {0}])
    assert a == 1.0 and f1 == 1.0
    a, f1 = ag.pick_alpha([np.log([0.5, 0.4, 0.05, 0.05])] * 4, [{0, 1}] * 4)
    assert a in (0.75, 0.5, 0.25) and f1 == 1.0


def test_option_scores_average_members():
    s = np.array([-1.0, -2.0, -3.0, -4.0])
    out = ag.option_scores(s, {"A": [0], "B": [1, 2]})
    assert out == {"A": -1.0, "B": -2.5}


def test_top_and_margin():
    p1, m = ag.top_and_margin(np.log([0.6, 0.3, 0.1]))
    assert p1 == pytest.approx(0.6) and m == pytest.approx(0.3)


def test_pick_alpha_with_label_names():
    """Regression: gold sets are label NAMES, predicted sets come from score indices (an earlier version compared indices to names -> F1 = 0)."""
    names = ["a", "b", "c", "d"]
    a, f1 = ag.pick_alpha([np.log([0.5, 0.4, 0.05, 0.05])] * 3, [{"a", "b"}] * 3, names)
    assert f1 == 1.0 and a == 0.75
    a, f1 = ag.pick_alpha([np.log([0.9, 0.05, 0.03, 0.02])] * 3, [{"a"}] * 3, names)
    assert f1 == 1.0 and a == 1.0
