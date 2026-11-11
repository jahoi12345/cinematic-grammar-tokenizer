"""Independent synthetic checks of phase3c_interface.md and its referenced plan.

No images, models, validation caches or holdout labels are used in this file.
"""
import numpy as np
import pytest

from cgt.p3c_tokens import aggregate as ag
from cgt.p3c_tokens import methods, selection
from cgt.p3c_tokens.vocab import ATTRS, DISPLAY, MULTI, labels


def test_window_starts_and_letter_rotation_are_the_registered_rule():
    assert ag.make_windows(7, 2) == [
        [0, 1, 2, 3], [2, 3, 4, 1], [5, 6, 3, 4], [1, 5, 6, 0],
    ]
    assert ag.make_windows(5, 1) == [[0, 1, 2, 3], [3, 4, 0, 2]]
    # Four-label lens keeps a single pass even for coverage 3.
    assert ag.make_windows(4, 3) == [[0, 1, 2, 3]]


def test_mean_aggregation_weights_occurrences_and_keeps_schema_order():
    wins = [[0, 1, 2, 3], [4, 2, 0, 3]]
    logq = [np.log([.4, .3, .2, .1]), np.log([.1, .2, .6, .1])]
    expected = np.log([np.sqrt(.4 * .6), .3, .2, .1, .1])
    np.testing.assert_allclose(ag.agg_mean(5, wins, logq), expected, atol=1e-12)
    with pytest.raises(ValueError):
        ag.agg_mean(6, wins, logq)


def test_choice_fit_solves_the_regularized_objective_and_centres_scores():
    q = np.array([.50, .30, .15, .05])
    theta = ag.agg_bt(4, [[0, 1, 2, 3]], [np.log(q)])
    p = np.exp(theta - theta.max())
    p /= p.sum()
    # Stationarity of KL(q||softmax(theta)) + .001 * ||theta||^2.
    np.testing.assert_allclose(p - q + .002 * theta, 0, atol=1e-5)
    assert theta.mean() == pytest.approx(0, abs=1e-12)
    assert list(np.argsort(-theta)) == [0, 1, 2, 3]


def test_tournament_final_padding_uses_runners_and_final_winner():
    expected_windows = [[0, 1, 2, 3], [4, 5, 6, 3], [0, 5, 6, 1]]
    probabilities = [[.7, .15, .1, .05], [.1, .6, .2, .1], [.1, .2, .6, .1]]
    seen = []

    def ask(window):
        pos = len(seen)
        assert window == expected_windows[pos]
        seen.append(window)
        return np.log(probabilities[pos])

    score, passes = ag.tournament(7, ask)
    assert passes == len(seen) == 3
    assert int(np.argmax(score)) == 6
    np.testing.assert_allclose(score[[0, 5, 6, 1]], np.log(probabilities[-1]))
    np.testing.assert_allclose(score[[2, 3, 4]], -100 + np.log([.1, .1, .1]))


def test_tournament_skips_final_when_all_windows_have_one_winner():
    calls = []

    def ask(window):
        calls.append(window)
        raw = np.array([5. if i == 3 else 0. for i in window])
        return raw - np.log(np.exp(raw).sum())

    scores, passes = ag.tournament(5, ask)
    assert passes == len(calls) == 2
    assert scores[3] == 0 and int(np.argmax(scores)) == 3
    assert np.delete(scores, 3).max() <= -95


def test_D_scores_all_labels_in_schema_order_with_one_letter_pass():
    class FakeEngine:
        passes = 9

        def letter_logits(self, state, user, n):
            assert state == "synthetic-state"
            names = labels("light_type")
            assert n == 12 == len(names)
            option_lines = [line for line in user.splitlines() if line[:1] in "ABCDEFGHIJKL" and line[1:3] == ". "]
            assert option_lines == [
                f"{letter}. {DISPLAY['light_type'][name]}"
                for letter, name in zip("ABCDEFGHIJKL", names)
            ]
            self.passes += 1
            return np.arange(n, dtype=float) / 8

    scores, passes = methods.score_attribute(FakeEngine(), "synthetic-state", "light_type", "D")
    raw = np.arange(12, dtype=float) / 8
    np.testing.assert_allclose(scores, raw - np.log(np.exp(raw).sum()), atol=1e-12)
    assert scores.dtype == np.float64 and passes == 1


def test_alpha_inclusive_threshold_orders_ties_by_schema_index():
    scores = np.log([.5, .25, .125, .125])
    assert ag.label_set(scores, .5) == [0, 1]
    assert ag.label_set(scores, .25) == [0, 1, 2, 3]
    # The formal >= rule retains all equal maxima, even at alpha=1.
    assert ag.label_set(np.log([.4, .4, .2]), 1.) == [0, 1]


def test_alpha_uses_label_names_and_prefers_larger_alpha_on_F1_tie():
    names = ["one", "two", "three", "four"]
    scores = [np.log([.5, .4, .05, .05])]
    alpha, f1 = ag.pick_alpha(scores, [{"one", "two"}], names)
    assert (alpha, f1) == (.75, 1.)
    alpha, f1 = ag.pick_alpha([np.log([.9, .05, .03, .02])], [{"one"}], names)
    assert (alpha, f1) == (1., 1.)


def test_option_restriction_averages_log_scores_and_not_probabilities():
    scores = np.array([-.2, -1., -2., -5.])
    out = ag.option_scores(scores, {"A": [0, 3], "B": [1], "C": [2]})
    assert out == {"A": -2.6, "B": -1., "C": -2.}
    assert max(out, key=out.get) == "B" and np.argmax(scores) == 0


def test_selection_tie_band_is_inclusive_absolute_pp_then_passes_then_m():
    boundary = selection.select({"A3m": {"composite": .73}, "D": {"composite": .72}})
    assert boundary["within_tie_band"] == ["A3m", "D"] and boundary["selected"] == "D"
    outside = selection.select({"A3m": {"composite": .73}, "D": {"composite": .7199}})
    assert outside["within_tie_band"] == ["A3m"] and outside["selected"] == "A3m"
    tied_aggregation = selection.select({"A2b": {"composite": .746}, "A2m": {"composite": .741}})
    assert tied_aggregation["top_by_composite"] == "A2b"
    assert tied_aggregation["selected"] == "A2m"


def test_metric_join_keeps_same_id_different_attributes_and_named_gold_sets():
    samples, scores = [], {}
    for attr in ATTRS:
        names = labels(attr)
        samples.append({"id": "duplicate#0", "attribute": attr,
                        "options": {"A": DISPLAY[attr][names[0]]}, "answer": "A"})
        vector = np.full(len(names), -10.)
        vector[1 if attr == "lens" else 0] = 0.
        scores[("duplicate#0", attr)] = vector
    result = selection.evaluate_method(samples, scores)
    per = result["per_attribute"]
    assert all(p["n"] == 1 for p in per.values())
    assert per["size"]["full_acc"] == 1 and per["lens"]["full_acc"] == 0
    for attr in MULTI:
        assert per[attr]["set_f1"] == 1 and per[attr]["alpha"] == 1
    assert result["composite"] == pytest.approx(6 / 7)
