"""Independently computed option scores and full-vocabulary toy accuracy."""
import numpy as np
import pytest

from cgt.eval import metrics
from cgt.p3_static_attrs.labels import Example


def test_restricted_accuracy_differs_from_full_vocabulary():
    v = ['outside', 'correct', 'other', 'third', 'fourth']
    ex = Example('synthetic.png', 'size',
                 dict(zip('ABCD', [frozenset([c]) for c in v[1:]], strict=True)),
                 'A', frozenset(['correct']))
    p = np.array([[.4, .25, .15, .1, .1]])
    assert metrics.option_restricted_accuracy([ex], p, v, 'categorical') == 1
    assert metrics.hit_at_1(p, np.array([[0, 1, 0, 0, 0]])) == 0


def test_set_likelihood_mean_probability_and_letter_ties():
    z = np.array([1., -2., .5])
    sets = [[0], [0, 2], [1], [2]]
    p = 1 / (1 + np.exp(-z))
    expected = [sum(np.log(p[i] if i in s else 1-p[i]) for i in range(3)) for s in sets]
    np.testing.assert_allclose(metrics.multihot_option_scores(z, sets), expected)
    np.testing.assert_allclose(metrics.categorical_option_scores(p, sets),
                               [np.log(np.mean(p[s])) for s in sets])
    assert metrics.pick_option(np.array([2., 2., 0., 0.]), ['B', 'A', 'C', 'D']) == 'A'
    assert np.isfinite(metrics.multihot_option_scores(np.array([1000., -1000.]), [[0], [1]])).all()
    assert metrics.option_restricted_accuracy([], [], [], 'categorical') == 0
    with pytest.raises(ValueError):
        metrics.option_restricted_accuracy([], [z], ['a', 'b', 'c'])
