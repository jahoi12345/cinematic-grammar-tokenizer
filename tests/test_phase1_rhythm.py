"""Shot duration calculations in seconds."""

import numpy as np
import pytest

from cgt.p1_shots_color.rhythm import asl, shot_lengths


def test_lengths_and_average_from_contract_example():
    shots = [(0, 2), (2, 5), (5, 6)]
    lengths = shot_lengths(shots)
    assert isinstance(lengths, np.ndarray)
    assert lengths.shape == (3,)
    np.testing.assert_allclose(lengths, [2, 3, 1])
    assert asl(shots) == pytest.approx(2.0)


def test_fractional_lengths_with_nonzero_start():
    shots = [(10.0, 10.25), (10.25, 11.75)]
    np.testing.assert_allclose(shot_lengths(shots), [0.25, 1.5])
    assert asl(shots) == pytest.approx(0.875)


def test_single_one_frame_shot():
    shots = [(0.0, 1 / 24)]
    np.testing.assert_allclose(shot_lengths(shots), [1 / 24])
    assert asl(shots) == pytest.approx(1 / 24)
