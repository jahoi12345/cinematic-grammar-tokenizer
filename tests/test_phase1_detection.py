"""Shot boundary contract using lossless clips generated in tmp_path."""

from itertools import pairwise

import pytest

from cgt.p1_shots_color.detection import detect_shots


def assert_partition(shots, duration, fps=24):
    assert isinstance(shots, list)
    assert shots
    for shot in shots:
        assert isinstance(shot, tuple)
        assert len(shot) == 2
        start, end = shot
        assert 0 <= start < end
    assert shots[0][0] == pytest.approx(0.0, abs=1e-9)
    assert shots[-1][1] == pytest.approx(duration, abs=1 / fps)
    for previous, current in pairwise(shots):
        assert previous[1] == pytest.approx(current[0], abs=1e-9)


def test_three_solid_segments_have_two_correct_cuts(clip_factory):
    # Contract: brightness contrasts must exceed ContentDetector's default threshold.
    # Equal-brightness red/green/blue hue changes are below that threshold.
    path = clip_factory([((0, 0, 200), 48), ((20, 20, 20), 72), ((250, 250, 250), 24)])
    shots = detect_shots(str(path))
    assert_partition(shots, duration=6)
    assert len(shots) == 3
    for actual, expected in zip(shots, [(0, 2), (2, 5), (5, 6)]):
        assert actual == pytest.approx(expected, abs=1 / 24)


def test_constant_clip_is_one_shot(clip_factory):
    path = clip_factory([((128, 128, 128), 48)])
    shots = detect_shots(str(path))
    assert_partition(shots, duration=2)
    assert len(shots) == 1


def test_similar_grey_cut_documents_default_threshold(clip_factory, record_property):
    """Record count without prescribing unspecified sensitivity to similar greys."""
    path = clip_factory([((120, 120, 120), 48), ((125, 125, 125), 48)])
    shots = detect_shots(str(path))
    assert_partition(shots, duration=4)
    record_property("similar_grey_default_shot_count", len(shots))
    record_property("similar_grey_default_intervals", repr(shots))
