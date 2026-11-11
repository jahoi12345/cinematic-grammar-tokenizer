"""Time selection only; distinct times need not decode to distinct frames."""

import math

import pytest

from cgt.p1_shots_color.keyframes import pick_keyframe_times


@pytest.mark.parametrize("start,end", [(0.0, 2.0), (7.5, 10.5), (0.0, 1 / 24), (10.0, 10 + 1 / 24)])
def test_three_increasing_keyframe_times_strictly_inside_shot(start, end):
    times = pick_keyframe_times(start, end)
    assert isinstance(times, tuple)
    assert len(times) == 3
    assert all(math.isfinite(time) for time in times)
    assert start < times[0] < times[1] < times[2] < end


def test_middle_time_is_shot_midpoint():
    times = pick_keyframe_times(7.5, 10.5)
    assert times[1] == pytest.approx(9.0)
