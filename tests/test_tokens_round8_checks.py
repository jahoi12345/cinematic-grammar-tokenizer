"""Independent synthetic decision-0009 checks; no real films or fitted caches."""
import numpy as np
import pandas as pd
import pytest
import torch

from cgt.tokens import arc, style, v1
from cgt.tokens import palette_global as pg


def test_probability_vote_can_disagree_with_majority_and_is_shift_invariant():
    frames = [np.log([.99, .01]), np.log([.4, .6]), np.log([.4, .6])]
    expected = np.array([1.79, 1.21]) / 3
    p = v1.soft_vote(frames)
    np.testing.assert_allclose(p, expected, atol=1e-14)
    assert int(p.argmax()) == 0  # Majority of per-frame argmaxes is class 1.
    shifted = [f + shift for f, shift in zip(frames, [1000., -1000., 50.], strict=True)]
    np.testing.assert_allclose(v1.soft_vote(shifted), expected, atol=1e-13)


def test_all_seven_length_bins_and_missing_slots():
    sec = [0., 1., 2., 4., 8., 16., 32.]
    assert [v1.len_bin(s) for s in sec] == list(range(7))
    assert [v1.len_token(s) for s in sec] == ["<1s", "1-2s", "2-4s", "4-8s", "8-16s", "16-32s", "32+s"]
    slots = v1.parse_row(v1.format_row({"people": None, "len_s": np.nan}))
    assert list(slots) == ["size", "lens", "angle", "framing", "light", "comp", "key", "DoF", "people", "sound", "len", "palette", "move"]
    assert slots["light"] == "?/?" and slots["move"] == "n/a"
    assert all(slots[k] == "?" for k in slots if k not in ["light", "move"])
    assert "style" not in slots and "eye" not in slots.values()


def test_global_palette_uses_mean_lightness_and_lowest_tie_id():
    # The first (dominant) colour alone would order these centroids backwards.
    centers = np.zeros((2, 15))
    centers[0, ::3] = [90., 0., 0., 0., 0.]
    centers[1, ::3] = [10., 80., 80., 80., 80.]
    assert pg.lightness_rank(centers).tolist() == [0, 1]
    mid = centers.mean(axis=0)[None]
    assert pg.assign(mid, centers).tolist() == [0]


def test_arc_length_interval_is_iqr_and_partial_rate_is_shot_start_rate():
    d = pd.DataFrame({"start_min": [0., 0.2, 0.4, 0.6], "end_s": [12., 24., 36., 60.], "len_s": [1., 2., 4., 9.],
                      "size": "MS", "key": "mid-key", "people": 0, "sound": "score", "light_cond": "hard", "light_type": "daylight", "angle": "low", "kf_all_agree": True})
    rows = arc.series_table(d, n_boot=10).set_index("series")
    assert rows.loc["median_len_s", "value"] == 3.
    assert rows.loc["median_len_s", "ci_lo"] == pytest.approx(1.75)
    assert rows.loc["median_len_s", "ci_hi"] == pytest.approx(5.25)
    assert rows.loc["cuts_per_min", "value"] == 4.  # Includes the film's first shot start.


def test_style_statistics_and_frozen_encoding_do_not_update_codebooks():
    codes = np.column_stack([[0, 0, 1, 1], [0, 0, 0, 0]])
    stats = style.usage_stats(codes)["levels"]
    assert stats[0]["used"] == 2 and stats[0]["perplexity"] == pytest.approx(2.)
    assert stats[0]["max_share"] == .5 and stats[1]["max_share"] == 1.
    torch.manual_seed(0)
    model = style.RQVAE(d_in=4, hidden=8, d_lat=2, levels=2, codes=3)
    before = {k: x.clone() for k, x in model.state_dict().items()}
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    first = model.encode(x)
    assert np.array_equal(first, model.encode(x))
    assert all(torch.equal(before[k], x) for k, x in model.state_dict().items())
