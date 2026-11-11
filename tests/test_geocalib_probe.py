import numpy as np
import pytest

from cgt.geocalib_probe import core as G


def _img(h=100, w=200, val=128):
    return np.full((h, w, 3), val, np.uint8)


def test_crop_letterbox_rows():
    im = _img()
    im[:12] = 0
    im[-10:] = 5
    out, info = G.crop_black_bars(im)
    assert out.shape == (78, 200, 3)
    assert (info["top"], info["bottom"], info["left"], info["right"]) == (12, 10, 0, 0)
    assert info["cropped"] and not info["refused"]


def test_crop_pillarbox_columns_and_none():
    im = _img()
    im[:, :20] = 0
    out, info = G.crop_black_bars(im)
    assert out.shape == (100, 180, 3) and info["left"] == 20 and info["right"] == 0
    out, info = G.crop_black_bars(_img())
    assert out.shape == (100, 200, 3) and not info["cropped"]


def test_crop_keeps_dark_row_with_bright_detail():
    im = _img()
    im[:5] = 0
    im[2, 100:110] = 255      # bright detail: p99 > threshold, so row 2 is not a bar and the run stops at 2
    _, info = G.crop_black_bars(im)
    assert info["top"] == 2


def test_crop_side_cap_and_refusal():
    im = _img()
    im[:45] = 0                # more than 30% of the height: capped at 30 rows
    _, info = G.crop_black_bars(im)
    assert info["top"] == 30 and info["capped"]
    dark = _img(val=0)         # all black: both sides capped at 30%, 40% kept (refusal is only a safety net)
    out, info = G.crop_black_bars(dark)
    assert info["capped"] and not info["refused"] and out.shape == (40, 80, 3)
    _, info = G.crop_black_bars(_img())
    assert not info["capped"]


def test_fov_band_edges():
    assert G.fov_band(100) == "ultra_wide"
    assert G.fov_band(84.0) == "wide" and G.fov_band(84.01) == "ultra_wide"
    assert G.fov_band(54.0) == "wide" and G.fov_band(53.99) == "medium"
    assert G.fov_band(29.0) == "medium" and G.fov_band(28.99) == "long"
    with pytest.raises(ValueError):
        G.fov_band(float("nan"))


def test_balanced_sample_deterministic_and_order_free():
    by = {c: [f"{k}{i:03d}" for i in range(200)] for k, c in enumerate(G.CLASSES)}
    a = G.balanced_sample(by, per_class=10)
    b = G.balanced_sample({c: list(reversed(v)) for c, v in by.items()}, per_class=10)
    assert a == b and all(len(v) == 10 and len(set(v)) == 10 for v in a.values())
    with pytest.raises(ValueError):
        G.balanced_sample(by, per_class=300)


def test_auc_and_bootstrap():
    assert G.auc([3, 4, 1, 2], [True, True, False, False]) == 1.0
    assert G.auc([1, 1], [True, False]) == 0.5
    assert G.auc([1, 2], [True, True]) is None
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(1, 1, 100), rng.normal(0, 1, 100)]
    y = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    lo, hi = G.cluster_bootstrap_auc(x, y, [f"f{i % 20}" for i in range(200)], n=200)
    assert lo < G.auc(x, y) < hi


def test_reading_rule():
    p = lambda a, lo, g: {"hfov_auc": a, "hfov_ci": (lo, a + 0.05), "cv_gain": g}
    assert G.reading({"wide_vs_medium": p(0.70, 0.60, 0.05), "medium_vs_long": p(0.5, 0.4, 0.0)})["verdict"] == "yes"
    assert G.reading({"wide_vs_medium": p(0.70, 0.60, 0.02), "medium_vs_long": p(0.5, 0.4, 0.0)})["verdict"] == "unclear"
    assert G.reading({"wide_vs_medium": p(0.55, 0.45, 0.0), "medium_vs_long": p(0.57, 0.52, 0.005)})["verdict"] == "no"
    assert G.reading({"wide_vs_medium": p(0.60, 0.52, 0.0), "medium_vs_long": p(0.5, 0.4, 0.0)})["verdict"] == "unclear"
