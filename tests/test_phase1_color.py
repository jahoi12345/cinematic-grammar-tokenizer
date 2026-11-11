"""Color measurements on known BGR uint8 inputs."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from cgt.p1_shots_color.color import color_stats, lighting_key, palette_lab, palette_shares


def assert_palette(frame, k):
    centers = palette_lab(frame, k=k)
    shares = palette_shares(frame, k=k)
    assert isinstance(centers, np.ndarray)
    assert isinstance(shares, np.ndarray)
    assert centers.shape == (k, 3)
    assert shares.shape == (k,)
    assert np.isfinite(centers).all()
    assert np.isfinite(shares).all()
    assert np.all(shares >= 0)
    assert np.all(shares <= 1)
    assert shares.sum() == pytest.approx(1.0)
    assert np.all(np.diff(shares) <= 1e-6)
    return centers, shares


def test_solid_red_palette(frame_factory):
    frame = frame_factory((0, 0, 255))
    centers, shares = assert_palette(frame, k=1)
    assert centers[0, 1] > 0  # Contract's red a* sanity check; encoding is unresolved.
    assert shares[0] == pytest.approx(1.0)


def test_two_equal_colors_have_equal_shares(frame_factory):
    frame = frame_factory((0, 0, 255))
    frame[:, frame.shape[1] // 2 :] = frame_factory((255, 0, 0), width=48)
    _, shares = assert_palette(frame, k=2)
    np.testing.assert_allclose(shares, [0.5, 0.5], atol=0.01)


def test_default_five_clusters_sorted_by_pixel_share(frame_factory):
    frame = frame_factory((0, 0, 255), width=100)
    frame[:, 40:70] = (255, 0, 0)
    frame[:, 70:85] = (0, 255, 0)
    frame[:, 85:95] = (0, 0, 0)
    frame[:, 95:] = (255, 255, 255)
    centers = palette_lab(frame)
    shares = palette_shares(frame)
    assert centers.shape == (5, 3)
    assert shares.shape == (5,)
    np.testing.assert_allclose(shares, [0.4, 0.3, 0.15, 0.1, 0.05], atol=0.01)
    # First two centers must match dominant red and blue, in that order.
    red = palette_lab(frame_factory((0, 0, 255)), k=1)[0]
    blue = palette_lab(frame_factory((255, 0, 0)), k=1)[0]
    np.testing.assert_allclose(centers[:2], [red, blue], atol=1.0)


@pytest.mark.parametrize("bgr", [(0, 0, 0), (255, 255, 255), (0, 0, 255), (128, 128, 128)])
def test_color_stats_keys_ranges_and_uniform_contrast(frame_factory, bgr):
    stats = color_stats(frame_factory(bgr))
    assert isinstance(stats, dict)
    assert {"sat_mean", "val_mean", "warmth", "contrast"} <= stats.keys()
    assert all(np.isfinite(stats[key]) for key in ("sat_mean", "val_mean", "warmth", "contrast"))
    assert 0 <= stats["sat_mean"] <= 1
    assert 0 <= stats["val_mean"] <= 1
    assert 0 <= stats["contrast"] <= 100
    assert stats["contrast"] == pytest.approx(0, abs=1e-6)


def test_warm_and_cool_have_signed_lab_warmth(frame_factory):
    warm = color_stats(frame_factory((0, 165, 255)))
    cool = color_stats(frame_factory((255, 0, 0)))
    assert warm["warmth"] > 0 > cool["warmth"]


def test_saturated_red_and_neutral_grey(frame_factory):
    red = color_stats(frame_factory((0, 0, 255)))
    grey = color_stats(frame_factory((128, 128, 128)))
    assert red["sat_mean"] == pytest.approx(1, abs=1 / 255)
    assert grey["sat_mean"] == pytest.approx(0, abs=1 / 255)


def test_black_and_white_value_endpoints(frame_factory):
    assert color_stats(frame_factory((0, 0, 0)))["val_mean"] == pytest.approx(0, abs=1 / 255)
    assert color_stats(frame_factory((255, 255, 255)))["val_mean"] == pytest.approx(1, abs=1 / 255)


def test_black_white_halves_use_luminance_scale_zero_to_100(frame_factory):
    frame = frame_factory((0, 0, 0))
    frame[:, frame.shape[1] // 2 :] = 255
    assert color_stats(frame)["contrast"] == pytest.approx(50, abs=0.5)


def test_dark_with_small_bright_patch_is_low_key(frame_factory):
    frame = frame_factory((8, 8, 8))
    frame[0:8, 0:8] = 255
    assert lighting_key(frame) == "low-key"


def test_uniform_bright_frame_is_high_key(frame_factory):
    assert lighting_key(frame_factory((245, 245, 245))) == "high-key"


@pytest.mark.parametrize("value", [0, 64, 128, 192, 255])
def test_lighting_key_uses_schema_vocabulary(frame_factory, value):
    schema_path = Path(__file__).resolve().parents[1] / "configs" / "schema.yaml"
    schema = yaml.safe_load(schema_path.read_text())
    assert lighting_key(frame_factory((value, value, value))) in schema["attributes"]["key"]["values"]
