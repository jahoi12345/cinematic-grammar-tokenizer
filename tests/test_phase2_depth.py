"""Depth expectations come from docs/phase2_interface.md, not learned scene labels."""

import cv2
import numpy as np
import pytest

from cgt.p2_pretrained.depth import depth_map, dof_features, dof_label


def test_dof_vocabulary_at_clear_extremes_and_middle():
    assert dof_label(-10) == "deep"
    assert dof_label(10) == "shallow"
    assert dof_label(0.9) == "medium"


def test_dof_features_matches_written_laplacian_formula(frame_factory):
    frame = frame_factory((90, 90, 90), width=64)
    frame[:, :24] = np.where(np.indices((64, 24)).sum(axis=0)[..., None] % 2, 255, 0)
    depth = np.tile(np.linspace(1, 0, 64, dtype=np.float32), (64, 1))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) ** 2
    near = energy[depth >= np.quantile(depth, 0.7)].mean()
    far = energy[depth <= np.quantile(depth, 0.3)].mean()
    expected = np.log((near + 1e-3) / (far + 1e-3))
    result = dof_features(frame, depth)
    assert set(result) == {"dof_log_ratio", "dof"}
    assert result["dof_log_ratio"] == pytest.approx(float(expected), rel=1e-5)
    assert result["dof"] == "shallow"
    assert dof_features(frame, depth) == result


def test_blank_frame_dof_is_finite_deep(frame_factory):
    result = dof_features(frame_factory((0, 0, 0)), np.zeros((64, 96), dtype=np.float32))
    assert np.isfinite(result["dof_log_ratio"])
    assert result["dof_log_ratio"] == pytest.approx(0)
    assert result["dof"] == "deep"


@pytest.mark.ml
def test_model_depth_shape_dtype_range_and_repeatability(frame_factory, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    frame = frame_factory((80, 80, 80), height=48, width=64)
    frame[12:36, 20:44] = (10, 90, 220)
    first = depth_map(frame)
    second = depth_map(frame)
    assert first.shape == frame.shape[:2]
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert first.min() >= 0 and first.max() <= 1
    np.testing.assert_array_equal(first, second)
