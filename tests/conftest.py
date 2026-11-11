"""Small, lossless synthetic inputs; never touch film or held-out data."""

import cv2
import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "ml: uses locally cached pretrained model weights")


@pytest.fixture
def frame_factory(tmp_path):
    """Round-trip BGR uint8 frames through PNGs under pytest's temp directory."""
    counter = 0

    def make(bgr, *, height=64, width=96):
        nonlocal counter
        frame = np.full((height, width, 3), bgr, dtype=np.uint8)
        path = tmp_path / f"frame_{counter}.png"
        counter += 1
        assert cv2.imwrite(str(path), frame), "Could not write synthetic PNG"
        decoded = cv2.imread(str(path))
        np.testing.assert_array_equal(decoded, frame)
        return decoded

    return make


@pytest.fixture
def clip_factory(tmp_path):
    """Write FFV1 AVI clips and verify decoded pixels and frame counts."""
    counter = 0

    def make(segments, *, fps=24):
        nonlocal counter
        path = tmp_path / f"clip_{counter}.avi"
        counter += 1
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (96, 64))
        assert writer.isOpened(), "OpenCV must support FFV1 for lossless cut fixtures"
        try:
            for bgr, frame_count in segments:
                frame = np.full((64, 96, 3), bgr, dtype=np.uint8)
                for _ in range(frame_count):
                    writer.write(frame)
        finally:
            writer.release()

        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened(), "Could not reopen synthetic clip"
        try:
            assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(fps)
            for bgr, frame_count in segments:
                expected = np.full((64, 96, 3), bgr, dtype=np.uint8)
                for _ in range(frame_count):
                    ok, decoded = capture.read()
                    assert ok, "Synthetic clip lost frames"
                    np.testing.assert_array_equal(decoded, expected)
            ok, _ = capture.read()
            assert not ok, "Synthetic clip contains unexpected extra frames"
        finally:
            capture.release()
        return path

    return make
