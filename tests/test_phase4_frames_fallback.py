"""Amendment14: actual decoded count, unchanged bin rule and short-video refusal."""
import cv2
import numpy as np
import pytest

from cgt.p4_camera_move import frames_fallback as ff
from cgt.p4_camera_move import video_vlm as video


def synthetic_video(path, n):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24., (64, 64))
    assert writer.isOpened()
    for i in range(n):
        frame = np.empty((64, 64, 3), dtype=np.uint8)
        frame[:] = (i * 3 % 255, i * 7 % 255, i * 11 % 255)
        writer.write(frame)
    writer.release()


@pytest.mark.parametrize("reported_count", [120, 4])
def test_container_count_error_uses_true_count_and_preserves_bin_centres(tmp_path, monkeypatch, reported_count):
    path = tmp_path / "synthetic.avi"
    synthetic_video(path, 60)
    expected = video.extract_frames(path, 8, 64, tmp_path / "truth")
    real_capture = cv2.VideoCapture

    class BadCountCapture:
        def __init__(self, *args):
            self.real = real_capture(*args)

        def get(self, key):
            return reported_count if key == cv2.CAP_PROP_FRAME_COUNT else self.real.get(key)

        def set(self, *args):
            return self.real.set(*args)

        def read(self):
            return self.real.read()

        def release(self):
            return self.real.release()

    monkeypatch.setattr(ff.cv2, "VideoCapture", BadCountCapture)
    actual = ff.extract_frames_robust(path, 8, 64, tmp_path / "bad_header")
    assert len(actual) == 8
    assert [p.name for p in actual] == [p.name for p in expected]
    assert [p.read_bytes() for p in actual] == [p.read_bytes() for p in expected]
    # A second call must retain exactly the same bytes.
    assert [p.read_bytes() for p in ff.extract_frames_robust(path, 8, 64, tmp_path / "bad_header")] == [p.read_bytes() for p in expected]


def test_fewer_than_eight_decodable_frames_are_refused(tmp_path):
    path = tmp_path / "short.avi"
    synthetic_video(path, 5)
    with pytest.raises(ValueError):
        ff.extract_frames_robust(path, 8, 64, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.jpg"))
