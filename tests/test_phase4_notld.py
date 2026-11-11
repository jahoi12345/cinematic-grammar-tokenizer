"""Amendments 10/11 on synthetic inputs; no real film or evaluator invocation."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def movement_script():
    root = Path(os.environ.get("CGT_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
    path = root / "scripts/p4_notld_move.py"
    if not path.exists():
        pytest.skip("NOTLD movement script absent; run against the pinned scratch source tree")
    spec = importlib.util.spec_from_file_location("notld_sampling_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("start,end", [(0.11, 0.63), (8.731, 9.464), (190.4, 196.72)])
def test_resampling_uses_nearest_original_frames_at_training_fps(movement_script, monkeypatch, tmp_path, start, end):
    module = movement_script
    cv, written, seeks = module.cv2, [], []
    original_fps = 30000 / 1001

    class Capture:
        pos = 0

        def get(self, prop):
            return {cv.CAP_PROP_FPS: original_fps, cv.CAP_PROP_FRAME_WIDTH: 16,
                    cv.CAP_PROP_FRAME_HEIGHT: 16}[prop]

        def set(self, prop, value):
            assert prop == cv.CAP_PROP_POS_FRAMES
            self.pos = int(value)
            seeks.append(self.pos)

        def read(self):
            image = np.full((16, 16, 3), self.pos, dtype=np.int32)
            self.pos += 1
            return True, image

        def release(self):
            pass

    class Writer:
        def __init__(self, path, fourcc, fps, shape):
            assert fps == 24000 / 1001 and shape == (16, 16)

        def write(self, image):
            written.append(int(image[0, 0, 0]))

        def release(self):
            pass

    monkeypatch.setattr(cv, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(cv, "VideoWriter", Writer)
    count = module.cut_clip(tmp_path / "synthetic.mp4", start, end, tmp_path / "synthetic.avi")
    n = int(np.floor((end - start) * (24000 / 1001)))
    assert count == len(written) == n
    assert seeks == [written[0]]
    for j, frame in enumerate(written):
        source_time_in_frames = (start + j * 1001 / 24000) * original_fps
        assert abs(frame - source_time_in_frames) <= 0.5 + 1e-8


def test_sets_use_inclusive_threshold_and_schema_order_with_nonempty_fallback(movement_script):
    p = np.asarray([[0.5, 0.9, 0.499], [0.1, 0.3, 0.2], [0.2, 0.2, 0.2]])
    sets, fallback = movement_script.predict(p, ["pan-right", "static", "roll"])
    assert sets == [["pan-right", "static"], ["static"], ["pan-right"]]
    assert fallback == [False, True, True]
