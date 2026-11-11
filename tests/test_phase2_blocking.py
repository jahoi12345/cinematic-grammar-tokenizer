"""Controlled detections isolate the public blocking rules from detector accuracy."""

import numpy as np
import pytest

from cgt.p2_pretrained import blocking

KEYS = {"people", "faces", "subject_col", "subject_row", "headroom", "facing",
        "light_dir", "light_asym", "face_bg_ratio"}


def face(box=(20.0, 20.0, 60.0, 60.0), nose=40.0):
    return {"box": box, "eye_l": 30.0, "eye_r": 50.0, "nose": nose}


def test_facing_frontal_profile_and_degenerate_eye_span():
    assert blocking.facing(face()) == "camera"
    assert blocking.facing(face(nose=55)) == "profile"
    tiny = {**face(), "eye_r": 30.5}
    assert blocking.facing(tiny) == "profile"
    assert blocking.facing(face()) == blocking.facing(face())


def test_light_direction_front_side_back_and_sign():
    gray = np.full((80, 80), 100, dtype=np.uint8)
    frontal = blocking.light_direction(gray, face())
    assert frontal["light_dir"] == "front"
    assert frontal["light_asym"] == pytest.approx(0)
    assert frontal["face_bg_ratio"] == pytest.approx(1, abs=0.01)
    gray[20:60, 20:40] = 30
    gray[20:60, 40:60] = 170
    side = blocking.light_direction(gray, face())
    assert side["light_dir"] == "side"
    assert side["light_asym"] > 0
    gray[20:60, 20:40] = 1
    gray[20:60, 40:60] = 20
    back = blocking.light_direction(gray, face())
    assert back["light_dir"] == "back", "Back-light rule must win over asymmetry"
    assert back["face_bg_ratio"] < 0.6


def test_tiny_face_light_fields_are_none():
    result = blocking.light_direction(np.full((80, 80), 100, np.uint8), face(box=(0, 0, 2, 2)))
    assert result == {"light_dir": None, "light_asym": None, "face_bg_ratio": None}


def test_no_person_and_no_face_yields_none_fields(frame_factory, monkeypatch):
    monkeypatch.setattr(blocking, "detect_people", lambda frame: [])
    monkeypatch.setattr(blocking, "detect_faces", lambda frame: [])
    result = blocking.blocking_features(frame_factory((0, 0, 0), width=80, height=80))
    assert set(result) == KEYS
    assert result["people"] == result["faces"] == 0
    assert all(result[key] is None for key in KEYS - {"people", "faces"})


def test_face_only_closeup_can_have_facing_with_zero_people(frame_factory, monkeypatch):
    monkeypatch.setattr(blocking, "detect_people", lambda frame: [])
    monkeypatch.setattr(blocking, "detect_faces", lambda frame: [face()])
    result = blocking.blocking_features(frame_factory((100, 100, 100), width=80, height=80))
    assert result["people"] == 0 and result["faces"] == 1
    assert result["facing"] == "camera"
    assert result["light_dir"] == "front"
    assert result["subject_col"] == result["subject_row"] == 1
    assert result["headroom"] == pytest.approx(0.25)


def test_person_without_matched_face_is_away(frame_factory, monkeypatch):
    monkeypatch.setattr(blocking, "detect_people", lambda frame: [(0, 0, 25, 70)])
    monkeypatch.setattr(blocking, "detect_faces", lambda frame: [face(box=(40, 20, 60, 60))])
    result = blocking.blocking_features(frame_factory((100, 100, 100), width=80, height=80))
    assert result["people"] == result["faces"] == 1
    assert result["facing"] == "away"
    assert result["light_dir"] is result["light_asym"] is result["face_bg_ratio"] is None
    assert result["subject_col"] == 0


@pytest.mark.ml
def test_real_detectors_on_blank_frame_have_no_detections(frame_factory, monkeypatch):
    # Point only the model loader root at the existing local weights, never data.
    import os
    from pathlib import Path

    # Author fix 2026-10-02: parents[4] only reached the repo root from a nested worktree and gave "/"
    # in the main checkout. Use the env override if set; otherwise blocking.ROOT, skipping if weights are absent.
    if os.environ.get("CGT_TEST_MODELS_ROOT"):
        monkeypatch.setattr(blocking, "ROOT", Path(os.environ["CGT_TEST_MODELS_ROOT"]))
    weights = ("yolo11n.pt", "face_detection_yunet_2023mar.onnx")
    missing = [name for name in weights if not (blocking.ROOT / "models" / name).is_file()]
    if missing:
        pytest.skip(f"local model weights absent: {missing}; set CGT_TEST_MODELS_ROOT")
    frame = frame_factory((0, 0, 0), height=64, width=96)
    assert blocking.detect_people(frame) == []
    assert blocking.detect_faces(frame) == []
