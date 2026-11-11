"""Pipeline contract on tiny synthetic tables; expensive model calls are controlled."""

import hashlib

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from cgt.p1_shots_color import pipeline as phase1
from cgt.p2_pretrained import audio, blocking, depth, pipeline

COLUMNS = ["shot_id", "people", "faces", "subject_col", "subject_row", "headroom", "facing",
           "light_dir", "light_asym", "face_bg_ratio", "dof_log_ratio", "dof", "sound",
           "sound_db", "sound_top", "sound_top_p", "sound_p_speech", "sound_p_music", "subtitle_text"]
FLOATS = ["subject_col", "subject_row", "headroom", "light_asym", "face_bg_ratio",
          "dof_log_ratio", "sound_db", "sound_top_p", "sound_p_speech", "sound_p_music"]


@pytest.fixture
def synthetic_pipeline(tmp_path, monkeypatch):
    for module in (phase1, pipeline):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    film = {"id": "synthetic", "path": "tone.wav", "subtitles": "cues.srt"}
    (tmp_path / "configs/films.yaml").write_text(yaml.safe_dump({"films": [film]}))
    (tmp_path / "cues.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst\n")
    (tmp_path / "tone.wav").write_bytes(b"synthetic audio transport")
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    rows = []
    for shot_id, value, start, end in [(2, 100, 1, 2), (1, 0, 0, 1), (3, 100, 2, 301)]:
        image = tmp_path / f"frame_{shot_id}.png"
        assert cv2.imwrite(str(image), np.full((80, 80, 3), value, np.uint8))
        rows.append({"shot_id": shot_id, "start_s": float(start), "end_s": float(end), "kf_mid": image.name})
    pd.DataFrame(rows).to_parquet(processed / "synthetic_shots.parquet", index=False)
    (tmp_path / "data/interim/synthetic").mkdir(parents=True)
    monkeypatch.setattr(blocking, "detect_people", lambda frame: [])
    monkeypatch.setattr(blocking, "detect_faces", lambda frame: [] if not frame.any() else [{
        "box": (20, 20, 60, 60), "eye_l": 30, "eye_r": 50, "nose": 40
    }])
    monkeypatch.setattr(depth, "depth_map", lambda frame: np.zeros(frame.shape[:2], np.float32))
    monkeypatch.setattr(audio, "decode_audio", lambda path, cache: np.zeros(2 * audio.SR, np.float32))
    return tmp_path, film


def test_pipeline_columns_dtypes_values_and_none_semantics(synthetic_pipeline):
    root, _ = synthetic_pipeline
    result = pipeline.run_film("synthetic", max_seconds=300)
    stored = pd.read_parquet(root / "data/processed/synthetic_p2.parquet")
    assert list(stored.columns) == COLUMNS
    assert stored.shot_id.tolist() == [1, 2]
    for column in ("shot_id", "people", "faces"):
        assert str(stored[column].dtype) == "int64"
    for column in ("facing", "light_dir", "dof", "sound", "sound_top", "subtitle_text"):
        assert all(pd.isna(value) or isinstance(value, str) for value in stored[column])
    empty, closeup = stored.iloc[0], stored.iloc[1]
    assert empty.people == empty.faces == 0
    assert pd.isna(empty.facing) and pd.isna(empty.light_dir)
    assert empty[["subject_col", "subject_row", "headroom"]].isna().all()
    assert closeup.people == 0 and closeup.faces == 1 and closeup.facing == "camera"
    assert closeup.light_dir == "front"
    assert set(stored.dof) <= {"shallow", "medium", "deep"}
    assert set(stored.sound) == {"silence"}
    assert stored.sound_top.isna().all()
    assert stored.subtitle_text.tolist() == ["First", ""]
    pd.testing.assert_frame_equal(result, stored)


def test_pipeline_numeric_columns_keep_float64_when_all_values_missing(synthetic_pipeline):
    root, _ = synthetic_pipeline
    pipeline.run_film("synthetic", max_seconds=300)
    stored = pd.read_parquet(root / "data/processed/synthetic_p2.parquet")
    for column in FLOATS:
        assert str(stored[column].dtype) == "float64", column


def test_pipeline_cache_hit_does_not_rewrite_and_force_is_byte_stable(synthetic_pipeline, capsys):
    root, _ = synthetic_pipeline
    first = pipeline.run_film("synthetic", max_seconds=300)
    parquet = root / "data/processed/synthetic_p2.parquet"
    meta = root / "data/processed/synthetic_p2.meta.json"
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in (parquet, meta)]
    capsys.readouterr()
    second = pipeline.run_film("synthetic", max_seconds=300)
    output = capsys.readouterr().out
    assert output.count("cache hit") == 4 and "compute" not in output
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in (parquet, meta)]
    pd.testing.assert_frame_equal(first, second)
    pipeline.run_film("synthetic", max_seconds=300, force=True)
    assert hashlib.sha256(parquet.read_bytes()).digest() == hashlib.sha256(before[0][0]).digest()


def test_pipeline_boundary_change_invalidates_all_stages(synthetic_pipeline, capsys):
    root, _ = synthetic_pipeline
    pipeline.run_film("synthetic", max_seconds=300)
    path = root / "data/processed/synthetic_shots.parquet"
    shots = pd.read_parquet(path)
    shots.loc[shots.shot_id == 2, "end_s"] = 2.5
    shots.to_parquet(path, index=False)
    capsys.readouterr()
    pipeline.run_film("synthetic", max_seconds=300)
    output = capsys.readouterr().out
    assert output.count("compute") == 4 and "cache hit" not in output


def test_pipeline_absent_subtitles_omits_column_and_limit_follows_slice(synthetic_pipeline):
    root, film = synthetic_pipeline
    film["subtitles"] = None
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": [film]}))
    result = pipeline.run_film("synthetic", max_seconds=300, limit=1)
    assert result.shot_id.tolist() == [2]
    assert "subtitle_text" not in result.columns
    assert list(result.columns) == COLUMNS[:-1]
