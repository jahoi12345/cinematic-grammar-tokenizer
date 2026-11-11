"""Synthetic source-grouped split contract checks; no dataset files are used."""
import hashlib
import json

import pytest

from cgt.p4_camera_move import split


def test_seed_zero_split_is_sorted_unique_grouped_and_order_independent():
    sources = [f"S{i:02}" for i in range(10)]
    expected = {s: "val" if s in {"S04", "S06"} else "train" for s in sources}
    assert split.source_grouped_split(sources) == expected
    assert split.source_grouped_split(sources[::-1] + sources[:3]) == expected
    assert split.source_grouped_split(sources, seed=1) != expected


def test_129_synthetic_sources_have_19_validation_sources():
    side = split.source_grouped_split([f"SOURCE{i:03}" for i in range(129)])
    assert len(side) == 129 and list(side.values()).count("val") == 19
    assert list(side.values()).count("train") == 110
    assert set(side.values()) == {"train", "val"}


def test_manifest_is_deterministic_complete_disjoint_and_hashed(tmp_path):
    rows = [{"clip": f"S{s:02}.webm_{j}.mp4", "source": f"S{s:02}"}
            for s in range(10) for j in range(3)]
    path = tmp_path / "split.json"
    result = split.write_split_manifest(rows, path)
    original = path.read_bytes()
    assert split.write_split_manifest(rows[::-1], path) == result
    assert path.read_bytes() == original
    assert result["n_train_clips"] == 24 and result["n_val_clips"] == 6
    assert set(result["train_sources"]).isdisjoint(result["val_sources"])
    assert set(result["clips"]["train"]).isdisjoint(result["clips"]["val"])
    assert set(result["clips"]["train"] + result["clips"]["val"]) == {r["clip"] for r in rows}
    by_clip = {r["clip"]: r["source"] for r in rows}
    assert all(by_clip[c] in result["val_sources"] for c in result["clips"]["val"])
    body = {k: v for k, v in result.items() if k != "sha256"}
    assert result["sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert json.loads(path.read_text()) == result


def test_split_refuses_empty_sides_and_keeps_at_least_one_validation_source():
    for sources, fraction in (([], .15), (["only"], .15), (["one", "two"], 1.)):
        with pytest.raises(ValueError):
            split.source_grouped_split(sources, val_frac=fraction)
    assert list(split.source_grouped_split(["one", "two"], val_frac=0.).values()).count("val") == 1


def test_manifest_refuses_duplicate_clip_identity(tmp_path):
    rows = [{"clip": "duplicate.mp4", "source": "S00"},
            {"clip": "duplicate.mp4", "source": "S04"},
            *[{"clip": f"unique{i}.mp4", "source": f"S{i:02}"} for i in range(10)]]
    with pytest.raises(AssertionError):
        split.write_split_manifest(rows, tmp_path / "split.json")
    assert not (tmp_path / "split.json").exists()
