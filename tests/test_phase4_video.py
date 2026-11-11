"""Video adapter/cross-fit contract on synthetic clips and source IDs; CPU only."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cgt.p4_camera_move import crossfit as x
from cgt.p4_camera_move import video_vlm as video


def test_eight_bin_centres_and_budget():
    assert video.frame_indices(96, 8) == [6, 18, 30, 42, 54, 66, 78, 90]
    assert video.frame_indices(8, 8) == list(range(8))
    assert video.budget_px(64) == 65536


def test_extract_rgb_frames_budget_content_key_and_idempotence(clip_factory, tmp_path):
    clip = clip_factory([((i * 20, 0, 200), 12) for i in range(8)])
    paths = video.extract_frames(clip, 8, 64, tmp_path / "frames")
    assert len(paths) == 8 and len(set(paths)) == 8
    for path, i in zip(paths, range(8), strict=True):
        with Image.open(path) as image:
            w, h = image.size
            assert image.mode == "RGB" and w % 32 == h % 32 == 0
            assert w * h <= 65536
            assert w / h == pytest.approx(1.5, rel=0.05)
            rgb = np.asarray(image).mean(axis=(0, 1))
            np.testing.assert_allclose(rgb, [200, 0, i * 20], atol=3)
    before = [p.stat().st_mtime_ns for p in paths]
    assert video.extract_frames(clip, 8, 64, tmp_path / "frames") == paths
    assert [p.stat().st_mtime_ns for p in paths] == before
    # Replace only synthetic video bytes while retaining the same path.
    replacement = clip_factory([((0, 200, 0), 96)])
    clip.write_bytes(replacement.read_bytes())
    assert set(video.extract_frames(clip, 8, 64, tmp_path / "frames")).isdisjoint(paths)


def test_sample_has_eight_images_one_answer_and_unchanged_options():
    row = {"clip": "synthetic.mp4", "gold_letter": "C",
           "options": {"D": "Pan right", "B": "Static shot", "A": "Tilt up", "C": "Push in"}}
    frames = [Path(f"synthetic_{i}.jpg") for i in range(8)]
    sample = video.build_sample(row, frames)
    assert sample["images"] == [str(p) for p in frames]
    assert sample["id"] == row["clip"]
    assert sample["messages"][1] == {"role": "assistant", "content": "C"}
    user = sample["messages"][0]
    assert user["role"] == "user" and "<video>" not in user["content"]
    assert user["content"].startswith("These 8 frames are sampled in time order from one movie shot. Question:")
    assert "\nA. Tilt up\nB. Static shot\nC. Push in\nD. Pan right\n" in user["content"]


def test_image_grid_sum_and_processor_budget_floor():
    grid = [[1, 12, 20], [1, 8, 8]] * 4
    assert video.image_token_count(grid) == 4 * (60 + 16)
    processor = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=65536))
    video.configure_processor(processor)
    assert processor.image_processor.min_pixels == 4096


def test_source_folds_follow_registered_permutation_and_are_oof_once():
    sources = [f"S{i:03d}" for i in range(110)]
    expected = {sources[int(j)]: r % 3 for r, j in enumerate(np.random.default_rng(0).permutation(110))}
    assert x.fold_assignment(sources + sources[:20]) == expected
    assert x.fold_assignment(list(reversed(sources))) == expected
    assert Counter(expected.values()) == {0: 37, 1: 37, 2: 36}
    rows = [{"clip": f"{s}.{i}.mp4", "source": s} for s in sources for i in range(2)]
    held = []
    for k in range(3):
        train, val = x.stage_split(rows, f"fold{k}")
        assert not {r["source"] for r in train} & {r["source"] for r in val}
        assert {r["source"] for r in val} == {s for s, v in expected.items() if v == k}
        assert len(train) + len(val) == len(rows)
        held.extend(r["clip"] for r in val)
    assert Counter(held) == Counter(r["clip"] for r in rows)
    assert x.stage_split(rows, "final") == (rows, [])


def test_epoch_orders_learning_rates_and_partial_accumulation():
    for stage, seed in [("final", 0), ("fold0", 1), ("fold1", 2), ("fold2", 3)]:
        for epoch in (1, 2, 3):
            np.testing.assert_array_equal(x.epoch_order(37, stage, epoch), np.random.default_rng([seed, epoch]).permutation(37))
    assert x.updates_per_epoch(17) == 3 and x.total_updates(17) == 9
    total = x.total_updates(903)
    assert x.lr_at(1, total) == pytest.approx(1e-5)
    assert x.lr_at(10, total) == pytest.approx(1e-4)
    assert x.lr_at(total, total) == pytest.approx(1e-5)
    assert all(x.lr_at(i + 1, total) <= x.lr_at(i, total) for i in range(10, total))


def test_epoch_selection_uses_pooled_accuracy_and_earliest_tie():
    correct = {1: [True] * 4 + [False] * 6, 2: [True] * 8 + [False] * 2, 3: [True] * 8 + [False] * 2}
    epoch, values = x.choose_epoch(correct)
    assert epoch == 2 and values == {1: 0.4, 2: 0.8, 3: 0.8}
    assert x.choose_epoch({3: [True], 2: [True], 1: [True]})[0] == 1
