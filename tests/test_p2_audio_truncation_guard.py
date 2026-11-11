"""the decode_audio truncation guard (F2 fix) had no test.
The guard cuts the aligned array back to the container duration when gap padding runs more than 1 s past it (garbage timestamps in a
damaged tail). A real damaged file that triggers it is hard to build, so the container duration is monkeypatched."""
import json
import shutil
import subprocess

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from cgt.p2_pretrained import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")
SR = audio.SR


@pytest.fixture(scope="module")
def tone3(tmp_path_factory):
    p = tmp_path_factory.mktemp("tg") / "tone3.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3:r=44100", "-c:a", "pcm_s16le", str(p)], check=True)
    return p


def test_real_duration_no_truncation(tone3, tmp_path):
    c = tmp_path / audio.CACHE_NAME
    a = audio.decode_audio(str(tone3), c)
    assert abs(len(a) / SR - 3.0) < 0.05
    assert not (tmp_path / (audio.CACHE_NAME + ".decode.json")).exists()  # clean file: no damage record


def test_padding_past_container_is_truncated(tone3, tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "_container_seconds", lambda _p: 1.5)  # decoded 3 s > 1.5 + 1 s tolerance
    c = tmp_path / audio.CACHE_NAME
    a = audio.decode_audio(str(tone3), c)
    assert len(a) == int(1.5 * SR)
    info = json.loads((tmp_path / (audio.CACHE_NAME + ".decode.json")).read_text())
    assert abs(info["truncated_from_seconds"] - 3.0) < 0.05 and abs(info["seconds_decoded"] - 1.5) < 1e-6


def test_within_tolerance_or_unknown_duration_not_truncated(tone3, tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "_container_seconds", lambda _p: 2.5)  # 3 s <= 2.5 + 1 s: kept whole
    assert abs(len(audio.decode_audio(str(tone3), tmp_path / audio.CACHE_NAME)) / SR - 3.0) < 0.05
    monkeypatch.setattr(audio, "_container_seconds", lambda _p: None)  # ffprobe failed: no truncation
    d = tmp_path / "b"
    d.mkdir()
    assert abs(len(audio.decode_audio(str(tone3), d / audio.CACHE_NAME)) / SR - 3.0) < 0.05
