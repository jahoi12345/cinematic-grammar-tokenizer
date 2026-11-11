"""F2: decode_audio must place every sample on the film timeline (sample i = i / SR s) even when the stream has timestamp gaps.
Synthetic clip: a 440 Hz tone from 0 to 3 s with the frames between 1 and 2 s removed (their timestamps kept), so the stream has a 1 s gap.
docs/p2_audio_drift_fix_2026-10-13.md"""
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from cgt.p2_pretrained import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")
SR = audio.SR


@pytest.fixture(scope="module", params=["gap.mkv", "gap.mp4"])
def gap_clip(request, tmp_path_factory):
    p = tmp_path_factory.mktemp("gap") / request.param
    codec = ["-c:a", "pcm_s16le"] if p.suffix == ".mkv" else ["-c:a", "aac"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3:r=44100,aselect='not(between(t\\,1\\,2))'", *codec, str(p)], check=True)
    return p


def _rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))))) if len(x) else 0.0


def _old_decode(path):  # the v1 decode (no timestamp alignment), kept here only to show the drift the fix removes
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def test_old_decode_drifts(gap_clip):
    a = _old_decode(gap_clip)
    assert len(a) / SR < 2.2  # the 1 s gap is dropped: everything after 1 s comes about 1 s early
    assert _rms(a[int(1.2 * SR): int(1.8 * SR)]) > 0.03  # tone where the film is silent


def test_new_decode_aligned(gap_clip, tmp_path):
    a = audio.decode_audio(str(gap_clip), tmp_path / audio.CACHE_NAME)
    assert abs(len(a) / SR - 3.0) < 0.05
    assert _rms(a[int(1.1 * SR): int(1.9 * SR)]) < 1e-3  # the gap is silence at its own time
    assert _rms(a[int(2.1 * SR): int(2.9 * SR)]) > 0.03  # the tone after the gap sits at 2..3 s
    assert _rms(a[int(0.1 * SR): int(0.9 * SR)]) > 0.03
    # onset of the second tone within 25 ms of 2.0 s
    env = np.abs(np.asarray(a[int(1.5 * SR): int(2.5 * SR)]))
    onset = 1.5 + np.argmax(env > 0.02) / SR
    assert abs(onset - 2.0) < 0.025


def test_cache_versioned_and_not_v1(tmp_path):
    assert audio.CACHE_NAME == "audio16k_v2.npy"  # the drifting v1 cache (audio16k.npy) is never reused
    import inspect

    from cgt.p2_pretrained import pipeline as p2
    assert "audio16k.npy" not in inspect.getsource(p2)


def test_cache_reused(gap_clip, tmp_path):
    c = tmp_path / audio.CACHE_NAME
    a = audio.decode_audio(str(gap_clip), c)
    np.save(c, np.zeros(5, np.float32))  # a cached array is returned as is
    assert len(audio.decode_audio(str(gap_clip), c)) == 5 and len(a) > 5


def test_damaged_tail_keeps_partial(tmp_path):
    good = tmp_path / "t.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=4:r=44100", "-c:a", "aac", str(good)], check=True)
    b = bytearray(good.read_bytes())
    cut = len(b) // 2
    b[cut:] = bytes((x * 7 + 13) & 0xFF for x in range(len(b) - cut))  # garbage second half
    bad = tmp_path / "bad.mkv"
    bad.write_bytes(bytes(b))
    a = audio.decode_audio(str(bad), tmp_path / audio.CACHE_NAME)
    assert 0 < len(a) <= int(5.0 * SR)  # partial audio kept, never padded beyond the container


# an earlier review (docs/cleanup_2026-10-14.md): `first_pts=0` must matter. A clip whose video starts at t = 0 and whose audio stream starts at pts 1 s
# (a late audio start on the film timeline). ffmpeg only removes the file's start time (0 here), so without first_pts=0 the first decoded
# sample is the first audio sample and the whole soundtrack sits 1 s early; with it the 1 s before the audio is padded with silence.
@pytest.fixture(scope="module", params=["late.mkv", "late.mp4"])
def late_audio_clip(request, tmp_path_factory):
    p = tmp_path_factory.mktemp("late") / request.param
    codec = ["-c:a", "pcm_s16le"] if p.suffix == ".mkv" else ["-c:a", "aac"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=3", "-itsoffset", "1", "-f", "lavfi", "-i", "sine=f=440:d=2:r=44100",
                    "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", *codec, str(p)], check=True)
    return p


def _onset(a, thr=0.02):
    return int(np.argmax(np.abs(np.asarray(a)) > thr)) / SR


def _decode_with(path, af):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-af", af, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def test_late_audio_start_without_first_pts_misaligns(late_audio_clip):
    a = _decode_with(late_audio_clip, "aresample=async=1")  # the filter without first_pts=0
    assert _onset(a) < 0.1  # the tone that starts at 1 s on the film timeline is decoded at about 0 s


def test_late_audio_start_aligned(late_audio_clip, tmp_path):
    a = audio.decode_audio(str(late_audio_clip), tmp_path / audio.CACHE_NAME)
    assert abs(len(a) / SR - 3.0) < 0.05
    assert _rms(a[int(0.1 * SR): int(0.9 * SR)]) < 1e-3  # silence before the audio stream starts
    assert abs(_onset(a) - 1.0) < 0.025  # the tone sits at its own time, 1 s
