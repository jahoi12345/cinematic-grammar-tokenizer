"""Synthetic waveforms; model labels are checked for validity, not musical truth."""

import wave

import numpy as np
import pytest

from cgt.p2_pretrained.audio import SR, classify_clip, decode_audio, rms_db


def test_rms_db_silence_full_scale_and_half_scale():
    assert rms_db(np.zeros(100, np.float32)) == pytest.approx(-180)
    assert rms_db(np.ones(100, np.float32)) == pytest.approx(0, abs=1e-6)
    assert rms_db(np.full(100, 0.5, np.float32)) == pytest.approx(-6.0206, abs=1e-4)


def test_silence_empty_and_under_quarter_second_semantics():
    for signal in (np.zeros(SR, np.float32), np.empty(0, np.float32), np.ones(SR // 4 - 1, np.float32)):
        result = classify_clip(signal)
        assert set(result) == {"sound", "sound_db", "sound_top", "sound_top_p", "sound_p_speech", "sound_p_music"}
        assert result["sound"] == "silence"
        assert result["sound_top"] is result["sound_top_p"] is None
        assert result["sound_p_speech"] is result["sound_p_music"] is None
        assert np.isfinite(result["sound_db"])
    assert classify_clip(np.empty(0, np.float32))["sound_db"] == -120.0


def test_decode_synthetic_wav_and_readonly_cache(tmp_path):
    assert SR == 16000
    path = tmp_path / "tone.wav"
    signal = (8000 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SR)
        output.writeframes(signal.tobytes())
    cache = tmp_path / "tone.npy"
    first = decode_audio(str(path), cache)
    assert first.dtype == np.float32
    assert first.shape == (SR,)
    np.testing.assert_allclose(first, signal.astype(np.float32) / 32768, atol=1e-5)
    path.unlink()  # Scratch file only: second call must use cache rather than ffmpeg.
    second = decode_audio(str(path), cache)
    assert isinstance(second, np.memmap)
    assert not second.flags.writeable
    np.testing.assert_array_equal(first, second)


@pytest.mark.ml
def test_model_audio_vocabulary_probabilities_and_repeatability(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    signal = (0.2 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    first = classify_clip(signal)
    second = classify_clip(signal)
    assert first["sound"] in {"dialogue", "score", "ambient"}
    assert isinstance(first["sound_top"], str) and first["sound_top"]
    assert 0 <= first["sound_top_p"] <= 1
    assert 0 <= first["sound_p_speech"] <= first["sound_top_p"]
    assert 0 <= first["sound_p_music"] <= first["sound_top_p"]
    assert np.isfinite(first["sound_db"])
    assert first == second
