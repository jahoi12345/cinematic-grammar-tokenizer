"""decode_audio keeps a partial decode of a damaged file (and records it) but still fails when nothing decodes."""
import json
import subprocess

import numpy as np
import pytest

from cgt.p2_pretrained import audio


class R:
    def __init__(self, rc, out=b"", err=b""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_partial_decode_is_kept_and_recorded(tmp_path, monkeypatch):
    data = np.ones(3 * audio.SR, np.float32).tobytes()
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: R(234, data, b"Invalid NAL unit size"))
    a = audio.decode_audio("x.mp4", tmp_path / "a.npy")
    assert len(a) == 3 * audio.SR and (tmp_path / "a.npy").exists()
    info = json.loads((tmp_path / "a.npy.decode.json").read_text())
    assert info["ffmpeg_returncode"] == 234 and info["seconds_decoded"] == 3.0 and "NAL" in info["stderr_tail"]


def test_nothing_decoded_is_still_a_failure_and_clean_decode_writes_no_note(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: R(1, b"", b"boom"))
    with pytest.raises(subprocess.CalledProcessError):
        audio.decode_audio("x.mp4", tmp_path / "a.npy")
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: R(0, np.zeros(audio.SR, np.float32).tobytes()))
    audio.decode_audio("x.mp4", tmp_path / "b.npy")
    assert not (tmp_path / "b.npy.decode.json").exists()
