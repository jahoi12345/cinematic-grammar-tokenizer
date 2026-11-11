"""Tests of the runner's cache provenance and per-batch memory check, with synthetic clips and stubbed extractors. The independent versions of these checks are in a sibling working copy."""
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("p4_move_probes_script", ROOT / "scripts/p4_move_probes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _clip(path):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24000 / 1001, (64, 48))
    for i in range(9):
        fr = np.zeros((48, 64, 3), np.uint8)
        fr[:, :32, 1] = 200
        fr[:, :, 2] = i
        wr.write(fr)
    wr.release()


def _stub(m, monkeypatch, source, events):
    def cut(_f, _s, _e, dest):
        shutil.copyfile(source, dest)
        return 9

    def flow(path):
        events.append("flow")
        cap = cv2.VideoCapture(str(path))
        _, fr = cap.read()
        cap.release()
        return {"left_green": np.array([float(fr[:, :32, 1].mean())])}
    monkeypatch.setattr(m.FM, "cut_clip", cut)
    monkeypatch.setattr(m.F, "cache_key", lambda: "ff")
    monkeypatch.setattr(m.D, "cache_key", lambda: "dd")
    monkeypatch.setattr(m.F, "clip_series", flow)
    monkeypatch.setattr(m.D, "clip_series", lambda _p: events.append("depth") or {})
    monkeypatch.setattr(m.F, "clip_features", lambda sf: {"left_green": float(sf["left_green"][0])})
    monkeypatch.setattr(m.D, "clip_features", lambda _: {})


def test_a_cached_series_without_or_with_another_identity_is_not_reused(mod, tmp_path, monkeypatch):
    src = tmp_path / "s.avi"
    _clip(src)
    _stub(mod, monkeypatch, src, [])
    args = (0, str(src), 0.0, 1.0, "film", str(tmp_path / "acc"), str(tmp_path / "cache"), False)
    mod.probe_shot(args)
    impl = mod.PR.impl_identity()
    f, _ = mod._names(tmp_path / "cache", 0, "H", "ff.dd.film", impl)
    assert mod._load(f, impl) is not None
    # the same file name but a stored identity of another implementation (or none at all) is never certified
    old = mod._load(f, impl)
    mod._save(f, {**old, "impl_identity": np.array("another implementation")})
    assert mod._load(f, impl) is None
    mod._save(f, old)  # a file written without any identity
    assert mod._load(f, impl) is None


def test_memory_is_checked_before_every_feature_batch(mod, tmp_path, monkeypatch):
    src = tmp_path / "s.avi"
    _clip(src)
    events = []
    _stub(mod, monkeypatch, src, events)
    monkeypatch.setattr(mod.FM, "memory_guard", lambda: events.append("guard"))
    mod.probe_shot((0, str(src), 0.0, 1.0, "film", str(tmp_path / "acc"), str(tmp_path / "cache"), False))
    flows = [i for i, e in enumerate(events) if e == "flow"]
    assert len(flows) == 3 and all(events[i - 1] == "guard" for i in flows)


def test_the_worker_waits_for_memory_and_fails_closed(mod, monkeypatch):
    calls = []

    def guard():
        calls.append(1)
        if len(calls) < 3:
            raise mod.FM.Refused(12, "low")
    monkeypatch.setattr(mod.FM, "memory_guard", guard)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    mod.wait_memory(max_wait=1000, step=1)
    assert len(calls) == 3
    monkeypatch.setattr(mod.FM, "memory_guard", lambda: (_ for _ in ()).throw(mod.FM.Refused(12, "low")))
    with pytest.raises(mod.FM.Refused):
        mod.wait_memory(max_wait=2, step=1)
