"""independent tests for the direction-probe guards of `a sibling working copy` that the branch's own tests leave unkilled: the memory check before the CONTROL feature batch (P-M1) and the
exactness of a recorded decision exception (only `data/external/shotqa/grpo.json`, never another ShotQA or forbidden path). Skipped where the probe code is absent (main before the merge)."""
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/p4_move_probes.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists() or not (ROOT / "src/cgt/p4_camera_move/probes.py").exists(), reason="direction-probe code not on this branch")


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("p4_move_probes_script_vb", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _clip(path):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24000 / 1001, (64, 48))
    for i in range(9):
        fr = np.zeros((48, 64, 3), np.uint8)
        fr[:, :, 2] = i
        wr.write(fr)
    wr.release()


def test_memory_is_checked_before_the_control_batches_too(mod, tmp_path, monkeypatch):
    src = tmp_path / "s.avi"
    _clip(src)
    events = []
    acc_f, acc_d = tmp_path / "acc_f.npz", tmp_path / "acc_d.npz"
    np.savez(acc_f, x=np.array([0.0]))
    np.savez(acc_d, y=np.array([0.0]))
    monkeypatch.setattr(mod.FM, "cut_clip", lambda _f, _s, _e, dest: shutil.copyfile(src, dest) and 9)
    monkeypatch.setattr(mod.FM, "cache_names", lambda *_a: (acc_f, acc_d))
    monkeypatch.setattr(mod.FM, "memory_guard", lambda: events.append("guard"))
    monkeypatch.setattr(mod.F, "cache_key", lambda: "ff")
    monkeypatch.setattr(mod.D, "cache_key", lambda: "dd")
    monkeypatch.setattr(mod.F, "clip_series", lambda _p: events.append("flow") or {"x": np.array([0.0])})
    monkeypatch.setattr(mod.D, "clip_series", lambda _p: events.append("depth") or {"y": np.array([0.0])})
    monkeypatch.setattr(mod.F, "clip_features", lambda _s: {"x": 0.0})
    monkeypatch.setattr(mod.D, "clip_features", lambda _s: {})
    _sid, _out, err = mod.probe_shot((0, str(src), 0.0, 1.0, "film", str(tmp_path / "acc"), str(tmp_path / "cache"), True))
    assert err is None, err
    flows = [i for i, e in enumerate(events) if e == "flow"]
    assert len(flows) == 5  # forward re-run, identity, T, H, V
    assert all(i > 0 and events[i - 1] == "guard" for i in flows)


def test_the_train_label_exception_admits_exactly_one_file(tmp_path):
    from cgt.p4_camera_move import probes as PR
    files = {}
    for rel in ("data/external/shotqa/grpo.json", "data/external/shotqa/test.jsonl", "elsewhere/shotqa/grpo.json", "data/external/shotqa/grpo.json.bak/videos.csv"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        files[rel] = p
    with PR.tripwire(), PR.train_labels_allowed():
        assert files["data/external/shotqa/grpo.json"].read_text() == "{}"
        for rel in ("data/external/shotqa/test.jsonl", "elsewhere/shotqa/grpo.json", "data/external/shotqa/grpo.json.bak/videos.csv"):
            with pytest.raises(PermissionError):
                files[rel].read_text()
