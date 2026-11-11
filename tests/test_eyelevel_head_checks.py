"""an earlier review tests for the track E eye-level head guards (scripts/eyelevel_head.py, cgt.eyelevel.head).

Synthetic data and temporary git repos only; no CineScale2, ShotQA or held-out data is read."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cgt.eyelevel import head as H

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eyelevel_head.py"


@pytest.fixture
def S():
    spec = importlib.util.spec_from_file_location("eyelevel_head_script_veh", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)


# ---------------------------------------------------------------- committed-before-use guard
def test_require_committed_untracked_modified_and_clean(S, tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "v@x")
    _git(tmp_path, "config", "user.name", "v")
    monkeypatch.setattr(S, "ROOT", tmp_path)
    (tmp_path / "plan.md").write_text("a\n")
    (tmp_path / "other.md").write_text("o\n")
    _git(tmp_path, "add", "other.md")
    _git(tmp_path, "commit", "-qm", "init")
    with pytest.raises(SystemExit, match="not committed"):
        S.require_committed("plan.md")
    _git(tmp_path, "add", "plan.md")
    with pytest.raises(SystemExit):   # staged but not committed: differs from HEAD
        S.require_committed("plan.md")
    _git(tmp_path, "commit", "-qm", "plan")
    assert len(S.require_committed("plan.md").split()[0]) == 40
    (tmp_path / "plan.md").write_text("b\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        S.require_committed("plan.md")


def test_cv_refuses_before_plan_is_committed(S, monkeypatch):
    def refuse(rel):
        raise SystemExit(f"{rel} is not committed")
    monkeypatch.setattr(S, "require_committed", refuse)
    monkeypatch.setattr(S, "load_train_val", lambda: pytest.fail("cv read data before the plan check"))
    with pytest.raises(SystemExit, match="eyelevel_head.md"):
        S.cmd_cv(None)


def test_final_refuses_unless_cv_json_committed(S, tmp_path, monkeypatch):
    seen = []

    def req(rel):
        seen.append(rel)
        if rel.endswith("cv.json"):
            raise SystemExit("cv.json is not committed")
        return "sha"
    monkeypatch.setattr(S, "require_committed", req)
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(S, "OUT", tmp_path / "out")
    monkeypatch.setattr(S, "load_train_val", lambda: pytest.fail("final read val before the cv.json check"))
    with pytest.raises(SystemExit, match="cv.json"):
        S.cmd_final(None)
    assert seen == [S.PLAN, "results/eyelevel_head/cv.json"]
    assert not (tmp_path / "cache" / "final.stamp").exists()


@pytest.mark.parametrize("which", ["stamp", "summary"])
def test_final_refuses_second_run(S, tmp_path, monkeypatch, which):
    monkeypatch.setattr(S, "require_committed", lambda rel: "sha")
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(S, "OUT", tmp_path / "out")
    monkeypatch.setattr(S, "load_train_val", lambda: pytest.fail("second final read val"))
    (tmp_path / "cache").mkdir()
    (tmp_path / "out").mkdir()
    if which == "stamp":
        (tmp_path / "cache" / "final.stamp").write_text("t\n")
    else:
        (tmp_path / "out" / "summary.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="already"):
        S.cmd_final(None)


def test_final_writes_stamp_before_reading_val(S, tmp_path, monkeypatch):
    """The once-only stamp must exist before val is loaded, so a crash mid-evaluation still blocks a rerun."""
    import hashlib
    import pickle
    blob = pickle.dumps({"det_key": H.grid()[0].key(), "threshold": {"threshold": 0.5}})
    (tmp_path / "out").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "heads.pkl").write_bytes(blob)
    (tmp_path / "out" / "cv.json").write_text(json.dumps({"heads_pkl_sha256": hashlib.sha256(blob).hexdigest()}))
    monkeypatch.setattr(S, "require_committed", lambda rel: "sha")
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(S, "OUT", tmp_path / "out")

    def load():
        assert (tmp_path / "cache" / "final.stamp").exists()
        raise RuntimeError("stop here")
    monkeypatch.setattr(S, "load_train_val", load)
    with pytest.raises(RuntimeError, match="stop here"):
        S.cmd_final(None)
    with pytest.raises(SystemExit, match="already"):
        S.cmd_final(None)


def test_final_refuses_tampered_heads(S, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "heads.pkl").write_bytes(b"x")
    (tmp_path / "out" / "cv.json").write_text(json.dumps({"heads_pkl_sha256": "0" * 64}))
    monkeypatch.setattr(S, "require_committed", lambda rel: "sha")
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(S, "OUT", tmp_path / "out")
    with pytest.raises(SystemExit, match="differs"):
        S.cmd_final(None)


# ---------------------------------------------------------------- train-only selection
def test_selection_refuses_a_single_val_id_among_many():
    ids = [f"cinescale2/angle/high/high_{k:05d}.jpg" for k in range(200)]
    val = ["cinescale2/angle/neutral/neutral_00007.jpg"]
    rng = np.random.default_rng(0)
    s, e = rng.random(201), rng.random(201) < 0.3
    with pytest.raises(ValueError, match="train only"):
        H.select_threshold(ids + val, s, e, val_ids=val)
    res = {"m": {"family": "multinomial", "C": 0.1, "cw": None, "balanced_acc": 0.5, "eye_ap": 0.5}}
    with pytest.raises(ValueError, match="train only"):
        H.select_configs(res, ids[:3] + val, val)
    H.select_threshold(ids, s[:200], e[:200], val_ids=val)   # train only: fine


def test_cv_passes_train_ids_and_val_ids_to_selection(S, monkeypatch):
    """Code read made executable: cmd_cv hands selection the TRAIN ids and the real val ids (never val features)."""
    train = [f"cinescale2/angle/{c}/{c}_{k:05d}.jpg" for c in H.CLASSES for k in range(12)]
    val = [f"cinescale2/angle/{c}/{c}_9{k:04d}.jpg" for c in H.CLASSES for k in range(2)]
    rng = np.random.default_rng(1)
    ytr = np.array([H.label_of(i) for i in train])
    xtr = rng.normal(size=(len(train), 4)) + np.eye(5, 4)[ytr]
    xva = np.full((len(val), 4), np.nan)  # any use of val features would poison the fit
    monkeypatch.setattr(S, "require_committed", lambda rel: "sha")
    monkeypatch.setattr(S, "load_train_val", lambda: (train, xtr, ytr, val, xva, {}))
    calls = []
    real_cfg, real_thr = H.select_configs, H.select_threshold
    monkeypatch.setattr(S.H, "select_configs", lambda r, ids, v: calls.append(("cfg", list(ids), list(v))) or real_cfg(r, ids, v))
    monkeypatch.setattr(S.H, "select_threshold", lambda ids, sc, e, v: calls.append(("thr", list(ids), list(v)))
                        or real_thr(ids, sc, e, v))
    monkeypatch.setattr(S.H, "grid", lambda: [H.Config("multinomial", 1.0, None), H.Config("binary", 1.0, None)])

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(S, "CACHE", Path(d) / "c")
        monkeypatch.setattr(S, "OUT", Path(d) / "o")
        (Path(d) / "c").mkdir()
        (Path(d) / "c" / "prep.json").write_text(json.dumps(
            {"train": train, "val": val, "groups": {i: i for i in train + val}}))
        S.cmd_cv(None)
    assert [c[0] for c in calls] == ["cfg", "thr"]
    for _, ids, v in calls:
        assert ids == train and v == val


# ---------------------------------------------------------------- adoption thresholds (exact boundaries)
@pytest.mark.parametrize("p,r,g,ok", [
    (0.75, 0.60, 0.05, True), (0.7499, 0.60, 0.05, False), (0.75, 0.5999, 0.05, False), (0.75, 0.60, 0.0501, False),
    (0.8687, 0.9451, 0.00835, True),   # the reported final values
])
def test_adoption_boundaries(p, r, g, ok):
    assert H.adoption(p, r, g, True)[0] is ok
    assert H.adoption(p, r, g, False)[0] is False
