"""F1 (move prerequisite: the ShotQA train label file grpo.json) and F4 (summary without year), docs/move_prereq_fix_2026-10-13.md.
Synthetic tmp roots and stub stages only; the real grpo.json is never read (a fake empty file stands in for it, and only its existence is checked)."""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import EXIT, Ctx, Stage, Status

REPO = Path(__file__).resolve().parents[1]
GRPO = "data/external/shotqa/grpo.json"

# the movement identity lists exactly as on main 4edb029
MAIN_MOVE_CODE_FILES = ["scripts/run_move_film.py", "scripts/p4_final_eval.py", "src/cgt/p4_camera_move/film_move.py", "src/cgt/p4_camera_move/flow.py",
                        "src/cgt/p4_camera_move/depth_parallax.py", "src/cgt/p4_camera_move/final_run.py", "src/cgt/p4_camera_move/final_eval.py",
                        "src/cgt/p4_camera_move/labels.py"]
MAIN_MOVE_CONFIG_FILES = ["results/phase4/frozen_config_flow_depth.json", "results/phase4/split_manifest.json",
                          "data/cache/p4/features_shotqa.parquet", "data/cache/p4/features_depth_shotqa.parquet"]


def _build_root(root: Path, video_bytes: bytes | None = b"video-pd", grpo: bool = False) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "pd", "title": "PD", "public_domain": True, "path": "pd.mp4"}]}))
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    for rel in stages.MOVE_CODE_FILES + stages.MOVE_CONFIG_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"v1 {rel}".encode())
    if video_bytes is not None:
        (root / "pd.mp4").write_bytes(video_bytes)
    pd.DataFrame({"shot_id": [0, 1, 2], "start_s": [0.0, 1.0, 2.0], "end_s": [1.0, 2.0, 3.0], "len_s": 1.0}).to_parquet(root / "data/processed/pd_shots.parquet")
    if grpo:
        (root / GRPO).parent.mkdir(parents=True, exist_ok=True)
        (root / GRPO).write_bytes(b"")


@pytest.fixture
def env(tmp_path, monkeypatch):
    _build_root(tmp_path)
    monkeypatch.setattr(stages, "_free_percent", lambda: 80)
    calls = []
    monkeypatch.setattr(stages, "_run_move_script", lambda ctx, cmd: (calls.append(cmd), 1)[1])
    return tmp_path, calls


def ctx_for(root):
    return Ctx("pd", {"id": "pd", "title": "PD", "public_domain": True, "path": "pd.mp4"}, root=root, log=lambda m: None)


def recording_registry(called: list):
    """Real move stage; p1, p2, p3c as recorders that would run (cache misses; p3c is the GPU stage); tokens with only its movement prerequisite."""
    reg = stages.registry()
    for name in ("p1", "p2", "p3c"):
        reg[name] = Stage(name, name == "p3c", lambda ctx: Status(False, "k", "miss"), lambda ctx, n=name: (called.append(n), {})[1], lambda ctx: [])
    inner = reg["move"].run
    reg["move"].run = lambda ctx: (called.append("move"), inner(ctx))[1]
    reg["tokens"].run = lambda ctx: (called.append("tokens"), {})[1]
    reg["tokens"].status = lambda ctx: Status(False, "t", "miss")
    reg["tokens"].prerequisites = lambda ctx: []
    return reg


# ---------------------------------------------------------------- F1
def test_missing_grpo_is_a_move_prerequisite_failure(env):
    root, calls = env
    probs = stages.move_prereq(ctx_for(root))
    assert [c for c, _ in probs] == ["unavailable_or_prerequisite"] and GRPO in probs[0][1]
    logs = []
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=logs.append) == EXIT["unavailable_or_prerequisite"]
    assert calls == []
    assert any("PREREQUISITE FAILED in move" in m and GRPO in m for m in logs)


def test_full_run_stops_with_exit_5_before_any_stage_runs(env):
    root, calls = env
    called = []
    lock = root / "gpu.lock"
    assert orch.run("pd", None, recording_registry(called), root=root, log=lambda m: None, lock_path=lock) == EXIT["unavailable_or_prerequisite"]
    assert called == [] and calls == [] and not lock.exists()


def test_dry_run_reports_it_and_writes_nothing(env):
    root, calls = env
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    logs = []
    assert orch.run("pd", ["move"], stages.registry(), dry_run=True, root=root, log=logs.append) == EXIT["unavailable_or_prerequisite"]
    assert any("prerequisite problem [unavailable_or_prerequisite]" in m and GRPO in m for m in logs)
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before and calls == []


def test_present_fake_grpo_clears_it_and_is_only_checked_for_existence(env):
    root, _calls = env
    (root / GRPO).parent.mkdir(parents=True)
    (root / GRPO).write_bytes(b"")  # empty fake: a check that parsed it would fail
    os.chmod(root / GRPO, 0)          # unreadable: a check that opened it would fail
    try:
        assert stages.move_prereq(ctx_for(root)) == []
    finally:
        os.chmod(root / GRPO, 0o600)


def test_a_directory_at_the_grpo_path_does_not_count(env):
    root, _calls = env
    (root / GRPO).mkdir(parents=True)
    assert any(GRPO in m for _, m in stages.move_prereq(ctx_for(root)))


def test_other_move_prerequisites_still_reported(env):
    root, _calls = env
    (root / "pd.mp4").unlink()
    (root / MAIN_MOVE_CONFIG_FILES[2]).unlink()
    msgs = " | ".join(m for _, m in stages.move_prereq(ctx_for(root)))
    assert "video file missing" in msgs and "train feature tables" in msgs and GRPO in msgs


def test_movement_identity_is_unchanged(env):
    root, _calls = env
    assert stages.MOVE_CODE_FILES == MAIN_MOVE_CODE_FILES and stages.MOVE_CONFIG_FILES == MAIN_MOVE_CONFIG_FILES
    assert GRPO not in stages.MOVE_CODE_FILES + stages.MOVE_CONFIG_FILES
    ctx = ctx_for(root)
    without = stages._move_code_identity(ctx)
    (root / GRPO).parent.mkdir(parents=True)
    (root / GRPO).write_bytes(b"")
    assert stages._move_code_identity(ctx) == without
    (root / GRPO).write_bytes(b"[]")
    assert stages._move_code_identity(ctx) == without


# ---------------------------------------------------------------- F1 through the v1 entry point (scripts/tokenize_video.py) dry run
@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")
def test_tokenize_video_dry_run_reports_missing_grpo(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("tokenize_video_f1", REPO / "scripts/tokenize_video.py")
    tv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tv)
    root = tmp_path / "repo"
    _build_root(root, video_bytes=None)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x48:r=10:d=1", "-pix_fmt", "yuv420p", str(root / "pd.mp4")], check=True)
    monkeypatch.setattr(stages, "_free_percent", lambda: 80)
    called, logs = [], []
    probes = {s: (lambda fid: None) for s in ("p1", "p2", "p3c", "move")}
    code = tv.main([str(root / "pd.mp4"), "--dry-run"], root=root, stage_registry=recording_registry(called), probes=probes, log=logs.append, lock_path=root / "gpu.lock")
    text = "\n".join(logs)
    assert code == EXIT["unavailable_or_prerequisite"] and called == []
    assert "prerequisite problem" in text and GRPO in text
    (root / GRPO).parent.mkdir(parents=True)
    (root / GRPO).write_bytes(b"")
    logs.clear()
    assert tv.main([str(root / "pd.mp4"), "--dry-run"], root=root, stage_registry=recording_registry(called), probes=probes, log=logs.append,
                   lock_path=root / "gpu.lock") == EXIT["ok"]
    assert GRPO not in "\n".join(logs) and called == []


# ---------------------------------------------------------------- F4
def _summary(tmp_path, film):
    ctx = Ctx(film["id"], film, root=tmp_path, log=lambda m: None)
    return orch.write_summary(ctx, [], [], 1.0, 0).read_text()


def test_summary_without_year_has_no_question_mark(tmp_path):
    text = _summary(tmp_path, {"id": "clip_ab12", "public_domain": False})
    assert "Film: clip_ab12, public_domain=False" in text and "(?)" not in text
    text = _summary(tmp_path, {"id": "clip_ab12", "title": "My Clip", "public_domain": False})
    assert "Film: My Clip, public_domain=False" in text and "(?)" not in text


def test_summary_with_year_keeps_it(tmp_path):
    text = _summary(tmp_path, {"id": "pd", "title": "PD Film", "year": 1968, "public_domain": False})
    assert "Film: PD Film (1968), public_domain=False" in text
