"""F-C1, F-C2, F-D1 fixes (docs/v1_fixes_2026-10-13.md). Synthetic only: a temporary registry root, stub stages and stub owner probes,
synthetic npz files. No model, no GPU, no film."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cgt.pipeline import adhoc, orchestrator, stages
from cgt.pipeline import registry as reg
from cgt.pipeline.orchestrator import Ctx, PipelineError, Stage, Status

REPO = Path(__file__).resolve().parents[1]
needs_ff = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "repo"
    (r / "configs").mkdir(parents=True)
    (r / "configs/films.yaml").write_text(yaml.safe_dump({"films": [
        {"id": "pd_film", "title": "PD", "public_domain": True, "path": "data/raw/films/pd_film/film.mp4", "cut_threshold": 12},
        {"id": "owned_film", "title": "Owned", "public_domain": False, "path": "data/raw/films/owned_film/film.mp4"}]}))
    return r


# ---------------------------------------------------------------- F-C1
def _move_cmd(root, monkeypatch, fid):
    seen = []
    monkeypatch.setattr(stages, "_free_percent", lambda: 99)
    monkeypatch.setattr(stages, "_run_move_script", lambda ctx, cmd: seen.append(cmd) or 1)
    with pytest.raises(PipelineError):
        stages.move_run(Ctx(fid, orchestrator.film_entry(fid, root), root=root, allow_owned=True))
    return seen[0]


@pytest.mark.parametrize("fid,pd_flag,expected", [("pdclaim", True, True), ("ownclip", False, True), ("pd_film", None, False), ("owned_film", None, True)])
def test_move_allow_owned_for_owned_and_every_overlay_film(root, monkeypatch, fid, pd_flag, expected):
    reg.write_overlay(root, [{"id": "pdclaim", "path": "/a.mp4", "public_domain": True}, {"id": "ownclip", "path": "/b.mp4"}])
    assert ("--allow-owned" in _move_cmd(root, monkeypatch, fid)) is expected


# ---------------------------------------------------------------- F-C2: plan header and p1 threshold source
def test_plan_header_marks_overlay_films_local_only(root):
    reg.write_overlay(root, [{"id": "pdclaim", "path": "/a.mp4", "public_domain": True}, {"id": "ownclip", "path": "/b.mp4"}])
    head = {f: orchestrator.format_plan(Ctx(f, orchestrator.film_entry(f, root), root=root), []) for f in ("pdclaim", "ownclip", "pd_film", "owned_film")}
    assert "LOCAL ONLY" in head["pdclaim"] and reg.OVERLAY in head["pdclaim"]
    assert "LOCAL" in head["ownclip"] and "LOCAL" in head["owned_film"]
    assert "LOCAL" not in head["pd_film"]


def test_p1_status_names_the_real_threshold_source(root):
    reg.write_overlay(root, [{"id": "clip", "path": "/a.mp4", "cut_threshold": 27.0, "cut_threshold_source": "default, unchecked"},
                             {"id": "clip2", "path": "/b.mp4", "cut_threshold": 30.0, "cut_threshold_source": "user"}])
    why = {f: stages.p1_status(Ctx(f, orchestrator.film_entry(f, root), root=root)).detail for f in ("clip", "clip2", "pd_film", "owned_film")}
    assert "threshold 27 from configs/films.local.yaml (overlay, default, unchecked)" in why["clip"] and "films.yaml" not in why["clip"]
    assert "threshold 30 from configs/films.local.yaml (overlay, user)" in why["clip2"]
    assert "threshold 12 from configs/films.yaml" in why["pd_film"]
    assert "threshold 27 from the default" in why["owned_film"]


# ---------------------------------------------------------------- F-C2: owner probe in the dry run
def _clip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x48:r=10:d=1", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _stub_registry(root, calls):
    def stage(name):
        def marker(ctx):
            return root / "data/processed" / f"{ctx.film_id}_{name}.stub"

        def status(ctx):
            return Status(marker(ctx).exists(), f"key-{name}", "stub", [marker(ctx)] if marker(ctx).exists() else [])

        def run(ctx):
            calls.append(name)
            marker(ctx).parent.mkdir(parents=True, exist_ok=True)
            marker(ctx).write_text("x")
            return {}
        return Stage(name, False, status, run)
    return {n: stage(n) for n in ("p1", "p2", "p3c", "move", "tokens")}


def _go(root, *argv, probes):
    tv = _load("tokenize_video_v1fix", "scripts/tokenize_video.py")
    logs, calls = [], []
    code = tv.main([str(a) for a in argv], root=root, stage_registry=_stub_registry(root, calls), probes=probes, log=logs.append, lock_path=root / "gpu.lock")
    return code, "\n".join(logs), calls


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")}


@needs_ff
def test_dry_run_prints_owner_blockers_as_warning(root, tmp_path):
    clip = _clip(tmp_path / "in" / "c.mp4")
    ok = {s: (lambda fid: None) for s in adhoc.OWNER_STAGES}
    assert _go(root, clip, "--id", "c", "--stages", "p1", probes=ok)[0] == 0  # registers it
    blocked = {**ok, "move": lambda fid: "film_move reads configs/films.yaml only"}
    before = _snapshot(root)
    code, log, calls = _go(root, clip, "--id", "c", "--dry-run", probes=blocked)
    assert code == 0 and not calls and _snapshot(root) == before
    assert "WARNING: a real run would STOP" in log and "move: film_move reads configs/films.yaml only" in log
    code, log, _ = _go(root, clip, "--id", "c", "--dry-run", "--stages", "p1,p2", probes=blocked)
    assert "WARNING: a real run would STOP" not in log and "owner probe: every probed stage resolves" in log
    code, log, _ = _go(root, clip, "--id", "c", "--dry-run", probes=ok)
    assert "owner probe: every probed stage resolves" in log and "LOCAL" in log


@needs_ff
def test_dry_run_of_a_new_entry_says_the_probe_cannot_run(root, tmp_path):
    clip = _clip(tmp_path / "in" / "c.mp4")
    probed = []
    probes = {s: (lambda fid, s=s: probed.append(s)) for s in adhoc.OWNER_STAGES}
    code, log, _ = _go(root, clip, "--id", "newc", "--dry-run", probes=probes)
    assert code == 0 and not probed and "owner probe not run" in log and not (root / reg.OVERLAY).exists()


# ---------------------------------------------------------------- F-D1
def _case(tmp_path, arrays):
    scratch, acc = tmp_path / "scratch", tmp_path / "acc"
    for kind in ("flow", "depth"):
        for base in (scratch, acc):
            (base / kind).mkdir(parents=True, exist_ok=True)
            np.savez(base / kind / "shot0.k.r24.npz", **arrays)
    t = pd.DataFrame({"shot_id": [0], "p_static": [0.25], "move_atoms": ["static"]}).set_index("shot_id")
    return scratch, acc, t, t.copy()


@pytest.mark.parametrize("arrays,n,fails", [({"a": np.zeros((0,))}, 0, True), ({"a": np.zeros((0, 3)), "b": np.zeros((0,))}, 0, True),
                                            ({"a": np.zeros((0,)), "b": np.arange(3.0)}, 2, False)])
def test_cold_compare_counts_only_non_empty_arrays(tmp_path, arrays, n, fails):
    cold = _load("cold_evidence_v1fix", "scripts/run_move_film_cold_evidence.py")
    r = cold.compare(*_case(tmp_path, arrays))
    assert r["n_arrays"] == n and bool(r["failures"]) is fails
    if fails:
        assert any("no cache array was compared (zero-size" in m for m in r["failures"])
