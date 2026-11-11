"""an earlier review independent checks for a sibling working copy (F1 grpo.json move prerequisite, F4 summary year), a verification record.
Synthetic tmp roots only; no real label file is touched. Skips cleanly on a tree without the F1 fix (no MOVE_PREREQ_FILES)."""
import builtins
import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import EXIT, Ctx, Stage, Status

pytestmark = pytest.mark.skipif(not hasattr(stages, "MOVE_PREREQ_FILES"), reason="F1 not present")
GRPO = "data/external/shotqa/grpo.json"


def _root(root: Path, grpo: bool) -> Ctx:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "vf", "title": "VF", "public_domain": True, "path": "vf.mp4"}]}))
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    for rel in stages.MOVE_CODE_FILES + stages.MOVE_CONFIG_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(rel.encode())
    (root / "vf.mp4").write_bytes(b"x")
    pd.DataFrame({"shot_id": [0], "start_s": [0.0], "end_s": [1.0], "len_s": 1.0}).to_parquet(root / "data/processed/vf_shots.parquet")
    if grpo:
        (root / GRPO).parent.mkdir(parents=True, exist_ok=True)
        (root / GRPO).write_text("NOT JSON {")  # a parser would fail
    return Ctx("vf", {"id": "vf", "title": "VF", "public_domain": True, "path": "vf.mp4"}, root=root, log=lambda m: None)


@pytest.fixture(autouse=True)
def _mem(monkeypatch):
    monkeypatch.setattr(stages, "_free_percent", lambda: 90)


def test_prereq_file_list_is_exactly_grpo_and_outside_identity():
    assert stages.MOVE_PREREQ_FILES == [GRPO]
    assert GRPO not in stages.MOVE_CODE_FILES + stages.MOVE_CONFIG_FILES


def test_grpo_never_opened_or_parsed(tmp_path, monkeypatch):
    ctx = _root(tmp_path, grpo=True)
    touched = []
    real_open, real_io_open, real_os_open, real_read_bytes, real_read_text, real_load = builtins.open, io.open, os.open, Path.read_bytes, Path.read_text, json.load

    def guard(p):
        if str(p).endswith("grpo.json"):
            touched.append(str(p))
            raise AssertionError("grpo.json was opened")
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (guard(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(io, "open", lambda f, *a, **k: (guard(f), real_io_open(f, *a, **k))[1])
    monkeypatch.setattr(os, "open", lambda f, *a, **k: (guard(f), real_os_open(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self: (guard(self), real_read_bytes(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (guard(self), real_read_text(self, *a, **k))[1])
    monkeypatch.setattr(json, "load", lambda f, *a, **k: (guard(getattr(f, "name", "")), real_load(f, *a, **k))[1])
    assert stages.move_prereq(ctx) == []
    stages._move_code_identity(ctx)
    assert touched == []


def test_present_grpo_gives_same_prereqs_as_without_the_check(tmp_path, monkeypatch):
    ctx = _root(tmp_path, grpo=True)
    with_check = stages.move_prereq(ctx)
    monkeypatch.setattr(stages, "MOVE_PREREQ_FILES", [])  # = main behaviour
    assert with_check == stages.move_prereq(ctx) == []
    (tmp_path / "vf.mp4").unlink()
    a = stages.move_prereq(ctx)
    monkeypatch.undo()
    monkeypatch.setattr(stages, "_free_percent", lambda: 90)
    assert stages.move_prereq(ctx) == a and a


def test_absent_grpo_full_run_exit5_no_stage_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    ctx = _root(tmp_path, grpo=False)
    called = []
    monkeypatch.setattr(stages, "_run_move_script", lambda c, cmd: (called.append("script"), 0)[1])
    reg = stages.registry()
    for name in ("p1", "p2", "p3c", "tokens"):
        reg[name] = Stage(name, name == "p3c", lambda c: Status(False, "k", "miss"), lambda c, n=name: (called.append(n), {})[1], lambda c: [])
    inner = reg["move"].run
    reg["move"].run = lambda c: (called.append("move"), inner(c))[1]
    logs = []
    assert orch.run("vf", None, reg, root=tmp_path, log=logs.append, lock_path=tmp_path / "l.lock") == EXIT["unavailable_or_prerequisite"] == 5
    assert called == [] and any(GRPO in m for m in logs)
    before = sorted(p.as_posix() for p in tmp_path.rglob("*"))
    assert orch.run("vf", ["move"], reg, dry_run=True, root=tmp_path, log=logs.append) == 5
    assert sorted(p.as_posix() for p in tmp_path.rglob("*")) == before and called == []
    # with the fake present the same full run gets past prerequisites (stages are then invoked)
    (tmp_path / GRPO).parent.mkdir(parents=True)
    (tmp_path / GRPO).write_bytes(b"")
    assert stages.move_prereq(ctx) == []


@pytest.mark.parametrize("film,want,bad", [
    ({"id": "c1", "public_domain": False}, "Film: c1, public_domain=False", "(?)"),
    ({"id": "c1", "title": "T", "year": "", "public_domain": True}, "Film: T, public_domain=True", "()"),
    ({"id": "c1", "title": "T", "year": 1968, "public_domain": True}, "Film: T (1968), public_domain=True", "(?)"),
])
def test_summary_label(tmp_path, film, want, bad):
    text = orch.write_summary(Ctx(film["id"], film, root=tmp_path, log=lambda m: None), [], [], 1.0, 0).read_text()
    assert want in text and bad not in text
