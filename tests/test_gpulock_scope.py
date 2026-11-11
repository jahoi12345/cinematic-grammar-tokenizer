"""a recorded decision (U2): the GPU lock is held only while a stage that uses the GPU runs. Synthetic stages, a tmp lock path; no model, no video."""
import json

import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline import stages as real_stages
from cgt.pipeline.orchestrator import EXIT, Stage, Status


@pytest.fixture
def env(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "pd", "title": "Public", "public_domain": True}]}))
    return tmp_path


def make(root, seen, fail=None, gpu=("g1", "g2"), uses_gpu=None, on_run=None, names=("g1", "mid", "g2", "last"), needs_gpu=None):
    """g1 (GPU) -> mid (CPU) -> g2 (GPU) -> last (CPU). Each run records whether the lock file exists while it runs."""
    lock = root / "gpu.lock"

    def stage(name, prev):
        marker = root / f"out_{name}"

        def status(ctx):
            return Status(marker.exists(), f"key-{name}", "marker", [marker] if marker.exists() else [])

        def run(ctx):
            seen.append((name, lock.exists() and "run_film:pd" in lock.read_text()))
            if on_run:
                on_run(name)
            if fail == name:
                raise RuntimeError("boom")
            marker.write_text(name)
            return {"n": 1}

        def prereq(ctx):
            return [("unavailable_or_prerequisite", f"needs:{prev}")] if prev and not (root / f"out_{prev}").exists() else []
        return Stage(name, name in gpu, status, run, prereq, uses_gpu=(uses_gpu or {}).get(name), needs_gpu=(needs_gpu or {}).get(name))
    return {n: stage(n, names[i - 1] if i else None) for i, n in enumerate(names)}


@pytest.fixture(autouse=True)
def synthetic_order(monkeypatch):
    monkeypatch.setattr(orch, "ORDER", ["g1", "mid", "g2", "last"])


def go(env, reg, **kw):
    logs = []
    code = orch.run("pd", kw.pop("stages", None), reg, root=env, lock_path=env / "gpu.lock", log=logs.append, **kw)
    return code, "\n".join(logs)


def test_lock_held_only_during_gpu_stages_not_during_the_cpu_stage_between_them(env):
    seen = []
    code, log = go(env, make(env, seen))
    assert code == 0
    assert seen == [("g1", True), ("mid", False), ("g2", True), ("last", False)]
    assert not (env / "gpu.lock").exists()
    assert "GPU lock acquired for [g1]" in log and "GPU lock acquired for [g2]" in log and "acquired for [mid]" not in log


def test_lock_released_after_each_gpu_stage_and_on_failure(env):
    seen = []
    code, _ = go(env, make(env, seen, fail="g2"))
    assert code == EXIT["stage_failed"] and seen == [("g1", True), ("mid", False), ("g2", True)]
    assert not (env / "gpu.lock").exists()
    assert json.loads((env / "data/cache/pipeline/pd/g2.json").read_text())["status"] == "failed"


def test_lock_released_right_after_failure_before_anything_else(env, monkeypatch):
    """The lock is gone by the time the failure is recorded (not only at the end of the run)."""
    seen, at_record = [], {}
    inner = orch._write_manifest

    def spy(ctx, name, rec):
        if rec.get("status") == "failed":
            at_record[name] = (env / "gpu.lock").exists()
        return inner(ctx, name, rec)
    monkeypatch.setattr(orch, "_write_manifest", spy)
    go(env, make(env, seen, fail="g1"))
    assert at_record == {"g1": False}


def test_refused_at_start_when_held_nothing_started(env):
    (env / "gpu.lock").write_text("other-job pid=1 nonce=abc\n")
    seen = []
    code, log = go(env, make(env, seen))
    assert code == EXIT["gpu_lock_held"] and seen == [] and "REFUSED [gpu_lock_held]" in log and "other-job" in log
    assert (env / "gpu.lock").read_text().startswith("other-job")
    assert not (env / "data").exists()


def test_cpu_only_run_ignores_a_held_lock(env):
    (env / "gpu.lock").write_text("other-job nonce=abc\n")
    (env / "out_g1").write_text("g1")
    seen = []
    code, _ = go(env, make(env, seen), stages=["g1", "mid"])
    assert code == 0 and seen == [("mid", False)]


def test_clean_stop_exit_3_when_held_mid_run_then_resume(env):
    seen = []

    def grab(name):  # another job takes the GPU while the CPU stage runs
        if name == "mid":
            (env / "gpu.lock").write_text("intruder pid=2 nonce=zzz\n")
    code, log = go(env, make(env, seen, on_run=grab))
    assert code == EXIT["gpu_lock_held"]
    assert seen == [("g1", True), ("mid", False)]  # g2 never started
    assert "STOPPED [gpu_lock_held] before [g2]" in log and "intruder" in log and "resume" in log
    assert (env / "gpu.lock").read_text().startswith("intruder")  # never removed
    assert (env / "out_g1").exists() and (env / "out_mid").exists() and not (env / "out_g2").exists()
    assert json.loads((env / "data/cache/pipeline/pd/g2.json").read_text())["status"] == "gpu_lock_held"
    assert "exit code 3" in (env / "results/films/pd.md").read_text()
    # the other job finishes; the same command resumes from g2 (g1 and mid are cache hits)
    (env / "gpu.lock").unlink()
    seen.clear()
    code, _ = go(env, make(env, seen))
    assert code == 0 and seen == [("g2", True), ("last", False)] and not (env / "gpu.lock").exists()


def test_dry_run_says_which_stages_hold_the_lock(env):
    code, log = go(env, make(env, []), dry_run=True)
    assert code == 0 and "GPU lock: held only while g1, g2 run" in log and "every other stage runs without it" in log
    assert not (env / "gpu.lock").exists() and not (env / "data").exists()
    # g1 cached, nothing upstream runs: only g2 is listed
    (env / "out_g1").write_text("g1")
    assert "held only while g2 runs" in go(env, make(env, []), dry_run=True)[1]
    # everything cached: no stage takes the lock
    for n in ("mid", "g2", "last"):
        (env / f"out_{n}").write_text(n)
    assert "GPU lock: no stage of this run will take it" in go(env, make(env, []), dry_run=True)[1]


def test_uses_gpu_marks_a_cpu_reported_stage_as_a_lock_stage_without_changing_its_resource(env):
    """p2's case: `gpu` stays False (manifests and summary unchanged) but it holds the lock while it runs on MPS."""
    seen = []
    reg = make(env, seen, gpu=("g2",), uses_gpu={"g1": lambda ctx: True})
    code, _ = go(env, reg)
    assert code == 0 and seen[0] == ("g1", True)
    assert json.loads((env / "data/cache/pipeline/pd/g1.json").read_text())["gpu"] is False
    assert "| g1 | CPU | ran" in (env / "results/films/pd.md").read_text()


def test_p2_and_p3c_are_lock_stages_in_the_real_registry():
    reg = real_stages.registry()
    assert reg["p2"].uses_gpu is real_stages.p2_uses_gpu and reg["p2"].gpu is False
    assert reg["p3c"].gpu is True and reg["p3c"].needs_gpu is real_stages.p3c_needs_gpu
    for n in ("p1", "move", "tokens"):
        assert reg[n].gpu is False and reg[n].needs_gpu is None and reg[n].uses_gpu is None
    assert orch.locks_gpu(Stage("p3c", True, None, None, needs_gpu=lambda ctx: True), None) is True
    assert orch.locks_gpu(Stage("p3c", True, None, None, needs_gpu=lambda ctx: False), None) is False  # table rebuild from a valid run: CPU
    assert orch.locks_gpu(reg["move"], None) is False


def test_real_stage_names_p2_and_p3c_hold_the_lock_move_does_not(env, monkeypatch):
    """Synthetic stages with the real names, flags and ORDER: p2 (on MPS) and p3c hold the lock while they run; p1, move and tokens do not."""
    names = ("p1", "p2", "p3c", "move", "tokens")
    monkeypatch.setattr(orch, "ORDER", list(names))
    seen = []
    reg = make(env, seen, names=names, gpu=("p3c",), uses_gpu={"p2": lambda ctx: True}, needs_gpu={"p3c": lambda ctx: True})
    code, log = go(env, reg)
    assert code == 0
    assert seen == [("p1", False), ("p2", True), ("p3c", True), ("move", False), ("tokens", False)]
    assert "GPU lock: held only while p2, p3c run" in log and not (env / "gpu.lock").exists()
