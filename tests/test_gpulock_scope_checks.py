"""an earlier review for a recorded decision U2, branch a sibling working copy: the GPU lock is held only while a GPU stage runs.
Own synthetic pipeline (tmp root, stub stages, tmp lock path); no model, no video, never /tmp/cgt_gpu.lock."""
import json
import subprocess
import sys

import pytest
import yaml

from cgt.pipeline import gpulock
from cgt.pipeline import orchestrator as orch
from cgt.pipeline.orchestrator import EXIT, Stage, Status

ORDER = ["a_cpu", "b_gpu", "c_cpu", "d_gpu", "e_cpu"]
FOREIGN = "other_job pid=99999 started=2026-10-15T00:00:00+0200 nonce=deadbeefdeadbeef\n"


@pytest.fixture(autouse=True)
def order(monkeypatch):
    monkeypatch.setattr(orch, "ORDER", ORDER)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "vr", "title": "VR", "public_domain": True}]}))
    return tmp_path


def build(root, obs, fail=None, hooks=None, gpu=("b_gpu", "d_gpu"), uses_gpu=None):
    """Each stage's run records the lock file state (absent / ours / foreign) while it runs. `hooks[name](root)` runs inside that stage's run."""
    lock = root / "gpu.lock"

    def lockstate():
        if not lock.exists():
            return "absent"
        return "ours" if "run_film:vr" in lock.read_text() else "foreign"

    def mk(i, name):
        out = root / f"o_{name}.txt"
        prev = ORDER[i - 1] if i else None

        def status(ctx):
            return Status(out.exists(), f"k-{name}", "", [out] if out.exists() else [])

        def run(ctx):
            obs.append((name, lockstate()))
            if hooks and name in hooks:
                hooks[name](root)
            if fail == name:
                raise RuntimeError("synthetic failure")
            out.write_text(f"{name}\n")
            return {"rows": 1}

        def prereq(ctx):
            return [("unavailable_or_prerequisite", f"needs:{prev}")] if prev and not (root / f"o_{prev}.txt").exists() else []
        return Stage(name, name in gpu, status, run, prereq, uses_gpu=(uses_gpu or {}).get(name))
    return {n: mk(i, n) for i, n in enumerate(ORDER)}


def go(root, reg, **kw):
    logs = []
    code = orch.run("vr", None, reg, root=root, lock_path=root / "gpu.lock", log=logs.append, **kw)
    return code, "\n".join(logs)


def manifest(root, name):
    p = root / f"data/cache/pipeline/vr/{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_lock_only_during_gpu_stages(root):
    obs = []
    code, _ = go(root, build(root, obs))
    assert code == 0
    assert obs == [("a_cpu", "absent"), ("b_gpu", "ours"), ("c_cpu", "absent"), ("d_gpu", "ours"), ("e_cpu", "absent")]
    assert not (root / "gpu.lock").exists()


def test_failure_in_gpu_stage_releases(root):
    obs = []
    code, _ = go(root, build(root, obs, fail="b_gpu"))
    assert code == EXIT["stage_failed"]
    assert obs == [("a_cpu", "absent"), ("b_gpu", "ours")]
    assert not (root / "gpu.lock").exists()
    assert manifest(root, "b_gpu")["status"] == "failed" and manifest(root, "c_cpu") is None


def test_failure_in_cpu_stage_never_locks(root):
    obs = []
    code, _ = go(root, build(root, obs, fail="c_cpu"))
    assert code == EXIT["stage_failed"]
    assert obs == [("a_cpu", "absent"), ("b_gpu", "ours"), ("c_cpu", "absent")]
    assert not (root / "gpu.lock").exists()


def test_held_at_start_refuses_and_leaves_holder_untouched(root):
    lock = root / "gpu.lock"
    lock.write_text(FOREIGN)
    before = (lock.read_bytes(), lock.stat().st_mtime_ns, lock.stat().st_ino)
    obs = []
    code, log = go(root, build(root, obs))
    assert code == EXIT["gpu_lock_held"] and "REFUSED [gpu_lock_held]" in log and "Nothing was started" in log
    assert obs == [] and manifest(root, "a_cpu") is None
    assert (lock.read_bytes(), lock.stat().st_mtime_ns, lock.stat().st_ino) == before


def test_held_at_start_when_gpu_stage_is_a_hit_but_upstream_reruns(root):
    """d_gpu is a hit at plan time but c_cpu (upstream) will run and invalidates it: the start check must still refuse."""
    (root / "o_a_cpu.txt").write_text("a_cpu\n")
    (root / "o_b_gpu.txt").write_text("b_gpu\n")
    (root / "o_d_gpu.txt").write_text("d_gpu\n")
    (root / "o_e_cpu.txt").write_text("e_cpu\n")
    obs = []
    reg = build(root, obs, hooks={"c_cpu": lambda r: (r / "o_d_gpu.txt").unlink()})
    (root / "gpu.lock").write_text(FOREIGN)
    code, log = go(root, reg)
    assert code == EXIT["gpu_lock_held"] and "REFUSED [gpu_lock_held]" in log and obs == []


def _take_lock_in_other_process(root):
    code = ("import sys; from pathlib import Path; from cgt.pipeline.gpulock import GpuLock; "
            "GpuLock('other_proc', Path(sys.argv[1])).acquire()")
    subprocess.run([sys.executable, "-c", code, str(root / "gpu.lock")], check=True, env={**__import__("os").environ, "PYTHONPATH": ":".join(sys.path)})


def test_lock_taken_by_other_process_between_stages_stops_cleanly_then_resumes(root):
    obs = []
    code, log = go(root, build(root, obs, hooks={"c_cpu": _take_lock_in_other_process}))
    lock = root / "gpu.lock"
    held = lock.read_bytes()
    assert b"other_proc" in held
    assert code == EXIT["gpu_lock_held"] and "STOPPED [gpu_lock_held] before [d_gpu]" in log and "resume" in log
    assert obs == [("a_cpu", "absent"), ("b_gpu", "ours"), ("c_cpu", "absent")]
    assert manifest(root, "d_gpu")["status"] == "gpu_lock_held" and manifest(root, "e_cpu") is None
    assert lock.read_bytes() == held  # the other process's lock is neither removed nor rewritten
    assert (root / "results/films/vr.md").exists() and "exit code 3" in (root / "results/films/vr.md").read_text()
    lock.unlink()  # the other job finishes
    obs2 = []
    code2, _ = go(root, build(root, obs2))
    assert code2 == 0 and obs2 == [("d_gpu", "ours"), ("e_cpu", "absent")]
    assert manifest(root, "b_gpu")["status"] == "hit"


def test_all_hits_never_touch_the_lock(root, monkeypatch):
    for n in ORDER:
        (root / f"o_{n}.txt").write_text(f"{n}\n")
    calls = []
    real = gpulock.GpuLock.acquire
    monkeypatch.setattr(gpulock.GpuLock, "acquire", lambda self: calls.append(1) or real(self))
    (root / "gpu.lock").write_text(FOREIGN)  # even a held lock does not matter: nothing needs the GPU
    obs = []
    code, log = go(root, build(root, obs))
    assert code == 0 and obs == [] and calls == []
    assert "GPU lock: no stage of this run will take it" in log
    assert (root / "gpu.lock").read_text() == FOREIGN


def test_uses_gpu_marks_a_cpu_reported_stage_for_locking_without_changing_gpu_field(root):
    obs = []
    code, _ = go(root, build(root, obs, uses_gpu={"a_cpu": lambda ctx: True}))
    assert code == 0 and obs[0] == ("a_cpu", "ours") and obs[2] == ("c_cpu", "absent")
    assert manifest(root, "a_cpu")["gpu"] is False


def test_dry_run_names_lock_stages_and_takes_nothing(root, monkeypatch):
    calls = []
    monkeypatch.setattr(gpulock.GpuLock, "acquire", lambda self: calls.append(1))
    code, log = go(root, build(root, []), dry_run=True)
    assert code == 0 and calls == [] and "b_gpu, d_gpu" in log and not (root / "data").exists()


def test_release_never_removes_a_lock_rewritten_by_someone_else(tmp_path):
    p = tmp_path / "l"
    lk = gpulock.GpuLock("run_film:vr", p).acquire()
    p.write_text(FOREIGN)  # lock replaced between our acquire and release
    assert lk.release() is False and p.read_text() == FOREIGN


def test_real_registry_fields_unchanged_except_lock_only_p2():
    from cgt.pipeline import stages
    reg = stages.registry()
    assert {n: s.gpu for n, s in reg.items()} == {"p1": False, "p2": False, "p3c": True, "tokens": False, "move": False}
    assert reg["p3c"].needs_gpu is stages.p3c_needs_gpu
    assert [n for n, s in reg.items() if s.uses_gpu is not None] == ["p2"] and reg["p2"].uses_gpu is stages.p2_uses_gpu
    assert all(s.needs_gpu is None for n, s in reg.items() if n != "p3c")


def test_p2_uses_gpu_false_without_torch(monkeypatch):
    from cgt.pipeline import stages
    monkeypatch.setitem(sys.modules, "torch", None)
    assert stages.p2_uses_gpu(None) is False


def test_lock_released_when_the_running_manifest_write_fails_after_acquire(root, monkeypatch):
    """Between acquire and the stage's own try (log + 'running' manifest) an error must not leak the lock: the outer finally covers it."""
    inner = orch._write_manifest

    def boom(ctx, name, rec):
        if name == "b_gpu" and rec.get("status") == "running":
            assert (root / "gpu.lock").exists()
            raise OSError("disk full")
        return inner(ctx, name, rec)
    monkeypatch.setattr(orch, "_write_manifest", boom)
    with pytest.raises(OSError):
        go(root, build(root, []))
    assert not (root / "gpu.lock").exists()
