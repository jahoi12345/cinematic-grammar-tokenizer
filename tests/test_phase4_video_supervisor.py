"""Execute the supervisor with a CPU stub and a private lock; never the real lock."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cgt.p4_camera_move import gpulock

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = "kIOGPUCommandBufferCallbackErrorImpactingInteractivity"
STUB = '''import json,sys
from pathlib import Path
r=Path.cwd()
c=r/'control.json'
steps=json.loads(c.read_text()); step=steps.pop(0); c.write_text(json.dumps(steps))
stage=sys.argv[sys.argv.index('--stage')+1]
log=r/'results/phase4/runs/video_synthetic'/stage
log.mkdir(parents=True,exist_ok=True)
f=log/'updates.jsonl'
last=json.loads(f.read_text().splitlines()[-1])['update'] if f.exists() and f.read_text().strip() else 0
with f.open('a') as out:
 for i in range(step.get('progress',0)): out.write(json.dumps({'update':last+i+1})+'\\n')
print(step.get('stdout',''),flush=True)
if step['exit']==0:
 d=r/'data/cache/p4/video_runs/synthetic'/stage;d.mkdir(parents=True,exist_ok=True);(d/'DONE').write_text('done')
sys.exit(step['exit'])
'''


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("synthetic_supervisor", ROOT / "scripts/p4_video_supervise.py")
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    monkeypatch.setattr(sup, "ROOT", tmp_path)
    monkeypatch.setattr(gpulock, "LOCK", tmp_path / "private_cpu_lock")
    monkeypatch.setenv("P4_SUP_NO_CAFFEINATE", "1")
    monkeypatch.setenv("P4_SUP_BACKOFF_SCALE", "0")
    monkeypatch.setenv("P4_SUP_POLL_S", "0.001")
    monkeypatch.setattr(sup.signal, "signal", lambda *args: None)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/p4_video_train.py").write_text(STUB)

    def run(steps, used=6, stages="fold2"):
        control = tmp_path / "control.json"
        control.write_text(json.dumps(steps))
        monkeypatch.setattr(sys, "argv", ["supervisor", "--tag", "synthetic", "--frames", "8", "--tokens", "64",
                                        "--stages", stages, "--restarts-used", str(used)])
        with pytest.raises(SystemExit) as exc:
            sup.main()
        assert not gpulock.LOCK.exists()
        directory = tmp_path / "results/phase4/runs/video_synthetic"
        records = lambda name: [json.loads(s) for s in (directory / name).read_text().splitlines()] if (directory / name).exists() else []
        return exc.value.code, records("restarts.jsonl"), records("supervisor.jsonl"), json.loads(control.read_text())
    return run


@pytest.mark.parametrize("code", [3, 4, 5, 6, 7, 8, 9, 10, 13])
def test_guard_exit_never_restarts_even_with_watchdog_text(scenario, code):
    exit_code, crashes, _, remaining = scenario([{"exit": code, "stdout": WATCHDOG}, {"exit": 0}])
    assert exit_code == code and crashes == [] and remaining == [{"exit": 0}], (exit_code, crashes, remaining)


def test_nonwatchdog_failure_refused_after_old_limit(scenario):
    code, crashes, _, remaining = scenario([{"exit": 1, "stdout": "ordinary crash", "progress": 1}, {"exit": 0}])
    assert code == 11 and len(crashes) == 1 and not crashes[0]["watchdog"] and remaining == [{"exit": 0}]


@pytest.mark.parametrize("n,expected", [(2, 0), (3, 11)])
def test_watchdog_limit_is_global_and_fifteenth_is_last(scenario, n, expected):
    code, crashes, events, _ = scenario([{"exit": 1, "stdout": WATCHDOG, "progress": 1}] * n + [{"exit": 0}], used=13)
    assert code == expected and [r["restart"] for r in crashes] == list(range(14, 14 + n))
    assert all(r["watchdog"] and {"by_cpu", "by_mem"} <= r["heaviest_processes"].keys() for r in crashes)
    assert [r["seconds"] for r in events if r["event"] == "back-off before restart"] == [120, 240]


def test_backoff_ladder_and_stale_watchdog_output(scenario):
    steps = [{"exit": 1, "stdout": WATCHDOG, "progress": 1}] * 4
    steps += [{"exit": 1, "stdout": "ordinary crash", "progress": 1}, {"exit": 0}]
    code, crashes, events, remaining = scenario(steps)
    assert code == 11 and crashes[-1]["watchdog"] is False and remaining == [{"exit": 0}]
    assert [r["seconds"] for r in events if r["event"] == "back-off before restart"] == [120, 240, 480, 600]


def test_counter_persists_across_stages(scenario):
    code, crashes, _, _ = scenario([{"exit": 1, "stdout": WATCHDOG, "progress": 1}, {"exit": 0},
                                   {"exit": 1, "stdout": WATCHDOG, "progress": 1}, {"exit": 0}],
                                  used=14, stages="fold2,final")
    assert code == 11 and [r["restart"] for r in crashes] == [15, 16]


def test_four_watchdog_crashes_without_progress_stop(scenario):
    steps = [{"exit": 1, "stdout": WATCHDOG, "progress": 0}] * 5
    code, crashes, _, remaining = scenario(steps)
    assert code == 12 and len(crashes) == 4 and len(remaining) == 1


def test_waiting_supervisor_cannot_release_another_process_lock(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("waiting_supervisor", ROOT / "scripts/p4_video_supervise.py")
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    private_lock = tmp_path / "other_process_lock"
    owner = "p4-video-train pid=999999 started=synthetic tag=other\n"
    private_lock.write_text(owner)
    monkeypatch.setattr(gpulock, "LOCK", private_lock)
    monkeypatch.setattr(sup, "ROOT", tmp_path)
    handlers = {}
    monkeypatch.setattr(sup.signal, "signal", lambda sig, fn: handlers.update({sig: fn}))
    monkeypatch.setattr(sup.time, "sleep", lambda _: handlers[sup.signal.SIGTERM]())
    monkeypatch.setattr(sys, "argv", ["supervisor", "--tag", "synthetic", "--frames", "8", "--tokens", "64"])
    with pytest.raises(SystemExit) as exc:
        sup.main()
    assert exc.value.code == 143
    assert private_lock.exists() and private_lock.read_text() == owner


def test_relaunch_cannot_reset_committed_restart_history(scenario, tmp_path):
    directory = tmp_path / "results/phase4/runs/video_synthetic"
    directory.mkdir(parents=True)
    (directory / "restarts.jsonl").write_text(json.dumps({"restart": 15, "stage": "fold2", "exit_code": 1, "watchdog": True}) + "\n")
    code, _, _, remaining = scenario([{"exit": 1, "stdout": WATCHDOG, "progress": 1}, {"exit": 0}], used=0)
    assert code == 11 and remaining == [{"exit": 0}]


def test_negative_restart_lower_bound_is_refused_before_trainer_launch(scenario):
    code, crashes, _, remaining = scenario([{"exit": 0}], used=-1)
    assert code == 2 and crashes == [] and remaining == [{"exit": 0}]


def test_lock_release_requires_acquired_pid_and_nonce(tmp_path, monkeypatch):
    private = tmp_path / "gpu_lock"
    monkeypatch.setattr(gpulock, "LOCK", private)
    assert gpulock.acquire("synthetic", "test")
    original = private.read_text()
    assert gpulock.owned()
    private.write_text(original.replace(f"pid={gpulock.os.getpid()} ", "pid=999999 "))
    assert not gpulock.release("synthetic")
    private.write_text(original.replace("nonce=", "nonce=altered"))
    assert not gpulock.release("synthetic")
    private.write_text(original)
    assert gpulock.release("synthetic") and not private.exists()


def test_lock_environment_override_is_used_at_import(tmp_path, monkeypatch):
    private = tmp_path / "isolated_lock"
    monkeypatch.setenv("CGT_GPU_LOCK", str(private))
    spec = importlib.util.spec_from_file_location("isolated_lock_module", ROOT / "src/cgt/p4_camera_move/gpulock.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Check before acquire: a regression must never create the real lock.
    assert module.LOCK == private
    assert module.acquire("cpu_synthetic")
    assert module.release() and not private.exists()
