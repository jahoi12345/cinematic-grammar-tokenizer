"""CPU self-check of the unattended supervisor (Amendment 7) with a STUB trainer in a temporary root (no model, no GPU, own lock file, back-off scaled to milliseconds): watchdog failures are
restarted up to 15 times in total with the documented back-off ladder and a process snapshot per crash; a non-watchdog crash is not restarted once the old limit is exhausted; guard and
stopping-rule exit codes (4, 13) are never restarted; four consecutive restarts without progress stop the run; the lock is released in every case.
Writes results/phase4/supervise_selfcheck.md. Usage: uv run python scripts/p4_video_supervise_selfcheck.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
WD = "RuntimeError: [METAL] Command buffer execution failed: Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)."
STUB = '''
import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
stage = sys.argv[sys.argv.index("--stage") + 1]
tag = sys.argv[sys.argv.index("--tag") + 1]
ctl = root / "control" / f"{stage}.json"
plan = json.loads(ctl.read_text())
step = plan.pop(0)
ctl.write_text(json.dumps(plan))
logd = root / "results/phase4/runs" / f"video_{tag}" / stage
logd.mkdir(parents=True, exist_ok=True)
run_dir = root / "data/cache/p4/video_runs" / tag / stage
run_dir.mkdir(parents=True, exist_ok=True)
lines = (logd / "updates.jsonl").read_text().splitlines() if (logd / "updates.jsonl").exists() else []
last = int(json.loads(lines[-1])["update"]) if lines else 0
with open(logd / "updates.jsonl", "a") as f:
    for i in range(step.get("progress", 0)):
        f.write(json.dumps({"update": last + i + 1}) + "\\n")
if step.get("stdout"):
    print(step["stdout"], flush=True)
if step["exit"] == 0:
    (run_dir / "DONE").write_text("done")
sys.exit(step["exit"])
'''


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


def scenario(steps, restarts_used, stages="fold2", history=0, extra_env=None, wait=True):
    root = Path(tempfile.mkdtemp())
    (root / "scripts").mkdir()
    (root / "scripts/p4_video_train.py").write_text(STUB)
    (root / "control").mkdir()
    for s in stages.split(","):
        (root / "control" / f"{s}.json").write_text(json.dumps(steps[s] if isinstance(steps, dict) else steps))
    if history:
        hd = root / "results/phase4/runs/video_t"
        hd.mkdir(parents=True, exist_ok=True)
        (hd / "restarts.jsonl").write_text("".join(json.dumps({"restart": i + 1, "stage": "fold2", "exit_code": 1, "watchdog": True}) + "\n" for i in range(history)))
    env = {**os.environ, **(extra_env or {}), "P4_SUP_ROOT": str(root), "P4_SUP_BACKOFF_SCALE": "0.001", "P4_SUP_POLL_S": "0.05", "P4_SUP_NO_CAFFEINATE": "1", "CGT_GPU_LOCK": str(root / "lock")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/p4_video_supervise.py"), "--tag", "t", "--frames", "8", "--tokens", "64", "--stages", stages, *(["--restarts-used", str(restarts_used)] if restarts_used is not None else [])],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False)
    top = root / "results/phase4/runs/video_t"
    rj = [json.loads(x) for x in (top / "restarts.jsonl").read_text().splitlines()] if (top / "restarts.jsonl").exists() else []
    sj = [json.loads(x) for x in (top / "supervisor.jsonl").read_text().splitlines()] if (top / "supervisor.jsonl").exists() else []
    return r, rj, sj, root, top


wd = {"exit": 1, "stdout": WD, "progress": 5}
r, rj, sj, root, top = scenario([wd, wd, wd, {"exit": 0, "progress": 5}], 6)
back = [e["seconds"] for e in sj if e.get("event") == "back-off before restart"]
check("3 watchdog crashes (with progress) are restarted, then the stage finishes (exit 0, DONE, lock released)", r.returncode == 0 and len(rj) == 3 and all(x["watchdog"] for x in rj) and (root / "data/cache/p4/video_runs/t/fold2/DONE").exists() and not (root / "lock").exists(), f"exit {r.returncode}, restarts {[x['restart'] for x in rj]} (continuing the count from 6)")
check("back-off ladder 2, 4, 8 minutes and the process snapshot is recorded at every crash", back == [120, 240, 480] and all({"by_cpu", "by_mem"} <= set(x["heaviest_processes"]) and x["heaviest_processes"]["by_cpu"] for x in rj), f"back-off {back}; snapshot keys {sorted(rj[0]['heaviest_processes'])}")
r, rj, sj, root, top = scenario([{"exit": 1, "stdout": "RuntimeError: out of memory", "progress": 3}, {"exit": 0}], 6)
check("a NON-watchdog crash is not restarted once the old limit of 5 is exhausted (exit 11)", r.returncode == 11 and len(rj) == 1 and rj[0]["watchdog"] is False and not (root / "lock").exists(), f"exit {r.returncode}, watchdog flag {rj[0]['watchdog'] if rj else None}")
for code, label in ((4, "leak probe failure (4)"), (13, "resume loss mismatch (13)"), (3, "stopping rule (3)"), (10, "dirty tree (10)"), (9, "the trainer's testing stop code (9)"), (5, "data root (5)"), (6, "unreadable image (6)"), (7, "adapter hash (7)"), (8, "token length / skipped example (8)")):
    r, rj, sj, root, top = scenario([{"exit": code, "stdout": WD}, {"exit": 0}], 6)
    check(f"{label} is never restarted, even if the output contains the watchdog text", r.returncode == code and not rj and not (root / "lock").exists(), f"exit {r.returncode}, restarts {len(rj)}")
r, rj, sj, root, top = scenario([{**wd, "progress": 0}] * 6, 6)
check("four consecutive watchdog restarts WITHOUT progress stop the run (exit 12)", r.returncode == 12 and len(rj) == 4 and not (root / "lock").exists(), f"exit {r.returncode}, restarts {len(rj)}")
r, rj, sj, root, top = scenario([wd, wd, {"exit": 0}], 13)
check("limit 15 in total: with 13 restarts used, two more watchdog restarts are allowed (13 -> 15)", r.returncode == 0 and [x["restart"] for x in rj] == [14, 15], f"restarts {[x['restart'] for x in rj]}, exit {r.returncode}")
r, rj, sj, root, top = scenario([wd, wd, wd, {"exit": 0}], 13)
check("the 16th restart is refused (exit 11)", r.returncode == 11 and [x["restart"] for x in rj] == [14, 15, 16], f"restarts {[x['restart'] for x in rj]}, exit {r.returncode}")
r, rj, sj, root, top = scenario({"fold2": [{"exit": 0}], "final": [{"exit": 0}]}, 6, "fold2,final")
check("stages run in order and the supervisor finishes with the lock released", r.returncode == 0 and (root / "data/cache/p4/video_runs/t/final/DONE").exists() and not (root / "lock").exists(), f"exit {r.returncode}")
# ---- V5: the cumulative restart history is file-backed
r, rj, sj, root, top = scenario([wd, {"exit": 0}], None, history=15)
check("V5: with 15 restarts on record and NO --restarts-used flag, the next watchdog failure is refused (the 16th), the count is read from restarts.jsonl", r.returncode == 11 and rj and rj[-1]["restart"] == 16, f"exit {r.returncode}, last restart {rj[-1]['restart'] if rj else None}")
r, rj, sj, root, top = scenario([wd, {"exit": 0}], 0, history=15)
check("V5: an understated --restarts-used 0 cannot reset a history of 15 (the history wins, the flag is ignored and recorded; the 16th is refused)", r.returncode == 11 and rj[-1]["restart"] == 16 and any(e.get("flag_understated_and_ignored") for e in sj), f"exit {r.returncode}")
r, rj, sj, root, top = scenario([{"exit": 0}], 3, history=6)
check("V5: --restarts-used 3 below the 6 recorded restarts is ignored (history 6 counts) and the run proceeds from 6", r.returncode == 0 and len(rj) == 6 and any(e.get("restarts_already_used") == 6 for e in sj), f"exit {r.returncode}")
r, rj, sj, root, top = scenario([{"exit": 0}], -1, history=0)
check("V5: a negative --restarts-used is refused (exit 2)", r.returncode == 2, f"exit {r.returncode}")
r, rj, sj, root, top = scenario([wd, wd, {"exit": 0}], None, history=13)
check("V5: with 13 on record and no flag, two more watchdog restarts are allowed and counted from the history (14, 15)", r.returncode == 0 and [x["restart"] for x in rj][-2:] == [14, 15] and len(rj) == 15, f"restarts {[x['restart'] for x in rj][-3:]}")
# ---- V4: only a lock this process acquired is ever removed; the supervisor honours CGT_GPU_LOCK and never touches the real path
real_lock = Path("/tmp/cgt_gpu.lock")
before = (real_lock.read_text(), real_lock.stat().st_mtime_ns) if real_lock.exists() else None
root = Path(tempfile.mkdtemp())
(root / "scripts").mkdir()
(root / "scripts/p4_video_train.py").write_text(STUB)
(root / "control").mkdir()
(root / "control/fold2.json").write_text(json.dumps([{"exit": 0}]))
other = root / "lock"
other.write_text("p4-video-train pid=999999 nonce=someoneelse started=x tag=other\n")
env = {**os.environ, "P4_SUP_ROOT": str(root), "P4_SUP_BACKOFF_SCALE": "0.001", "P4_SUP_POLL_S": "0.05", "P4_SUP_NO_CAFFEINATE": "1", "CGT_GPU_LOCK": str(other)}
import signal as _sig
import time as _time

proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/p4_video_supervise.py"), "--tag", "t", "--frames", "8", "--tokens", "64", "--stages", "fold2"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
_time.sleep(4)
proc.send_signal(_sig.SIGTERM)
proc.communicate(timeout=60)
check("V4: a supervisor WAITING for another owner's lock (same job name, other pid) and then terminated does not remove that lock", other.exists() and "pid=999999" in other.read_text() and not (root / "data/cache/p4/video_runs/t/fold2/DONE").exists(), f"exit {proc.returncode}, lock still names pid 999999: {other.exists()}")
from cgt.p4_camera_move import gpulock as GL

GL.LOCK = root / "lock2"
ok1 = GL.acquire("p4-video-train", "t") and GL.owned()
(root / "lock2").write_text("p4-video-train pid=424242 nonce=thief started=x tag=t\n")  # another process replaced the lock text
ok2 = (not GL.owned()) and (not GL.release("p4-video-train")) and (root / "lock2").exists()
(root / "lock2").unlink()
ok3 = GL.acquire("p4-video-train", "t") and GL.release("p4-video-train") and not (root / "lock2").exists() and not GL.release("p4-video-train")
check("V4: gpulock.release removes only a lock carrying THIS process's pid and nonce (a replaced lock of the same job name stays; our own lock is removed once)", ok1 and ok2 and ok3)
after = (real_lock.read_text(), real_lock.stat().st_mtime_ns) if real_lock.exists() else None
check("the real /tmp/cgt_gpu.lock was never touched by this self-check (content and mtime identical, or still absent)", before == after, "sentinel: unchanged" if before == after else "CHANGED")
(ROOT / "results/phase4/supervise_selfcheck.md").write_text("# Supervisor self-check (stub trainer, temporary root, CPU; Amendment 7 rules)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
    f"| {n} | **{'PASS' if ok else 'FAIL'}** | {d.replace('|', '/')} |" for n, ok, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(x[1] for x in rows)}.\n")
raise SystemExit(0 if all(x[1] for x in rows) else 1)
