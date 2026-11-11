"""Step B runner (docs/plans/stepb_cinescale2.md section 4, BINDING): holds the shared GPU lock /tmp/cgt_gpu.lock for the whole
sequence (atomic create with owner nonce; never waits, never steals: exit 11 if held), then runs, in order:
  0. pipeline check (B1 stream, first 32 samples = 2 updates; nothing from it is used),
  1. baseline CineScale2 val of the frozen adapter,
  2. B0 -> B1 -> B2: train, then validate the 50% and 100% checkpoints (ShotQA dedup + CineScale2 val),
  3. selection (CPU).
Budget: 10 GPU hours from lock acquisition and a hard stop at --deadline (default 18:00 local today). Before each candidate the
projected time (the previous candidate's measured time, 3.2 h before any is measured) must fit both; otherwise it and every later
candidate are DROPPED and recorded. A child still running at the deadline is terminated and its candidate dropped.
A training crash (exit code not 0/3-10) is resumed once from its resume state. Memory guard: the trainer pauses below 25% free; this
runner also waits for >= 25% before each step. Run under caffeinate and nice:
  nice -n 19 caffeinate -i <venv>/bin/python scripts/stepb_run.py"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.cinescale2.semantic import wait_for_memory
from cgt.pipeline.gpulock import GpuLock, LockHeld
from cgt.stepb.data import RUN_ORDER

PY = sys.executable
LOG = ROOT / "results/stepb/runner.jsonl"
FROZEN = ROOT / "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1/ckpt_100.safetensors"
BUDGET_H, FIRST_PROJECTION_H = 10.0, 3.2
FAIL_CLOSED = {3, 4, 5, 6, 7, 8, 9, 10}


def event(**kw) -> None:
    kw = {"time": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **kw}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def child(args: list[str], deadline: dt.datetime, logf: Path) -> int:
    """Run a child (nice 19) until it exits or the deadline passes (then terminate it; returns -999)."""
    wait_for_memory(25, log=lambda m: event(msg=m), poll_s=60)
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as out:
        p = subprocess.Popen(["nice", "-n", "19", PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                             env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                                  "HOME": str(Path.home())})
        while p.poll() is None:
            if dt.datetime.now().astimezone() >= deadline:
                p.terminate()
                try:
                    p.wait(120)
                except subprocess.TimeoutExpired:
                    p.kill()
                return -999
            time.sleep(15)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="18:00")
    a = ap.parse_args()
    now = dt.datetime.now().astimezone()
    hh, mm = map(int, a.deadline.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    lock = GpuLock("stepb_cinescale2")
    try:
        lock.acquire()
    except LockHeld as e:
        event(msg="GPU lock held, not starting", holder=e.holder.strip())
        sys.exit(11)
    t0 = time.monotonic()
    used_h = lambda: (time.monotonic() - t0) / 3600
    runs = ROOT / "results/stepb/runs"
    try:
        event(msg="lock acquired", deadline=deadline.isoformat())
        rc = child(["scripts/stepb_train.py", "--cand", "B1", "--max-samples", "32", "--tag", "stepb_pipecheck"],
                   deadline, runs / "stepb_pipecheck/stdout.log")
        event(step="pipeline check", rc=rc, hours=round(used_h(), 3))
        if rc != 0:
            event(msg="STOP: pipeline check failed")
            return
        rc = child(["scripts/stepb_eval.py", "--adapter", str(FROZEN), "--sets", "cs2", "--out", str(runs / "baseline/val_frozen")],
                   deadline, runs / "baseline/stdout.log")
        event(step="baseline CineScale2 val", rc=rc, hours=round(used_h(), 3))
        if rc != 0:
            event(msg="STOP: baseline evaluation failed")
            return
        projection = FIRST_PROJECTION_H
        dropped = []
        for k, cand in enumerate(RUN_ORDER):
            now = dt.datetime.now().astimezone()
            if used_h() + projection > BUDGET_H or now + dt.timedelta(hours=projection) > deadline:
                dropped = list(RUN_ORDER[k:])
                event(msg="candidates DROPPED (budget/deadline)", dropped=dropped, used_h=round(used_h(), 3),
                      projection_h=projection, deadline=deadline.isoformat())
                break
            c0 = time.monotonic()
            tag = f"stepb_{cand}"
            logf = runs / tag / "stdout.log"
            rc = child(["scripts/stepb_train.py", "--cand", cand], deadline, logf)
            if rc not in (0, -999, *FAIL_CLOSED):
                event(step=f"{cand} train crashed, resuming once", rc=rc)
                rc = child(["scripts/stepb_train.py", "--cand", cand], deadline, logf)
            event(step=f"{cand} train", rc=rc, hours=round(used_h(), 3))
            if rc != 0:
                event(msg=f"{cand} not finished: dropped (its checkpoints are not eligible)", rc=rc)
                dropped.append(cand)
                if rc == -999:
                    dropped += list(RUN_ORDER[k + 1:])
                    break
                continue
            manifest = runs / tag / "manifest.json"
            ok = True
            for pct in (50, 100):
                ck = ROOT / "data/cache/stepb/runs" / tag / f"ckpt_{pct:03d}.safetensors"
                rc = child(["scripts/stepb_eval.py", "--adapter", str(ck), "--manifest", str(manifest),
                            "--out", str(runs / tag / f"val_ckpt{pct:03d}")], deadline, logf)
                event(step=f"{cand} eval {pct}%", rc=rc, hours=round(used_h(), 3))
                ok = ok and rc == 0
            if not ok:
                event(msg=f"{cand} evaluation incomplete: dropped")
                dropped.append(cand)
            projection = max((time.monotonic() - c0) / 3600, 0.5)
        event(msg="training phase finished", dropped=dropped, used_h=round(used_h(), 3))
    finally:
        removed = lock.release()
        event(msg="lock released", removed=removed, used_h=round(used_h(), 3))
    rc = subprocess.run([PY, "scripts/stepb_select.py"], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin",
                                                                      "HOME": str(Path.home())}, check=False).returncode
    event(step="selection", rc=rc)


if __name__ == "__main__":
    main()
