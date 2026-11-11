"""Base-generation matched-budget runner (docs/plans/base_gen_matched_budget.md section 4, BINDING). Training only: validation is
run by the author afterwards with scripts/base_gen_eval.py (plan section 5).
GPU lock /tmp/cgt_gpu.lock (cgt.pipeline.gpulock), held only while the child runs; polls every 60 s if another job holds it.
Budget ENFORCED here: --deadline (C.TRAIN_DEADLINE, 09:30 CET) and --cap-h (C.GPU_CAP_H) GPU hours (cgt.stepc.select.Budget).
Before the first launch the projection (updates to the target x --sec-per-update + 0.1 h load) must fit both, else training is
DROPPED and recorded (never shortened). A child still running at the stop time is terminated (checkpoints already saved stay valid:
a 25% checkpoint does not depend on later updates). A crash (exit not 0/3-10) is resumed from the resume state (every 25 updates)
at most C.MAX_TRAIN_RESUMES times. Events: results/base_gen/runner.jsonl (speed check: results/base_gen/speedcheck/runner.jsonl).
--speedcheck: 10 updates on the first 160 samples of the stream (tag base_gen_speed, fresh), then 20 dedup_v2 val questions scored
with the base model (timing only, no accuracy); summary to results/base_gen/speedcheck/summary.json.
  cd <worktree> && HF_HUB_OFFLINE=1 nohup caffeinate -i nice -n 19 <venv>/bin/python scripts/base_gen_run.py --target-pct 25 \
      --sec-per-update <measured> > results/base_gen/runner_stdout.log 2>&1 &"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.base_gen import config as C
from cgt.cinescale2.semantic import wait_for_memory
from cgt.pipeline.gpulock import GpuLock, LockHeld
from cgt.stepc.select import FAIL_CLOSED, TERMINATED, Budget, prior_gpu_hours

PY = sys.executable
LOG = C.RESULTS / "runner.jsonl"
TAG = "base_gen_3b"
LOAD_H = 0.1


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def event(**kw) -> None:
    kw = {"time": now().isoformat(timespec="seconds"), **kw}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def child_env() -> dict:
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1",
           **C.MLX_ENV}
    for k in ("CGT_P3B_DIR", "CGT_SHOTQA_ROOT", "CGT_GPU_LOCK", "HF_HOME", "HF_HUB_CACHE", "CGT_BASE_GEN_IMG"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def gpu_stage(name: str, args: list[str], budget: Budget, stop_at: dt.datetime, logf: Path) -> int:
    """One child under nice 19 holding the GPU lock, terminated at `stop_at`. Returns its exit code, TERMINATED, or -998 (lock never free)."""
    wait_for_memory(C.MEM_FLOOR, log=lambda m: event(stage=name, msg=m), poll_s=60)
    lock = GpuLock(f"base_gen_{name}")
    while True:
        try:
            lock.acquire()
            break
        except LockHeld as e:
            if now() + dt.timedelta(minutes=1) >= stop_at:
                event(stage=name, msg="GPU lock still held at the stage's stop time: not started", holder=e.holder.strip())
                return -998
            event(stage=name, msg="GPU lock held by another job, waiting 60 s", holder=e.holder.strip())
            time.sleep(60)
    t0 = time.monotonic()
    event(stage=name, msg="lock acquired, child started", stop_at=stop_at.isoformat(), used_h=round(budget.used_h, 3))
    rc = None
    try:
        logf.parent.mkdir(parents=True, exist_ok=True)
        with open(logf, "a") as out:
            p = subprocess.Popen(["nice", "-n", "19", PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=child_env())
            event(stage=name, msg="child pid", child_pid=p.pid)
            while p.poll() is None:
                if now() >= stop_at:
                    p.terminate()
                    try:
                        p.wait(180)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    rc = TERMINATED
                    break
                time.sleep(15)
            if rc is None:
                rc = p.returncode
    finally:
        hours = (time.monotonic() - t0) / 3600
        budget.add(hours)
        removed = lock.release()
        event(stage=name, rc=rc, stage_h=round(hours, 3), used_h=round(budget.used_h, 3), lock_released=removed)
    return rc


def speedcheck() -> None:
    global LOG
    out = C.RESULTS / "speedcheck"
    tag = "base_gen_speed"
    shutil.rmtree(C.DATA / "runs" / tag, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    LOG = out / "runner.jsonl"
    event(msg="speedcheck started", mlx_env=C.MLX_ENV)
    budget = Budget(1.0, now() + dt.timedelta(hours=1))
    rc_t = gpu_stage("speed_train", ["scripts/base_gen_train.py", "--tag", tag, "--n-samples", "160", "--stop-after-update", "10",
                                     "--target-pct", "25", "--results-dir", str(out / "train")], budget, budget.stop_at(now()),
                     out / "stdout.log")
    rc_e = gpu_stage("speed_val20", ["scripts/base_gen_eval.py", "--arm", "3b", "--adapter", "none", "--limit", "20",
                                     "--out", str(out / "val20_base_model")], budget, budget.stop_at(now()), out / "stdout.log")
    ups = [json.loads(x) for x in (out / "train/updates.jsonl").read_text().splitlines()] if (out / "train/updates.jsonl").exists() else []
    leak = [json.loads(x) for x in (out / "train/leak_probes.jsonl").read_text().splitlines()] if (out / "train/leak_probes.jsonl").exists() else []
    val = json.loads((out / "val20_base_model.summary.json").read_text()) if (out / "val20_base_model.summary.json").exists() else {}
    secs = [u["sec"] for u in ups[1:]]      # update 1 includes compilation
    summ = {"train_rc": rc_t, "val_rc": rc_e, "ok": rc_t == 9 and rc_e == 0 and len(ups) == 10,
            "updates": [{k: u[k] for k in ("update", "micro", "loss", "sec", "peak_gb", "gnorm", "lr")} for u in ups],
            "sec_per_update_mean_2_10": round(sum(secs) / len(secs), 2) if secs else None,
            "sec_per_update_max_2_10": max(secs) if secs else None, "peak_gb": max((u["peak_gb"] for u in ups), default=None),
            "leak_probe": [{k: v for k, v in x.items() if k != "time"} for x in leak],
            "val20": {k: val.get(k) for k in ("n", "load_seconds", "seconds", "sec_per_question", "peak_gb", "note")},
            "total_gpu_h": round(budget.used_h, 4), "model": f"{C.REPO_3B}@{C.REVISION_3B}"}
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    sys.exit(0 if summ["ok"] else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speedcheck", action="store_true")
    ap.add_argument("--target-pct", type=int, choices=(25, 50), default=C.TARGET_PCT_DEFAULT)
    ap.add_argument("--sec-per-update", type=float, default=None, help="measured s/update (plan amendment) for the projection")
    ap.add_argument("--deadline", default=C.TRAIN_DEADLINE)
    ap.add_argument("--cap-h", type=float, default=C.GPU_CAP_H)
    a = ap.parse_args()
    if a.speedcheck:
        speedcheck()
        return
    if a.sec_per_update is None:
        raise SystemExit("--sec-per-update (the measured value recorded in the plan amendment) is required for a real run")
    deadline = dt.datetime.fromisoformat(a.deadline)
    budget = Budget(a.cap_h, deadline)
    budget.add(prior_gpu_hours(LOG))
    run_dir = C.DATA / "runs" / TAG
    runs = C.RESULTS / "runs" / TAG
    stop_update = {v: k for k, v in C.MILESTONES.items()}[a.target_pct]
    resume_update = json.loads((run_dir / "resume.json").read_text())["update"] if (run_dir / "resume.json").exists() else 0
    proj_h = (stop_update - resume_update) * a.sec_per_update / 3600 + LOAD_H
    event(msg="runner started", deadline=deadline.isoformat(), cap_h=a.cap_h, pid=os.getpid(), prior_gpu_h=round(budget.used_h, 3),
          target_pct=a.target_pct, stop_update=stop_update, resume_update=resume_update, sec_per_update=a.sec_per_update,
          projection_h=round(proj_h, 3), expected_end=(now() + dt.timedelta(hours=proj_h)).isoformat(timespec="minutes"),
          mlx_env=C.MLX_ENV, max_train_resumes=C.MAX_TRAIN_RESUMES, resume_every=C.RESUME_EVERY,
          head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip())
    if (run_dir / "DONE").exists():
        event(msg="training already DONE, not re-run")
        return
    if not budget.fits(proj_h):
        event(msg="training DROPPED: the projection to the target checkpoint does not fit the cap or the deadline",
              projection_h=round(proj_h, 3), used_h=round(budget.used_h, 3))
        return
    args = ["scripts/base_gen_train.py", "--tag", TAG, "--target-pct", str(a.target_pct)]
    rc = gpu_stage("train", args, budget, budget.stop_at(now()), runs / "stdout.log")
    used = 0
    while rc not in (0, TERMINATED, -998, *FAIL_CLOSED) and used < C.MAX_TRAIN_RESUMES and now() < budget.stop_at(now()):
        used += 1
        event(msg=f"training crashed: crash resume {used} of {C.MAX_TRAIN_RESUMES} from the resume state", rc=rc)
        rc = gpu_stage("train_resume", args, budget, budget.stop_at(now()), runs / "stdout.log")
    manifest_f = runs / "manifest.json"
    have = sorted(json.loads(manifest_f.read_text()).get("checkpoints", {})) if manifest_f.exists() else []
    if rc == 0:
        event(msg=f"training reached the {a.target_pct}% target", checkpoints=have)
    elif rc == TERMINATED:
        event(msg="training TERMINATED at the deadline/cap", checkpoints=have)
    else:
        event(msg="training did not finish", rc=rc, checkpoints=have)
    event(msg="GPU phase finished", used_h=round(budget.used_h, 3))


if __name__ == "__main__":
    main()
