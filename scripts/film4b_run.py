"""Runner of the opt-in film mode 4B + C5 (docs/plans/film_4b_c5.md section 3). Pattern: scripts/stepa_4b_run.py.
Holds /tmp/cgt_gpu.lock (cgt.pipeline.gpulock; polls every 60 s if another job holds it, never removes it) ONLY while a GPU child runs; the child
(`scripts/p3c_run_film.py --method D --model 4b --c5 ...`) checks the runner's nonce in the lock. Children: nice 19, MLX buffers 10/10,
HF_HUB_OFFLINE=1. Memory >= 25% free before each child. Budget: --cap-h GPU hours (lock-held time, summed across launches from the runner log)
and --deadline: the child gets the earlier of the two as its stop time and starts no shot that would not finish before it (per-shot projection
from the last measured shot); a child still running at the stop time is terminated. Crash resume from the cache, at most 10 in total.
SIGTERM (`pkill -TERM -f scripts/film4b_run.py`) terminates the child and releases the lock. Then the CPU report.
  Pipeline check (3 shots, two children: --model 4b, then --model 4b --c5 on the same cache):
    HF_HUB_OFFLINE=1 caffeinate -i nice -n 19 <venv python> scripts/film4b_run.py --pipecheck --data-root <PM worktree> --keyframe-root <main>
  Sample (200 shots):
    HF_HUB_OFFLINE=1 nohup caffeinate -i nice -n 19 <venv python> scripts/film4b_run.py --data-root <PM worktree> --keyframe-root <main> \
        > results/film4b/runner_stdout.log 2>&1 &"""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.cinescale2.semantic import wait_for_memory
from cgt.film4b import mode as M
from cgt.pipeline.gpulock import GpuLock, LockHeld
from cgt.stepc.select import Budget, prior_gpu_hours

PY = sys.executable
RESULTS = ROOT / "results/film4b"
LOG = RESULTS / "runner.jsonl"
FILM = "night_of_the_living_dead"
SEED, N = 20261017, 200
LOAD_H = 0.03                 # model load per child (replaced by nothing: conservative constant)
SEC_PER_SHOT_PLAN = 15.0      # before the pipeline check: conservative guess (Track F 2B default 2.3 s/shot); the check replaces it
FAIL_CLOSED = tuple(range(3, 21))
TERMINATED, LOCK_TIMEOUT = -999, -998
CRASH_MSG = "child crashed: crash resume "


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def event(**kw) -> None:
    kw = {"time": now().isoformat(timespec="seconds"), **kw}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def child_env(nonce: str) -> dict:
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1",
           **M.MLX_ENV, M.LOCK_NONCE_ENV: nonce}
    for k in ("CGT_GPU_LOCK", "CGT_WORKTREES", "CGT_STEPA4B_ADAPTER", "CGT_P3A_HEADS_DIR", "HF_HOME", "HF_HUB_CACHE"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def prior_crash_resumes(log: Path) -> int:
    if not log.exists():
        return 0
    return sum(str(json.loads(x).get("msg", "")).startswith(CRASH_MSG) for x in log.read_text().splitlines() if x.strip())


def rerun_after(rc: int, resumes_used: int, max_resumes: int = M.MAX_CRASH_RESUMES) -> bool:
    return rc not in (0, TERMINATED, LOCK_TIMEOUT, *FAIL_CLOSED) and resumes_used < max_resumes


def shots_todo(cache: Path, shot_ids: list[int]) -> int:
    done = M.done_keys(cache / "logits.jsonl")
    return sum(not all((s, kf) in done for kf in M.KEYFRAMES) for s in shot_ids)


def measured_rate() -> float:
    f = RESULTS / "pipecheck/summary.json"
    if f.exists():
        r = json.loads(f.read_text()).get("sec_per_shot_projection")
        if r:
            return float(r)
    return SEC_PER_SHOT_PLAN


def _stop_child(p) -> None:
    p.terminate()
    try:
        p.wait(120)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def _on_sigterm(signum, frame):
    raise SystemExit(143)


def gpu_stage(name: str, args: list[str], budget: Budget, stop_at: dt.datetime, logf: Path, lock_path: Path | None = None,
              popen=subprocess.Popen, poll_s: float = 10) -> int:
    wait_for_memory(25, log=lambda m: event(stage=name, msg=m), poll_s=60)
    lock = GpuLock(f"film4b_{name}", lock_path)
    while True:
        try:
            lock.acquire()
            break
        except LockHeld as e:
            if now() + dt.timedelta(minutes=1) >= stop_at:
                event(stage=name, msg="GPU lock still held at the stop time: not started", holder=e.holder.strip())
                return LOCK_TIMEOUT
            event(stage=name, msg="GPU lock held by another job, waiting 60 s", holder=e.holder.strip())
            time.sleep(60)
    t0 = time.monotonic()
    event(stage=name, msg="lock acquired, child started", stop_at=stop_at.isoformat(), used_h=round(budget.used_h, 4))
    rc = None
    try:
        logf.parent.mkdir(parents=True, exist_ok=True)
        with open(logf, "a") as out:
            p = popen(["nice", "-n", "19", PY, "scripts/p3c_run_film.py", *args, "--stop-at", stop_at.isoformat()], cwd=ROOT, stdout=out,
                      stderr=subprocess.STDOUT, env=child_env(lock.nonce))
            try:
                while p.poll() is None:
                    if now() >= stop_at + dt.timedelta(seconds=60):  # the child stops itself at stop_at; this is the backstop
                        _stop_child(p)
                        rc = TERMINATED
                        break
                    time.sleep(poll_s)
            except BaseException:
                _stop_child(p)
                event(stage=name, msg="runner stopped: child terminated, lock released")
                raise
            if rc is None:
                rc = p.returncode
    finally:
        hours = (time.monotonic() - t0) / 3600
        budget.add(hours)
        removed = lock.release()
        event(stage=name, rc=rc, stage_h=round(hours, 4), used_h=round(budget.used_h, 4), lock_released=removed)
    return rc


def child_args(a, c5: bool, first: int | None, rate: float) -> list[str]:
    out = ["--film", FILM, "--method", "D", "--model", "4b", "--sample", str(N), "--seed", str(SEED), "--data-root", a.data_root,
           "--keyframe-root", a.keyframe_root, "--est-shot-s", f"{rate:.2f}", "--budget-hours", str(M.GPU_CAP_H)]
    return out + (["--c5"] if c5 else []) + (["--first", str(first)] if first else [])


def run_children(a, budget: Budget, c5: bool, first: int | None, cache: Path, shot_ids: list[int], logf: Path, name: str) -> int:
    """Start children until every shot is cached (and, with c5, the SigLIP scores exist), the budget stops them, or a non-resumable exit.
    A child exiting 0 with shots left stopped at its stop time: nothing more is started."""
    rate = measured_rate()
    resumes = prior_crash_resumes(LOG)
    first_try = True
    while True:
        todo = shots_todo(cache, shot_ids)
        if todo == 0 and not (c5 and not (cache / "siglip3a_scores.json").exists()):
            event(msg=f"{name}: all {len(shot_ids)} shots cached")
            return 0
        proj = todo * rate / 3600 + LOAD_H
        if not budget.fits(LOAD_H + rate / 3600):
            event(msg=f"{name}: not even one shot fits the cap or the deadline: DROPPED", todo=todo, used_h=round(budget.used_h, 4),
                  cap_h=budget.cap_h)
            return 0
        if not budget.fits(proj):
            event(msg=f"{name}: projection {proj:.3f} h does not fit; the child starts no shot that would pass its stop time", todo=todo)
        if not first_try:
            resumes += 1
            event(msg=f"{CRASH_MSG}{resumes} of {M.MAX_CRASH_RESUMES}", todo=todo)
        rc = gpu_stage(name, child_args(a, c5, first, rate), budget, budget.stop_at(now()), logf)
        first_try = False
        if rc == 0:
            left = shots_todo(cache, shot_ids)
            if left:
                event(msg=f"{name}: child stopped at its stop time with {left} shots left (partial)")
            return 0
        if not rerun_after(rc, resumes):
            event(msg=f"{name}: not resumed", rc=rc, crash_resumes=resumes)
            return rc


def sample_ids(a, first: int | None) -> list[int]:
    from cgt.p3s_stepa import film_run as R
    s = R.load_sample(Path(a.data_root), FILM, N, SEED, Path(a.keyframe_root))
    ids = [int(x) for x in s.shot_id]
    return ids[:first] if first else ids


def main() -> None:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipecheck", action="store_true")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--keyframe-root", required=True)
    ap.add_argument("--deadline", default=M.HARD_STOP)
    ap.add_argument("--cap-h", type=float, default=M.GPU_CAP_H)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)
    deadline = dt.datetime.fromisoformat(a.deadline)
    first = 3 if a.pipecheck else None
    if a.pipecheck:
        LOG = RESULTS / "pipecheck/runner.jsonl"
    budget = Budget(a.cap_h, deadline)
    budget.add(prior_gpu_hours(LOG))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    event(msg="runner started", pipecheck=a.pipecheck, deadline=deadline.isoformat(), cap_h=a.cap_h, pid=os.getpid(),
          prior_gpu_h=round(budget.used_h, 4), prior_crash_resumes=prior_crash_resumes(LOG), mlx_env=M.MLX_ENV, head=head)
    ids = sample_ids(a, first)
    cache = M.out_dir(FILM, N, SEED, first)
    logf = (RESULTS / "pipecheck" if a.pipecheck else RESULTS) / "child_stdout.log"
    if a.pipecheck:
        rc1 = run_children(a, budget, False, first, cache, ids, logf, "pipecheck_4b")
        rc2 = run_children(a, budget, True, first, cache, ids, logf, "pipecheck_4b_c5")
        event(msg="pipecheck GPU phase finished", rc_4b=rc1, rc_4b_c5=rc2, used_h=round(budget.used_h, 4))
        report_args = ["--cache", str(cache), "--out", str(RESULTS / "pipecheck/notld_first3"), "--pipecheck-summary",
                       str(RESULTS / "pipecheck/summary.json"), "--projection-shots", str(N)]
    else:
        rc = run_children(a, budget, True, None, cache, ids, logf, "sample_4b_c5")
        event(msg="GPU phase finished", rc=rc, used_h=round(budget.used_h, 4))
        report_args = ["--cache", str(cache), "--out", str(RESULTS / "notld_sample")]
    r = subprocess.run([PY, "scripts/film4b_report.py", "--data-root", a.data_root, *report_args], cwd=ROOT,
                       env={**child_env(""), M.LOCK_NONCE_ENV: ""}, capture_output=True, text=True, check=False)
    event(step="report", rc=r.returncode, tail=(r.stdout + r.stderr)[-1500:])


if __name__ == "__main__":
    main()
