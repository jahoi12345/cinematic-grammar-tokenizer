"""Token-table rebuild for the p3c-v2-4b default. Subcommands, in order:
  preflight  CPU: inputs, 4B adapter sha, BEFORE hashes of the live tables, live token meta == current inputs   -> data/cache/rebuild_4b/<film>/preflight.json
  run        GPU runner: children `scripts/p3c_run_film.py --model 4b --out <stage>` under /tmp/cgt_gpu.lock (nonce), nice 19, MLX 10/10,
             HF_HUB_OFFLINE=1, memory >= 25% free, cap (--cap-h GPU-h of lock-held time) and hard stop (--deadline); a film whose
             projection does not fit is DROPPED (not started, not shortened); backstop terminate at stop + 60 s; crash resume <= 10;
             SIGTERM terminates the child and releases the lock. Log: data/cache/rebuild_4b/runner.jsonl (gitignored: names local films)
  build      CPU: stage root (byte copies of the unchanged inputs + the 4B run) -> unchanged token builder -> staged tables
  verify     CPU: coverage, one 4B run key, complete p3c-v2-4b manifest, columns and vocabulary unchanged, non-Phase-3C columns identical,
             meta inputs equal except p3c/scores  -> verify.json
  swap       CPU: live -> *_pre_4b backups (sha checked), staged -> live (sha checked) -> swap.json. Never deletes.
Films: --films notld,owned,clip (selectors resolved from the registry; owned/clip are local only and never named in committed files).
Usage (from the checkout holding the film's data, merged code, PYTHONPATH=src):
  HF_HUB_OFFLINE=1 nohup caffeinate -i nice -n 19 <python> scripts/rebuild_default_4b.py run --films notld,owned,clip \\
      --deadline 2026-10-22T10:30:00+02:00 --cap-h 3.0 > data/cache/rebuild_4b/runner_stdout.log 2>&1 &"""
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

from cgt.p3b_vlm import guards
from cgt.p3c_tokens import models, rebuild
from cgt.pipeline.gpulock import GpuLock, LockHeld

PY = sys.executable
LOG = ROOT / rebuild.WORK / "runner.jsonl"
LOCK_NONCE_ENV = "CGT_FILM4B_LOCK_NONCE"  # = cgt.film4b.mode.LOCK_NONCE_ENV (checked by the child)
MIN_FREE = 25
_STATE: dict = {"child": None, "lock": None, "term": False}


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def event(**kw) -> None:
    kw = {"time": now().isoformat(timespec="seconds"), **kw}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def prior_gpu_seconds(log: Path = LOG) -> float:
    if not log.exists():
        return 0.0
    return sum(r.get("lock_seconds", 0.0) for r in map(json.loads, log.read_text().splitlines()) if r.get("step") == "child_end")


def child_env(nonce: str) -> dict:
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1",
           **models.model_identity("4b")["mlx_env"], LOCK_NONCE_ENV: nonce}
    for k in ("CGT_GPU_LOCK", "CGT_WORKTREES", "CGT_STEPA4B_ADAPTER", "HF_HOME", "HF_HUB_CACHE"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def child_cmd(f: dict, stop_at: dt.datetime, est_shot_s: float) -> list[str]:
    cmd = ["nice", "-n", "19", PY, str(ROOT / "scripts/p3c_run_film.py"), "--film", f["film"], "--method", "D", "--model", "4b",
           "--out", str(rebuild.staged_paths(f["film"], ROOT)["p3c"]), "--stop-at", stop_at.isoformat(), "--est-shot-s", f"{est_shot_s:.3f}"]
    return cmd + (["--allow-owned"] if f["allow_owned"] else [])


def items_done(film: str) -> tuple[int, int]:
    """(keyframe items scored in this code's 4B run dir, items needed)."""
    import pandas as pd

    from cgt.p3c_tokens import runner
    n = len(pd.read_parquet(ROOT / f"data/processed/{film}_shots.parquet", columns=["shot_id"]))
    rd = ROOT / "data/cache/p3c/runs" / f"{film}_{rebuild.run_key_4b(film, runner.code_sha())}"
    done = sum(len(runner.repair_cache(rd / f"scores_{kf}.jsonl")) for kf in runner.KEYFRAMES) if rd.exists() else 0
    return done, 3 * n


def free_percent() -> int:
    from cgt.p3s_stepa.film_run import free_percent as fp
    return fp()


def take_lock(wait_min: float):
    t_end = time.time() + 60 * wait_min
    while True:
        try:
            return GpuLock("rebuild_default_4b").acquire()
        except LockHeld as e:
            if time.time() >= t_end or _STATE["term"]:
                raise
            print(f"GPU lock held, waiting: {e}", flush=True)
            time.sleep(60)


def run_child(cmd: list[str], env: dict, stop_at: dt.datetime, grace_s: float = 60.0, poll_s: float = 10.0) -> int:
    p = subprocess.Popen(cmd, cwd=ROOT, env=env)
    _STATE["child"] = p
    try:
        while True:
            try:
                return p.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                if _STATE["term"] or now() > stop_at + dt.timedelta(seconds=grace_s):
                    p.terminate()
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    return -999
    finally:
        _STATE["child"] = None


def on_term(signum, frame) -> None:
    _STATE["term"] = True
    if _STATE["child"] is not None:
        _STATE["child"].terminate()


def cmd_run(a, films: list[dict]) -> int:
    signal.signal(signal.SIGTERM, on_term)
    deadline = dt.datetime.fromisoformat(a.deadline).astimezone()
    used = prior_gpu_seconds()
    s_item = rebuild.SEC_PER_SHOT_W27 / 3
    crashes = 0
    event(step="start", films=[rebuild.label(f) for f in films], deadline=deadline, cap_h=a.cap_h, prior_gpu_h=used / 3600)
    for f in films:
        lab = rebuild.label(f)
        while not _STATE["term"]:
            done, need = items_done(f["film"])
            cap_end = now() + dt.timedelta(seconds=max(0.0, a.cap_h * 3600 - used))
            pl = rebuild.plan_film(now(), deadline, cap_end, need - done, s_item)
            if done >= need and rebuild.staged_paths(f["film"], ROOT)["p3c"].exists():
                event(step="film_done", film=lab)
                break
            if not pl["fits"]:
                event(step="drop", film=lab, remaining_items=need - done, need_h=pl["need_s"] / 3600, stop_at=pl["stop_at"])
                break
            if free_percent() < MIN_FREE:
                event(step="drop", film=lab, why=f"memory below {MIN_FREE}% free")
                break
            try:
                lock = take_lock(a.wait_lock_min)
            except LockHeld as e:  # after the wait
                event(step="stop", why=f"GPU lock not obtained: {e}")
                return 3
            _STATE["lock"] = lock
            t0 = time.time()
            try:
                rc = run_child(child_cmd(f, pl["stop_at"], 3 * s_item), child_env(lock.nonce), pl["stop_at"])
            finally:
                lock.release()
                _STATE["lock"] = None
            secs = time.time() - t0
            used += secs
            done2, _ = items_done(f["film"])
            if done2 > done:
                s_item = max(s_item, (secs - rebuild.LOAD_S) / (done2 - done)) if secs > rebuild.LOAD_S else s_item
            kind = rebuild.classify_exit(rc) if rc != -999 else "terminated"
            event(step="child_end", film=lab, rc=rc, kind=kind, lock_seconds=secs, items=done2, of=need, s_per_item=s_item,
                  gpu_h_used=used / 3600)
            if kind == "complete":
                break
            if kind in ("stopped", "terminated", "refused"):
                event(step="drop", film=lab, why=kind)
                break
            crashes += 1
            if crashes > rebuild.MAX_CRASH_RESUMES:
                event(step="stop", why="crash resume cap reached")
                return 6
            event(step="crash_resume", film=lab, n=crashes)
    event(step="end", gpu_h_used=used / 3600, terminated=_STATE["term"])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["preflight", "run", "build", "verify", "swap"])
    ap.add_argument("--films", default="notld,owned,clip")
    ap.add_argument("--deadline", default=rebuild.DEADLINE)
    ap.add_argument("--cap-h", type=float, default=3.0)
    ap.add_argument("--wait-lock-min", type=float, default=120.0)
    ap.add_argument("--root", default=None, help="preflight only: read-only data checkout to check (writes nothing with --no-write)")
    ap.add_argument("--no-write", action="store_true", help="preflight only: print, write no file")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else ROOT
    if a.root and a.cmd != "preflight":
        raise SystemExit("--root is accepted only by preflight; run the other steps from the checkout holding the data")
    try:
        films = rebuild.resolve_films([s.strip() for s in a.films.split(",") if s.strip()], root)
        if a.cmd == "run":
            return cmd_run(a, films)
        rc = 0
        for f in films:
            if a.cmd == "preflight":
                r = rebuild.preflight(f, root, write=not a.no_write)
            elif a.cmd == "build":
                r = rebuild.build_stage(f, root)
            elif a.cmd == "verify":
                r = rebuild.verify(f, root)
            else:
                r = rebuild.swap(f, root)
            ok = r.get("ok", True)
            rc = rc or (0 if ok else 1)
            print(json.dumps({"film": rebuild.label(f), "cmd": a.cmd, "ok": ok, "problems": r.get("problems", []),
                              **({"changes": r.get("primary_changes_vs_live")} if a.cmd == "verify" else {})}, default=str), flush=True)
        return rc
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}): {e}", flush=True)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
