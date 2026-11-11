"""Unattended supervisor for the Phase 4 video-LoRA cross-fitting (docs/phase4_video_plan.md Amendment 2 item 7, Amendments 6 and 7: 15 restarts in total for Metal-watchdog failures only, back-off 2/4/8/10 minutes, process snapshot at each crash): takes `/tmp/cgt_gpu.lock` (job name, released in a finally/signal handler,
another owner's lock is never removed), runs the stages fold0, fold1, fold2, final one after the other with scripts/p4_video_train.py, resumes a stage from its latest resume state after a crash
(e.g. the Metal watchdog); restarts are limited to 15 IN TOTAL for the whole run for Metal-watchdog failures and 5 for any other failure (Amendments 6-8), the count is derived from the file-backed restarts.jsonl and survives supervisor relaunches; it NEVER restarts after a guard failure or the stopping rule (exit codes 3-10 and 13). Writes
results/phase4/runs/video_<tag>/RUN_PROGRESS.md every 30 minutes. Stops after the last stage (the validation, controls, combination weights and the freeze are separate steps).
Usage: nohup caffeinate -i uv run python scripts/p4_video_supervise.py --tag v1 --frames 8 --tokens 64 &"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cgt.p3b_vlm.runstatus import append_jsonl, now_iso
from cgt.p4_camera_move import gpulock

ROOT = Path(os.environ.get("P4_SUP_ROOT", Path(__file__).resolve().parents[1]))  # the environment override exists for the CPU test of this supervisor with a stub trainer
NO_RESTART = {3, 4, 5, 6, 7, 8, 9, 10, 13}  # 3-10 per the plan (9 is the trainer's testing stop code and is NOT restarted either, Amendment 8); 13: a resume did not reproduce the earlier loss (Amendment 7)
MAX_RESTARTS = 5  # limit for failures that are NOT the Metal watchdog (exhausted)
MAX_WATCHDOG_RESTARTS_TOTAL = 15  # Amendment 7: total for the whole run, only for failures whose stdout shows the watchdog error
WATCHDOG = "kIOGPUCommandBufferCallbackErrorImpactingInteractivity"
BACKOFF_S = [120, 240, 480, 600]  # 2, 4, 8 minutes, then 10 minutes for every further restart
MAX_CONSECUTIVE_NO_PROGRESS = 4
JOB = "p4-video-train"


def last_logged_update(logd: Path) -> int:
    """Highest update number written to the stage's updates.jsonl (finer progress than the resume file, which is saved every 25 updates)."""
    f = logd / "updates.jsonl"
    if not f.exists() or not f.read_text().strip():
        return 0
    return int(json.loads(f.read_text().splitlines()[-1])["update"])


def resume_state(run_dir: Path) -> dict:
    f = run_dir / "resume.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_overall(top: Path, tag: str, stages: list[str], state: str, restarts: int, started: float, note: str = "") -> None:
    lines = [f"# STATUS video {tag} (supervisor)", "", f"- written: {now_iso()}", f"- state: **{state}**", f"- restarts so far (limits: {MAX_WATCHDOG_RESTARTS_TOTAL} total for Metal-watchdog failures, {MAX_RESTARTS} for any other): {restarts}",
             f"- wall time so far: {(time.monotonic() - started) / 3600:.2f} h", ""]
    for s in stages:
        run_dir = ROOT / "data/cache/p4/video_runs" / tag / s
        st = top / s / "RUN_PROGRESS.md"
        done = (run_dir / "DONE").exists()
        rs = resume_state(run_dir)
        lines.append(f"## {s}: {'DONE' if done else 'pending / running'}" + (f" (resume state: epoch {rs.get('epoch')}, update {rs.get('update')})" if rs and not done else ""))
        if st.exists():
            lines += ["", st.read_text().strip(), ""]
        oofs = sorted((top / s).glob("oof_e*.json"))
        for o in oofs:
            d = json.loads(o.read_text())
            lines.append(f"- OOF epoch {d['epoch']}: option accuracy {d['acc']:.4f} (n={d['n']})")
        lines.append("")
    if note:
        lines += ["", note]
    (top / "RUN_PROGRESS.md").write_text("\n".join(lines) + "\n")


def heaviest_processes() -> dict:
    """The heaviest processes by CPU and by memory at a crash (names only; recorded in restarts.jsonl)."""
    def top(sort_flag: str, n: int) -> list[str]:
        r = subprocess.run(["ps", "-axo", "pid,pcpu,pmem,rss,comm", sort_flag], capture_output=True, text=True, check=False).stdout.splitlines()[1: n + 1]
        return [" ".join(x.split()[:4]) + " " + Path(" ".join(x.split()[4:])).name for x in r]
    return {"by_cpu": top("-r", 8), "by_mem": top("-m", 8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--stages", default="fold0,fold1,fold2,final")
    ap.add_argument("--resume-every", type=int, default=25)
    ap.add_argument("--lock-wait-hours", type=float, default=3.0)
    ap.add_argument("--restarts-used", type=int, default=None, help="optional lower bound; the restarts already used are derived from restarts.jsonl and the larger value counts (an understated value is ignored and recorded, a negative one is refused)")
    ap.add_argument("--settle-seconds", type=int, default=0, help="wait this long before the first launch (let a thrashing machine settle)")
    ap.add_argument("--extra", default="", help="extra arguments for p4_video_train.py (pipeline check only)")
    a = ap.parse_args()
    stages = a.stages.split(",")
    top = ROOT / "results/phase4/runs" / f"video_{a.tag}"
    top.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hist_recs = [json.loads(x) for x in (top / "restarts.jsonl").read_text().splitlines() if x.strip()] if (top / "restarts.jsonl").exists() else []
    history = max([len(hist_recs), *[int(r.get("restart", 0)) for r in hist_recs]])  # file-backed: the highest restart number on record
    if a.restarts_used is not None and a.restarts_used < 0:
        print(f"refused: negative --restarts-used {a.restarts_used}", file=sys.stderr)
        sys.exit(2)
    understated = a.restarts_used is not None and a.restarts_used < history
    restarts = max(history, a.restarts_used or 0)  # the cumulative restart history is file-backed: a relaunch cannot reset it
    local_restarts: list[bool] = []
    n_backoff = 0
    child: subprocess.Popen | None = None

    def cleanup(*_):
        if child is not None and child.poll() is None:
            child.terminate()
        gpulock.release(JOB)
        sys.exit(143)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    t_lock = time.monotonic()
    while not gpulock.acquire(JOB, a.tag):
        write_overall(top, a.tag, stages, f"waiting for the GPU lock held by: {gpulock.owner()}", restarts, started)
        if time.monotonic() - t_lock > a.lock_wait_hours * 3600:
            write_overall(top, a.tag, stages, "STOPPED: the GPU lock stayed with another owner", restarts, started)
            sys.exit(11)
        time.sleep(60)
    append_jsonl(top / "supervisor.jsonl", {"event": "supervisor start (lock taken)", "stages": stages, "frames": a.frames, "tokens": a.tokens, "restarts_already_used": restarts, "restarts_used_flag": a.restarts_used, "flag_understated_and_ignored": understated,
                                            "settle_seconds": a.settle_seconds})
    time.sleep(a.settle_seconds)
    code_final = 0
    try:
        for stage in stages:
            run_dir = ROOT / "data/cache/p4/video_runs" / a.tag / stage
            logd = top / stage
            logd.mkdir(parents=True, exist_ok=True)
            cmd = [*([sys.executable] if os.environ.get("P4_SUP_NO_CAFFEINATE") else ["caffeinate", "-i", "uv", "run", "python"]), "scripts/p4_video_train.py", "--stage", stage, "--tag", a.tag, "--frames", str(a.frames), "--tokens", str(a.tokens),
                   "--resume-every", str(a.resume_every), *a.extra.split()]
            while not (run_dir / "DONE").exists():
                before = {"resume": resume_state(run_dir), "last_update": last_logged_update(logd)}
                stdout_off = (logd / "train.stdout").stat().st_size if (logd / "train.stdout").exists() else 0
                t_launch = time.monotonic()
                with open(logd / "train.stdout", "a") as out:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                    t_status = 0.0
                    while child.poll() is None:
                        if time.monotonic() - t_status >= 1800:
                            t_status = time.monotonic()
                            write_overall(top, a.tag, stages, f"running stage {stage}", restarts, started)
                        time.sleep(float(os.environ.get("P4_SUP_POLL_S", "30")))
                code = child.returncode
                child = None
                if code == 0:
                    append_jsonl(top / "supervisor.jsonl", {"event": f"stage {stage} finished", "restarts": restarts, "wall_s": round(time.monotonic() - started)})
                    break
                tail = "\n".join((logd / "train.log").read_text().splitlines()[-4:]) if (logd / "train.log").exists() else ""
                if code in NO_RESTART:
                    append_jsonl(top / "supervisor.jsonl", {"event": "STOPPED (no restart)", "stage": stage, "exit_code": code, "log_tail": tail})
                    write_overall(top, a.tag, stages, f"STOPPED by a guard or the stopping rule in stage {stage} (exit {code}); NOT restarted", restarts, started, f"Last log lines:\n\n```\n{tail}\n```")
                    code_final = code
                    sys.exit(code_final)
                restarts += 1
                after = {"resume": resume_state(run_dir), "last_update": last_logged_update(logd)}
                with open(logd / "train.stdout", "rb") as fh:
                    fh.seek(stdout_off)
                    watchdog = WATCHDOG in fh.read().decode(errors="replace")
                append_jsonl(top / "restarts.jsonl", {"restart": restarts, "stage": stage, "exit_code": code, "watchdog": watchdog, "ran_s": round(time.monotonic() - t_launch),
                                                      "resume_before": before, "resume_after": after, "heaviest_processes": heaviest_processes(), "log_tail": tail})
                limit = MAX_WATCHDOG_RESTARTS_TOTAL if watchdog else MAX_RESTARTS
                if restarts > limit:
                    write_overall(top, a.tag, stages, f"STOPPED: restart limit ({limit}) exceeded; this failure was {'a watchdog kill' if watchdog else 'NOT a watchdog kill'}", restarts, started,
                                  f"Stage {stage}, last exit code {code}. Log tail:\n\n```\n{tail}\n```")
                    code_final = 11
                    sys.exit(code_final)
                progressed = after["last_update"] > before["last_update"]
                local_restarts.append(not progressed)  # progress is judged on THIS invocation's restarts only (the update log, not the 25-update resume file)
                if len(local_restarts) >= MAX_CONSECUTIVE_NO_PROGRESS and all(local_restarts[-MAX_CONSECUTIVE_NO_PROGRESS:]):
                    write_overall(top, a.tag, stages, f"STOPPED: no progress across {MAX_CONSECUTIVE_NO_PROGRESS} consecutive restarts (stage {stage})", restarts, started, f"Log tail:\n\n```\n{tail}\n```")
                    code_final = 12
                    sys.exit(code_final)
                if time.monotonic() - t_launch >= 1800 and progressed:
                    n_backoff = 0  # a long productive run resets the back-off ladder
                delay = BACKOFF_S[min(n_backoff, len(BACKOFF_S) - 1)]
                n_backoff += 1
                append_jsonl(top / "supervisor.jsonl", {"event": "back-off before restart", "restart": restarts, "watchdog": watchdog, "seconds": delay})
                write_overall(top, a.tag, stages, f"back-off {delay}s before restart {restarts} (stage {stage})", restarts, started)
                time.sleep(delay * float(os.environ.get("P4_SUP_BACKOFF_SCALE", "1")))
        write_overall(top, a.tag, stages, "ALL STAGES FINISHED (training complete; validation, controls, combination weights and freeze are separate steps)", restarts, started)
    finally:
        gpulock.release(JOB)
        append_jsonl(top / "supervisor.jsonl", {"event": "supervisor exit (lock released if ours)", "code": code_final})
    sys.exit(code_final)


if __name__ == "__main__":
    main()
