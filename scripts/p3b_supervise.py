"""Overnight supervisor for the Phase 3B run: keeps the machine awake (caffeinate), runs scripts/p3b_train.py, and resumes it from the latest
resume state if the process dies (e.g. the Metal GPU watchdog), at most 5 restarts. NEVER restarts after a guard failure or the stopping rule
(exit codes 3-10): it stops and writes the reason to RUN_PROGRESS.md. After a successful run it runs scripts/p3b_validate_freeze.py (inference only).
Usage: nohup uv run python scripts/p3b_supervise.py --tag <tag> [--n-samples N] [--resume-every 200] [--no-post] &"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from cgt.p3b_vlm.runstatus import append_jsonl, write_status

ROOT = Path(__file__).resolve().parents[1]
NO_RESTART = {3, 4, 5, 6, 7, 8, 10}
MAX_RESTARTS = 5


def last_update(run_dir: Path) -> int:
    f = run_dir / "resume.json"
    return json.loads(f.read_text())["update"] if f.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--resume-every", type=int, default=200)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-post", action="store_true", help="skip the post-training validation/freeze step")
    ap.add_argument("--post-args", default="", help="extra arguments for p3b_validate_freeze.py (pipeline check: '--val-n 100 --out-dir ...')")
    a = ap.parse_args()
    run_dir = ROOT / "data/cache/p3b/runs" / a.tag
    logd = ROOT / "results/phase3b/runs" / a.tag
    logd.mkdir(parents=True, exist_ok=True)
    cmd = ["caffeinate", "-i", "uv", "run", "python", "scripts/p3b_train.py", "--tag", a.tag, "--resume-every", str(a.resume_every)]
    if a.n_samples:
        cmd += ["--n-samples", str(a.n_samples)]
    if a.allow_dirty:
        cmd += ["--allow-dirty"]
    restarts, started = 0, time.monotonic()
    append_jsonl(logd / "supervisor.jsonl", {"event": "supervisor start", "cmd": cmd})
    while True:
        before = last_update(run_dir)
        with open(logd / "train.stdout", "a") as out:
            code = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=False).returncode
        if code == 0:
            append_jsonl(logd / "supervisor.jsonl", {"event": "training finished", "restarts": restarts, "wall_s": round(time.monotonic() - started)})
            break
        tail = "\n".join((logd / "train.log").read_text().splitlines()[-4:]) if (logd / "train.log").exists() else ""
        if code in NO_RESTART:
            append_jsonl(logd / "supervisor.jsonl", {"event": "STOPPED (no restart)", "exit_code": code, "log_tail": tail})
            write_status(logd, a.tag, state=f"STOPPED by a guard or the stopping rule (exit {code}); NOT restarted",
                         note=f"Supervisor did not restart (exit codes {sorted(NO_RESTART)} are final). Last log lines:\n\n```\n{tail}\n```")
            sys.exit(code)
        after = last_update(run_dir)
        restarts += 1
        append_jsonl(logd / "restarts.jsonl", {"restart": restarts, "exit_code": code, "update_before": before, "update_after": after,
                                               "reason": "process died (not a guard failure or divergence)", "log_tail": tail})
        if restarts > MAX_RESTARTS:
            append_jsonl(logd / "supervisor.jsonl", {"event": "STOPPED: more than 5 restarts"})
            write_status(logd, a.tag, state="STOPPED: restart limit (5) exceeded", note=f"Last exit code {code}. Log tail:\n\n```\n{tail}\n```")
            sys.exit(11)
        if after <= before and restarts >= 2 and (logd / "restarts.jsonl").exists():
            recent = [json.loads(x) for x in (logd / "restarts.jsonl").read_text().splitlines()][-2:]
            if all(r["update_after"] == r["update_before"] for r in recent):
                write_status(logd, a.tag, state="STOPPED: no progress across two restarts", note=f"Log tail:\n\n```\n{tail}\n```")
                sys.exit(12)
        time.sleep(20)
    if not a.no_post:
        r = subprocess.run(["caffeinate", "-i", "uv", "run", "python", "scripts/p3b_validate_freeze.py", "--tag", a.tag, *a.post_args.split()], cwd=ROOT, check=False)
        append_jsonl(logd / "supervisor.jsonl", {"event": "validate_freeze finished", "exit_code": r.returncode})
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
