"""SEALED matched ShotBench evaluation (docs/p3m_matched_protocol.md section 3): one evaluation per model tag, ever. NEVER runnable without the
author's GO file results/p3m_matched/GO_<tag> (written by scripts/p3m_matched_write_go.py after the user's go). Every refusal runs before any
ShotBench path is touched. There is no dry-run mode. Exit codes as scripts/p3b_eval_shotbench.py (7 identity, 10 dirty code, 13 no/unbound GO,
14 tag used/consumed, 15 crashed, 16 unacknowledged crash, 17 failure after the ledger commit, 18 tag not registered/pinned, 19 commit unknown)
and 11 when the shared GPU lock /tmp/cgt_gpu.lock is held (taken atomically with an owner nonce before anything else runs, never waits, never
steals, released in `finally`; `cgt.pipeline.gpulock.GpuLock`, as the other GPU stages).
a registered tag with protocol 'letterlogit_c5' (letterlogit_4b_c5_v1) is run by cgt.p3m_matched.letterlogit_c5.run_sealed (same lock, same
sealing, same exit codes); a tag whose registry entry declares `mlx_env` (the 4B tags: shorter Metal command buffers, as Step C / Step A-4B)
gets those variables set before MLX is imported. matched_v1 / shotvl3b_matched_v1 run exactly as before.
Usage: uv run python scripts/p3m_matched_eval.py --tag matched_v1"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.runstatus import now_iso

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/p3m_matched/raw_eval_logs"
REFUSAL_LOGS = ROOT / "data/cache/p3m_eval/refusal_logs"  # gitignored (data/cache/*)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--go-file", default=None, help="default results/p3m_matched/GO_<tag>")
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--ledger", default=None, help="ShotBench ledger (default results/phase3/shotbench_evaluated.json); tests use a scratch copy")
    ap.add_argument("--lock", default=None, help="GPU lock path (default /tmp/cgt_gpu.lock)")
    a = ap.parse_args()
    from cgt.eval.shotbench import LEDGER, SB
    from cgt.p3b_vlm import sealed
    from cgt.p3m_matched import sealed_eval
    from cgt.pipeline.gpulock import GpuLock, LockHeld
    ledger = Path(a.ledger) if a.ledger else LEDGER
    go = Path(a.go_file) if a.go_file else sealed_eval.go_path(a.tag)
    raw = Path(a.raw_dir)
    rawf = raw / f"{a.tag}.stdout.txt"
    # the committed canonical <tag>.stdout.txt is opened only after the preflight passed. Before that, lines go to a refusal log: the
    # gitignored REFUSAL_LOGS dir when --raw-dir is the canonical dir, else (scratch dirs, tests) the given raw dir's stdout file as before.
    pre = (REFUSAL_LOGS if raw.resolve() == RAW.resolve() else raw) / f"{a.tag}.stdout.txt"
    pre.parent.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()

    def _open(path: Path) -> None:
        if not path.exists():
            path.write_text(f"# raw stdout of scripts/p3m_matched_eval.py --tag {a.tag}\n# started {now_iso()}; code commit {sha}\n# argv: {sys.argv}\n")
    _open(pre)
    state = {"path": pre, "buffer": []}

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with open(state["path"], "a") as f:
            f.write(line + "\n")
        if state["path"] == pre:
            state["buffer"].append(line)

    def on_preflight() -> None:
        if state["path"] == rawf:
            return
        raw.mkdir(parents=True, exist_ok=True)
        _open(rawf)
        with open(rawf, "a") as f:
            f.writelines(line + "\n" for line in state["buffer"])
        state["path"] = rawf

    try:
        lock = GpuLock(f"p3m_matched_eval {a.tag}", Path(a.lock) if a.lock else None).acquire()
    except LockHeld as exc:
        log(f"REFUSED: {exc} (exit 11); nothing evaluated")
        sys.exit(11)
    try:
        _run(a, go, ledger, raw, sha, log, sealed, sealed_eval, SB, on_preflight)
    finally:
        lock.release()


def _run(a, go: Path, ledger: Path, raw: Path, sha: str, log, sealed, sealed_eval, SB, on_preflight) -> None:
    from cgt.p3m_matched import registry
    ev = sealed.SealedEvaluation(a.tag, raw, log, code_commit=sha)
    sealed.install_signal_handlers()
    spec = registry.MODELS.get(a.tag) or {}  # lookup only; every refusal (unregistered / unpinned tag included) is in the preflight
    for k, v in (spec.get("mlx_env") or {}).items():
        os.environ[k] = v
    try:
        if registry.protocol_of(spec) == "letterlogit_c5":
            from cgt.p3m_matched import letterlogit_c5
            letterlogit_c5.run_sealed(a.tag, go, ledger, raw, SB / "test.tsv", SB, ROOT / f"data/cache/p3m_eval/shotbench_images_{a.tag}",
                                      ROOT / f"data/cache/p3m_eval/siglip_features_{a.tag}", ROOT / "results/p3m_matched", log, ev,
                                      on_preflight=on_preflight)
        else:
            sealed_eval.run_sealed(a.tag, go, ledger, raw, SB / "test.tsv", SB, ROOT / f"data/cache/p3m_eval/shotbench_images_{a.tag}",
                                   ROOT / "results/p3m_matched", log, ev, on_preflight=on_preflight)
    except guards.FailClosed as e:
        if ev.started and not ev.finished:
            ev.record_crash(e)
        code = sealed.exit_code_for(ev, e)
        log(f"FAIL CLOSED ({sealed.safe_text(e)}, exit {code})")
        sys.exit(code)
    except BaseException as e:  # noqa: BLE001
        if ev.started and not ev.finished:
            ev.record_crash(e)
        log(f"CRASHED ({sealed.safe_text(e)}); no scores written")
        sys.exit(sealed.exit_code_for(ev, e))


if __name__ == "__main__":
    main()
