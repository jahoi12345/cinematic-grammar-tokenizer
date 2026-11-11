"""Run provenance and the human-readable RUN_PROGRESS.md for long Phase 3B runs (shared by the trainer and the supervisor)."""
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAN_COMMIT = "dfc14eca89594030112f1e26588dafd3344427a1"
PLAN_FILE = "docs/phase3b_plan.md"
AMEND_MARK = "\n## Amendments"


def now_iso() -> str:
    """Local time with offset and the same instant in UTC, e.g. '2026-10-02T02:25:05+02:00 (2026-10-02T00:25:05Z)'."""
    t = datetime.now().astimezone()
    return f"{t.isoformat(timespec='seconds')} ({t.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')})"


def _git(*a: str, root: Path = ROOT) -> str:
    return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout


def plan_original_unchanged(root: Path = ROOT, plan_commit: str = PLAN_COMMIT, plan_file: str = PLAN_FILE) -> dict:
    """The pre-registered plan text (everything before '## Amendments') must equal its text in the plan commit. Returns digests.
    `root`/`plan_commit`/`plan_file` exist so a test can use a synthetic repository."""
    orig = _git("show", f"{plan_commit}:{plan_file}", root=root).split(AMEND_MARK)[0]
    cur = (Path(root) / plan_file).read_text().split(AMEND_MARK)[0]
    return {"plan_commit": plan_commit, "original_text_sha256": hashlib.sha256(orig.encode()).hexdigest(),
            "current_text_sha256": hashlib.sha256(cur.encode()).hexdigest(), "unchanged": orig == cur}


def software_provenance() -> dict:
    import mlx.core as mx
    import mlx_vlm
    import transformers
    tdir = Path(mlx_vlm.__file__).parent / "trainer"
    return {"mlx": mx.__version__, "mlx_vlm": getattr(mlx_vlm, "__version__", None), "transformers": transformers.__version__,
            "trainer_sha256": {f: hashlib.sha256((tdir / f).read_bytes()).hexdigest() for f in ("sft_trainer.py", "datasets.py", "utils.py")}}


def write_status(logd: Path, tag: str, **kw) -> None:
    """Overwrite results/phase3b/runs/<tag>/RUN_PROGRESS.md with the supplied fields (state, update, total, loss, sec_per_update, peak_gb, ...)."""
    restarts = 0
    rf = Path(logd) / "restarts.jsonl"
    if rf.exists():
        restarts = len([x for x in rf.read_text().splitlines() if x.strip()])
    rem = kw.get("total", 0) - kw.get("update", 0)
    spu = kw.get("sec_per_update")
    eta = f"{rem * spu / 3600:.1f} h (training only, excludes the post-training validation)" if spu and rem > 0 else "-"
    lines = [f"# STATUS {tag}", "", f"- written: {now_iso()}", f"- state: **{kw.get('state', '?')}**"]
    if "update" in kw:
        def show(v):  # a field the caller did not supply is shown as "-", never "None"
            return "-" if v is None else v
        lines += [f"- update: {kw['update']} / {show(kw.get('total'))} (samples {show(kw.get('micro'))})",
                  f"- loss (mean of last 10 updates): {show(kw.get('loss'))}", f"- seconds per update (mean of last 20): {show(spu)}",
                  f"- peak memory GB: {show(kw.get('peak_gb'))}"]
    lines += [f"- restarts so far: {restarts}", f"- last checkpoint: {kw.get('last_checkpoint', '-')}", f"- ETA: {eta}"]
    if kw.get("note"):
        lines += ["", kw["note"]]
    (Path(logd) / "RUN_PROGRESS.md").write_text("\n".join(lines) + "\n")


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"time": now_iso(), **rec}) + "\n")


def monotonic() -> float:
    return time.monotonic()
