"""Fail-closed checks for the Phase 3B entry points (docs/phase3b_interface.md section 6, docs/phase3b_plan.md amendment 1).
Every check raises a `FailClosed` subclass with a distinct process exit code; the scripts catch it, log it and exit with that code.
A supervisor must NOT restart the run on these codes."""
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from cgt.p3_static_attrs.data import SHOTQA_ROOT, assert_shotqa_path

ROOT = Path(__file__).resolve().parents[3]


class FailClosed(RuntimeError):
    exit_code = 1


class LeakProbeFailed(FailClosed):
    exit_code = 4


class DataRootViolation(FailClosed):
    exit_code = 5


class UnreadableImage(FailClosed):
    exit_code = 6


class AdapterHashError(FailClosed):
    exit_code = 7


class TokenLengthViolation(FailClosed):
    exit_code = 8


class DirtyTree(FailClosed):
    exit_code = 10


class NotAuthorized(FailClosed):
    """The operator's go-ahead file for the ShotBench evaluation does not exist."""
    exit_code = 13


class EvaluationCrashed(FailClosed):
    """The ShotBench evaluation died before the ledger entry was recorded: a crash file (reason and progress count only) was written."""
    exit_code = 15


class CrashUnacknowledged(FailClosed):
    """A previous attempt crashed or was killed and the operator has not acknowledged it (no <tag>.CRASH_ACK file)."""
    exit_code = 16


class PostLedgerWriteFailed(FailClosed):
    """The ledger entry was recorded but writing the result files failed (the results were echoed to stdout instead)."""
    exit_code = 17


class TagNotBound(FailClosed):
    """The ledger tag given on the command line is not the evaluation tag designated by the frozen manifest."""
    exit_code = 18


class CommitUnknown(FailClosed):
    """The evaluation finished but it could not be established whether the ledger entry was recorded: no result is published (exit 19)."""
    exit_code = 19


class SameStemImages(FailClosed):
    """Two distinct image source files share a stem (different extension, extension case or directory): refused before any cache entry exists."""
    exit_code = 20


class LedgerUsed(FailClosed):
    """The model tag is already in the ShotBench ledger (each tag may be evaluated exactly once)."""
    exit_code = 14


def require_zero_leak(leak: dict) -> None:
    """The leakage probe must report exactly 0.0 without a mask (a missing or non-numeric value also fails)."""
    v = leak.get("without_mask") if isinstance(leak, dict) else None
    if not isinstance(v, int | float) or v != 0.0:
        raise LeakProbeFailed(f"leak probe without_mask must be exactly 0.0, got {v!r} (full result {leak})")


def require_samples_under_root(samples: Iterable[dict], root: Path = SHOTQA_ROOT) -> int:
    """Every sample's image must resolve (symlinks followed) inside the ShotQA root. Returns the number checked."""
    n = 0
    for s in samples:
        n += 1
        for p in s["images"]:
            try:
                assert_shotqa_path(p) if Path(root) == SHOTQA_ROOT else _inside(p, root)
            except ValueError as e:
                raise DataRootViolation(f"sample {s.get('id')}: {e}") from e
    return n


def _inside(p, root: Path) -> None:
    r, q = Path(root).resolve(), Path(p).resolve()
    if q != r and r not in q.parents:
        raise ValueError(f"{q} is outside {r}")


def require_unique_stems(paths: Iterable[str | Path]) -> int:
    """Distinct source files must have distinct stems (compared case-insensitively); repeated references to the SAME path are allowed.
    Works on names only (no file is opened). Raises SameStemImages (exit 20) naming at most the count, never reading content. Returns the file count."""
    seen: dict[str, str] = {}
    for p in dict.fromkeys(str(x) for x in paths):
        stem = Path(p).stem.casefold()
        if stem in seen:
            raise SameStemImages(f"two distinct image files share the stem '{stem}': {seen[stem]} and {p}")
        seen[stem] = p
    return len(seen)


def require_images_readable(paths: Iterable[str]) -> int:
    """Every image file must exist and open as an image (header read). Raises UnreadableImage listing the first failures."""
    bad, n = [], 0
    for p in paths:
        n += 1
        try:
            with Image.open(p) as im:
                im.size  # noqa: B018 - forces the header decode
        except (OSError, ValueError) as e:
            bad.append(f"{p}: {type(e).__name__}")
    if bad:
        raise UnreadableImage(f"{len(bad)} unreadable images, first: {bad[:5]}")
    return n


def total_teacher_forced_tokens(processor, sample: dict, image_tokens: int = 144) -> int:
    """Upper bound of the teacher-forced sequence length: the templated prompt AND the answer turn (letter + end-of-turn tokens)
    tokenised as text, plus `image_tokens` (the maximum image-token budget; the image placeholder itself is counted once in the text)."""
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["messages"][0]["content"]}]}
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": sample["messages"][1]["content"]}]}],
        add_generation_prompt=False)
    return len(processor.tokenizer.encode(full, add_special_tokens=False)) + image_tokens


def require_token_lengths(processor, samples: Iterable[dict], limit: int = 1024, image_tokens: int = 144) -> dict:
    """Abort (do NOT skip) if any sample's total teacher-forced length exceeds `limit`. Returns {n, max_tokens, n_over_limit: 0}."""
    n, mx_len, over = 0, 0, []
    for s in samples:
        n += 1
        t = total_teacher_forced_tokens(processor, s, image_tokens)
        mx_len = max(mx_len, t)
        if t > limit:
            over.append((s.get("id"), t))
    if over:
        raise TokenLengthViolation(f"{len(over)} samples exceed {limit} total tokens (would be truncated or skipped), first: {over[:3]}")
    return {"n": n, "max_tokens": mx_len, "n_over_limit": 0, "limit": limit}


def sha256_file(path: Path) -> str:
    """SHA-256 of an adapter file; raises AdapterHashError if it is missing or empty."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise AdapterHashError(f"adapter file missing or empty: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_registered_hash(manifest_path: Path, adapter: Path) -> str:
    """The adapter's SHA-256 must be listed in the run manifest (a checkpoint entry with the same digest)."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise AdapterHashError(f"manifest missing: {mp}")
    digest = sha256_file(adapter)
    registered = {c.get("sha256") for c in json.loads(mp.read_text()).get("checkpoints", {}).values()}
    if digest not in registered:
        raise AdapterHashError(f"adapter {adapter} (sha256 {digest}) is not registered in {mp}")
    return digest


def verify_manifest_files(manifest: dict, run_dir: Path) -> int:
    """Every checkpoint recorded in the manifest must still exist with the recorded SHA-256 (detects corruption/tampering on resume)."""
    n = 0
    for pct, c in manifest.get("checkpoints", {}).items():
        n += 1
        if "sha256" not in c or not c["sha256"]:
            raise AdapterHashError(f"manifest entry {pct}% has no sha256")
        if sha256_file(Path(run_dir) / c["file"]) != c["sha256"]:
            raise AdapterHashError(f"checkpoint {c['file']} no longer matches its recorded sha256")
    return n


def git_state(paths: Iterable[str]) -> dict:
    """{commit, dirty_files}: HEAD commit and tracked/untracked changes under `paths` (relative to the repo root)."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = [ln for ln in run("status", "--porcelain", "--", *paths).splitlines() if ln.strip()]
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": dirty}


def require_clean_tree(paths: Iterable[str]) -> dict:
    """The code that runs must be committed: raises DirtyTree if any file under `paths` is modified or untracked. Returns the git state."""
    st = git_state(paths)
    if st["dirty_files"]:
        raise DirtyTree(f"uncommitted changes under the run's code paths: {st['dirty_files'][:6]}")
    return st


def go_text(tag: str, adapter_sha256: str) -> str:
    """The exact content of the operator's go-ahead file: bound to the ledger tag and to the frozen adapter's SHA-256."""
    return f"SHOTBENCH_GO\ntag={tag}\nadapter_sha256={adapter_sha256}\n"


def require_go_file(path: Path, tag: str | None = None, adapter_sha256: str | None = None) -> None:
    """The ShotBench evaluation may only run if the operator's go-ahead file exists. With `tag` and `adapter_sha256` the file's content must be
    exactly `go_text(tag, adapter_sha256)`: a go-ahead for another tag or another adapter authorizes nothing."""
    p = Path(path)
    if not p.is_file():
        raise NotAuthorized(f"go-ahead file missing: {p} (the operator creates it after the freeze is verified)")
    if tag is not None and p.read_text() != go_text(tag, adapter_sha256 or ""):
        raise NotAuthorized(f"go-ahead file {p} is not bound to tag '{tag}' and the frozen adapter hash (write it with scripts/p3b_write_go.py)")


def require_evaluation_tag(tag: str, frozen_manifest_path: Path) -> dict:
    """The tag must equal the frozen manifest's designated ShotBench evaluation tag (one frozen adapter, one ledger slot). Returns the manifest."""
    fm = json.loads(Path(frozen_manifest_path).read_text())
    want = fm.get("shotbench_ledger_tag_for_the_evaluation")
    if not want or tag != want:
        raise TagNotBound(f"tag '{tag}' is not the frozen manifest's evaluation tag '{want}'")
    return fm


def require_plan_unchanged(root: Path | None = None, plan_commit: str | None = None, plan_file: str | None = None) -> dict:
    """The original pre-registered plan text (everything before '## Amendments') must equal its text in the plan commit (DirtyTree, exit 10).
    Appending to the amendments section stays allowed. Must run before any model or image access."""
    from cgt.p3b_vlm import runstatus
    kw = {k: v for k, v in (("root", root), ("plan_commit", plan_commit), ("plan_file", plan_file)) if v is not None}
    plan = runstatus.plan_original_unchanged(**kw)
    if not plan["unchanged"]:
        raise DirtyTree("the original pre-registered plan text differs from its committed text (amendments are allowed, edits above them are not)")
    return plan


def require_under_image_root(path: str | Path, root: Path) -> Path:
    """Resolve `path` and require it to lie inside `root` and to reach it without any symlink component (relative escapes, absolute paths and
    symlinks raise DataRootViolation, exit 5). Never opens the file. Returns the resolved path."""
    r = Path(root).resolve()
    p = Path(path)
    if not p.is_absolute() and ".." in p.parts:
        raise DataRootViolation(f"path contains '..': {path}")
    q = p if p.is_absolute() else r / p
    if p.is_absolute() and r not in q.resolve().parents:
        raise DataRootViolation(f"absolute path outside the image root {r}: {path}")
    if r not in q.resolve().parents:
        raise DataRootViolation(f"{path} resolves outside the image root {r}")
    cur = q
    while cur != r and cur != cur.parent:
        if cur.is_symlink():
            raise DataRootViolation(f"symlink on the path (not allowed): {cur}")
        cur = cur.parent
    return q.resolve()


def require_frozen_adapter(frozen_manifest_path: Path, run_dir: Path) -> dict:
    """Recompute the SHA-256 of the frozen adapter and require it to equal both the freeze manifest and the run manifest entry.
    Returns {"adapter_path", "sha256", "checkpoint_percent", "size_bytes", "mtime"}; raises AdapterHashError otherwise."""
    fm = json.loads(Path(frozen_manifest_path).read_text())
    adapter = Path(run_dir) / fm["adapter_file"]
    digest = sha256_file(adapter)
    if digest != fm["adapter_sha256"]:
        raise AdapterHashError(f"frozen adapter {adapter} has sha256 {digest}, the freeze manifest says {fm['adapter_sha256']}")
    run_manifest = json.loads((ROOT / "results/phase3b/runs" / Path(run_dir).name / "manifest.json").read_text())
    entry = run_manifest["checkpoints"][str(fm["selected_checkpoint_percent"])]
    if entry["sha256"] != digest:
        raise AdapterHashError("the run manifest entry of the selected checkpoint does not match the recomputed sha256")
    st = adapter.stat()
    return {"adapter_path": str(adapter), "sha256": digest, "checkpoint_percent": fm["selected_checkpoint_percent"],
            "size_bytes": st.st_size, "mtime": st.st_mtime}
