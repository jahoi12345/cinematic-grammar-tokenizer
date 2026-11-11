"""The agreed cross-worker GPU lock `/tmp/cgt_gpu.lock` (one heavy GPU job at a time), overridable with the environment variable CGT_GPU_LOCK (CPU tests only). `acquire` never overwrites another
owner's lock and records a unique owner token (job, pid, random nonce); `release` removes the lock ONLY if it carries OUR pid and nonce, i.e. only a lock this process acquired (never another
process's lock, even one with the same job name; Amendment 8 / review V4)."""
import os
import secrets
import time
from pathlib import Path

DEFAULT = "/tmp/cgt_gpu.lock"
LOCK = Path(os.environ.get("CGT_GPU_LOCK", DEFAULT))  # read at import; tests may also patch this attribute
_mine: dict[str, str] = {}  # lock path -> nonce of the lock THIS process acquired


def lock_path() -> Path:
    return LOCK


def owner() -> str | None:
    p = lock_path()
    return p.read_text().strip() if p.exists() else None


def acquire(job: str, tag: str = "") -> bool:
    """Create the lock atomically (exclusive create). Returns False if another owner holds it; on success remembers the nonce written into the lock."""
    p = lock_path()
    nonce = secrets.token_hex(8)
    try:
        with open(p, "x") as f:
            f.write(f"{job} pid={os.getpid()} nonce={nonce} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')} tag={tag}\n")
    except FileExistsError:
        return False
    _mine[str(p)] = nonce
    return True


def owned() -> bool:
    """True iff the lock file exists and carries the pid and nonce of the lock this process acquired."""
    p = lock_path()
    nonce = _mine.get(str(p))
    return bool(nonce) and p.exists() and f"pid={os.getpid()} " in p.read_text() and f"nonce={nonce}" in p.read_text()


def release(job: str = "") -> bool:
    """Remove the lock only if this process acquired it and it still carries our pid and nonce. `job` is accepted for compatibility and is not trusted. Returns whether it was removed."""
    if not owned():
        return False
    lock_path().unlink(missing_ok=True)
    _mine.pop(str(lock_path()), None)
    return True
