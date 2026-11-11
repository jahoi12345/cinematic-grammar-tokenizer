"""The shared GPU lock (/tmp/cgt_gpu.lock): one heavy GPU job at a time on the Mac. Created atomically (O_EXCL) with an owner nonce; released only by the holder
that wrote that nonce; never waits and never steals (a held lock is a refusal)."""
import os
import secrets
import time
from pathlib import Path

DEFAULT = Path(os.environ.get("CGT_GPU_LOCK", "/tmp/cgt_gpu.lock"))


class LockHeld(RuntimeError):
    def __init__(self, path: Path, holder: str):
        super().__init__(f"GPU lock {path} is held: {holder.strip() or '(empty)'}")
        self.path, self.holder = path, holder


class GpuLock:
    def __init__(self, job: str, path: Path | None = None):
        self.job, self.path = job, Path(path or DEFAULT)
        self.nonce = secrets.token_hex(8)
        self.held = False

    def acquire(self) -> "GpuLock":
        line = f"{self.job} pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')} nonce={self.nonce}\n"
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = self.path.read_text()
            except OSError:
                holder = "(unreadable)"
            raise LockHeld(self.path, holder) from None
        with os.fdopen(fd, "w") as f:
            f.write(line)
        self.held = True
        return self

    def release(self) -> bool:
        """Remove the lock only if it still carries OUR nonce (another job's lock is never removed). Returns True if removed."""
        if not self.held:
            return False
        self.held = False
        try:
            if f"nonce={self.nonce}" in self.path.read_text():
                self.path.unlink()
                return True
        except OSError:
            pass
        return False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
