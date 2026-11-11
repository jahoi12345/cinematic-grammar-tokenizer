"""Content identity of pipeline inputs (video files, keyframe images), cached against (size, mtime) so repeated checks are cheap. A cache that predates identity recording is NEVER
adopted from timestamps or matching ids: it must be re-derived or re-scored from the actual inputs (docs/film_onboarding.md)."""
import hashlib
import json
import os
from pathlib import Path

CHUNK = 1 << 22


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def cached_sha256(path: Path, cache_file: Path, write: bool = True) -> tuple[str, int]:
    """(sha256, size) of `path`; the digest is reused from `cache_file` only while the recorded size AND mtime_ns still match (a replaced file is re-hashed)."""
    st = path.stat()
    c = _load(cache_file)
    e = c.get(str(path))
    if e and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns:
        return e["sha256"], st.st_size
    sha = file_sha256(path)
    if not write:  # dry run: hash, but leave no file behind
        return sha, st.st_size
    c[str(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(c))
        os.replace(tmp, cache_file)
    except OSError:
        pass  # a read-only location only costs a re-hash next time
    return sha, st.st_size


def digest_of_files(paths: list[Path], cache_file: Path, write: bool = True) -> str:
    """One digest over the CONTENT of every file, in the given order: sha256 of the lines `<path-relative-name> <sha256>`."""
    h = hashlib.sha256()
    for p in paths:
        sha, _ = cached_sha256(Path(p), cache_file, write)
        h.update(f"{Path(p).name} {sha}\n".encode())
    return h.hexdigest()
