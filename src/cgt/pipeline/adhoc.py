"""Ad-hoc inputs for the one-command entry point: input checks, a stable content-derived id, and the overlay entry.
Nothing here computes a model output; dry runs write no file."""
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from cgt.pipeline import identity
from cgt.pipeline import registry as reg
from cgt.pipeline.orchestrator import PipelineError

DEFAULT_CUT_THRESHOLD = 27.0  # the library default (cgt.p1_shots_color.pipeline.DEFAULT_CUT_THRESHOLD; repeated so planning stays import-light)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
SOURCE = "ad-hoc input via scripts/tokenize_video.py"
OWNER_STAGES = ("p1", "p2", "p3c", "move")  # stages whose code (owned by others) resolves the film itself


def hash_cache(root: Path) -> Path:
    return Path(root) / "data/cache/adhoc/hash_cache.json"


def check_readable(path: Path) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise PipelineError("bad_input", f"input not found: {p}")
    if not p.is_file():
        raise PipelineError("bad_input", f"input is not a regular file: {p}")
    if not os.access(p, os.R_OK):
        raise PipelineError("bad_input", f"input is not readable: {p}")
    return p.resolve()


def probe_video(path: Path) -> dict:
    """ffprobe must report at least one video stream with a positive frame size; anything else (text, audio only, a still image without a frame count) is refused."""
    exe = shutil.which("ffprobe")
    if exe is None:
        raise PipelineError("bad_input", "ffprobe not found on PATH: the input cannot be checked, so it is refused (install ffmpeg)")
    r = subprocess.run([exe, "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,nb_frames,duration:format=duration,format_name", "-of", "json", str(path)],
                       capture_output=True, text=True, check=False)
    try:
        info = json.loads(r.stdout or "{}")
    except ValueError:
        info = {}
    vids = [s for s in info.get("streams", []) if s.get("codec_type") == "video" and int(s.get("width") or 0) > 0 and int(s.get("height") or 0) > 0]
    fmt = info.get("format", {})
    if r.returncode != 0 or not vids:
        why = (r.stderr or "").strip().splitlines()[-1:] or ["no video stream"]
        raise PipelineError("bad_input", f"not a video file (ffprobe: {why[0]}): {path}")
    v = vids[0]
    if v.get("codec_name") in {"png", "mjpeg", "bmp", "gif", "webp", "tiff"} and float(fmt.get("duration") or 0) <= 0:
        raise PipelineError("bad_input", f"a still image, not a video: {path}")
    return {"width": int(v["width"]), "height": int(v["height"]), "codec": v.get("codec_name"), "duration_s": float(fmt.get("duration") or v.get("duration") or 0)}


def content_sha256(path: Path, root: Path, dry_run: bool) -> str:
    return identity.cached_sha256(Path(path), hash_cache(root), write=not dry_run)[0]


def slug(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:40].strip("_") or "video"


def derive_id(path: Path, sha: str) -> str:
    """slug of the file stem + the first 8 hex of the content SHA-256: depends on the name stem and the bytes only (same file -> same id, every run)."""
    return f"{slug(Path(path).stem)}_{sha[:8]}"


def build_entry(path: Path, sha: str, film_id: str, title: str | None, threshold: float | None, public_domain: bool, existing: dict | None) -> dict:
    """The overlay entry for this invocation. A stored threshold is kept when none is given; no permission field is ever stored."""
    if threshold is not None:
        thr, src = float(threshold), "user"
    elif existing is not None and existing.get("cut_threshold") is not None:
        thr, src = float(existing["cut_threshold"]), existing.get("cut_threshold_source", "default, unchecked")
    else:
        thr, src = DEFAULT_CUT_THRESHOLD, "default, unchecked"
    return {"id": film_id, "title": title or (existing or {}).get("title") or Path(path).stem, "path": str(path), "sha256": sha, "size": Path(path).stat().st_size,
            "public_domain": bool(public_domain), "subtitles": None, "cut_threshold": thr, "cut_threshold_source": src, "source": SOURCE}


def resolve(path: Path, root: Path, *, film_id: str | None = None, title: str | None = None, threshold: float | None = None, public_domain: bool = False,
            dry_run: bool = False) -> dict:
    """Decide what this invocation registers. Returns {"film_id", "entry" (the overlay entry, or None for a films.yaml film), "action": "main" | "new" | "unchanged" |
    "updated", "changes": [...], "video": ffprobe summary}. Raises PipelineError(bad_input | registry_clash). Writes nothing."""
    path = check_readable(path)
    video = probe_video(path)
    if film_id is not None and not ID_RE.match(film_id):
        raise PipelineError("bad_input", f"unsafe --id '{film_id}': use lowercase letters, digits and underscores (at most 64, starting with a letter or digit)")
    try:
        films = reg.load_films(root)
    except reg.RegistryClash as e:
        raise PipelineError("registry_clash", str(e)) from e
    for fid, e in films.items():  # the input IS a film of films.yaml: use it as registered
        if e["registry"] == "main" and (Path(root) / e.get("path", "")).resolve() == path and film_id in (None, fid):
            return {"film_id": fid, "entry": None, "action": "main", "changes": [], "video": video}
    sha = content_sha256(path, root, dry_run)
    fid = film_id or derive_id(path, sha)
    cur = films.get(fid)
    if cur is not None and cur["registry"] == "main":
        raise PipelineError("registry_clash", f"id '{fid}' is a film of configs/films.yaml with another file ({cur.get('path')}); choose another --id")
    if cur is not None and cur.get("sha256") != sha:
        raise PipelineError("registry_clash", f"id '{fid}' is already bound to another file in {reg.OVERLAY} (sha256 {str(cur.get('sha256'))[:12]}...); "
                                              "an id never switches files: choose another --id")
    entry = build_entry(path, sha, fid, title, threshold, public_domain, cur)
    if cur is None:
        return {"film_id": fid, "entry": entry, "action": "new", "changes": [], "video": video}
    old = {k: v for k, v in cur.items() if k != "registry"}
    changes = [f"{k}: {old.get(k)!r} -> {v!r}" for k, v in entry.items() if old.get(k) != v]
    return {"film_id": fid, "entry": entry, "action": "updated" if changes else "unchanged", "changes": changes, "video": video}


def register(root: Path, entry: dict) -> Path:
    """Insert or replace `entry` in the overlay (atomic)."""
    entries = [e for e in reg.read_overlay(root) if e["id"] != entry["id"]] + [entry]
    return reg.write_overlay(root, entries)


# ---------------------------------------------------------------- owner-module support probes (READ-ONLY modules; see docs/v1_entry_point.md "Needs owner change")
def _probe_p1(film_id: str) -> str | None:
    from cgt.p1_shots_color import pipeline as p1
    try:
        p1.film_entry(film_id)
    except (StopIteration, KeyError, ValueError) as e:
        return f"cgt.p1_shots_color.pipeline.film_entry reads configs/films.yaml only ({type(e).__name__})"
    return None


def _probe_p3c(film_id: str) -> str | None:
    try:
        from cgt.p3c_tokens import runner
        runner.require_public_domain(film_id, allow_owned=True)
    except ImportError as e:
        return f"Phase 3C runner cannot be imported here ({e})"
    except Exception as e:  # noqa: BLE001 - any refusal means the runner cannot resolve the film
        return f"cgt.p3c_tokens.runner.require_public_domain refuses it ({e})"
    return None


def _probe_move(film_id: str) -> str | None:
    try:
        from cgt.p4_camera_move import film_move as FM
    except ImportError as e:
        return f"film_move cannot be imported here ({e})"
    try:
        FM.resolve_film(film_id, allow_owned=True)
    except FM.Refused as e:
        if "not registered" in str(e):
            return f"scripts/run_move_film.py -> film_move.resolve_film reads configs/films.yaml only ({e})"
    return None  # registered: a missing shot table at this point is the normal p1 prerequisite


PROBES: dict[str, Callable[[str], str | None]] = {"p1": _probe_p1, "p2": _probe_p1, "p3c": _probe_p3c, "move": _probe_move}


def owner_blockers(film_id: str, stages: list[str], probes: dict[str, Callable[[str], str | None]] | None = None) -> dict[str, str]:
    """{stage: reason} for requested stages whose owner module cannot resolve an overlay film yet."""
    probes = PROBES if probes is None else probes
    out = {}
    for s in stages:
        if s in probes:
            why = probes[s](film_id)
            if why:
                out[s] = why
    return out
