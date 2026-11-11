"""Film-level movement slot by INFERENCE with the frozen geometry model (docs/phase4_film_move.md; Amendments 10 and 11 of docs/phase4_video_plan.md): the pure logic shared by scripts/run_move_film.py.
Film registry refusals, the lossless 23.976 fps shot clip, the cache identity of per-shot series, the decision rule (atoms with p >= 0.5, else the single top-1 atom), the descriptive statistics including the
CONTRADICTORY-SET rate. Nothing here trains, tunes or reads a label of the film; every output is a PREDICTION of unmeasured accuracy."""
import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

from cgt.p4_camera_move import depth_parallax as D
from cgt.p4_camera_move import flow as F
from cgt.p4_camera_move.labels import schema_moves
from cgt.pipeline import registry

ROOT = Path(__file__).resolve().parents[3]
TARGET_FPS = 24000 / 1001  # the frame rate of the training clips (Amendment 11)
THRESHOLD = 0.5  # Amendment 10: the only threshold
OPPOSITE_PAIRS = (("push-in", "pull-out"), ("pan-left", "pan-right"), ("tilt-up", "tilt-down"), ("boom-up", "boom-down"), ("zoom-in", "zoom-out"), ("move-left", "move-right"))
MIN_FREE_PERCENT = 25


class Refused(Exception):
    """A guard of the film runner (exit code in `code`): 2 = film/inputs/output refused, 12 = memory guard."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def resolve_film(film_id: str, *, allow_owned: bool = False, root: Path = ROOT, films_yaml: Path | None = None) -> dict:
    """The film entry of the merged registry (configs/films.yaml + local overlay; or of `films_yaml` alone when given) with absolute `film_path` and `shots_path`. Refuses an unregistered film, an owned (not public_domain) film without `allow_owned`, a missing film file or shot table; an overlay film always needs `allow_owned`."""
    if films_yaml is not None:  # an explicit registry file: that file only (as before)
        reg = yaml.safe_load(Path(films_yaml).read_text())["films"]
        entry = next((f for f in reg if f["id"] == film_id), None)
    else:  # the merged registry: configs/films.yaml + the local overlay configs/films.local.yaml; a films.yaml entry is returned unchanged
        try:
            entry = registry.load_films(Path(root)).get(film_id)
        except registry.RegistryClash as e:
            raise Refused(2, str(e)) from e
        if entry is not None and not registry.is_overlay(entry):
            entry = {k: v for k, v in entry.items() if k != "registry"}
    if entry is None:
        raise Refused(2, f"film '{film_id}' is not registered in configs/films.yaml")
    if registry.is_overlay(entry) and not allow_owned:
        raise Refused(2, f"film '{film_id}' is an ad-hoc film of the local overlay (local only): pass --allow-owned for local, unpublished use")
    if not entry.get("public_domain") and not allow_owned:
        raise Refused(2, f"film '{film_id}' is owned (not public_domain): pass --allow-owned for local, unpublished use")
    film, shots = Path(root) / entry["path"], Path(root) / f"data/processed/{film_id}_shots.parquet"
    if not film.is_file() or not shots.is_file():
        raise Refused(2, f"missing film file or shot table for '{film_id}'")
    return {**entry, "film_path": film, "shots_path": shots}


def refuse_existing(out: Path, overwrite: bool) -> None:
    if Path(out).exists() and not overwrite:
        raise Refused(2, f"{out} exists: pass --overwrite (the accepted tables are never overwritten by default)")


def free_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.split(":")[1].strip().rstrip("%"))
    return 0


def memory_guard(reader=free_percent) -> None:
    free = reader()
    if free < MIN_FREE_PERCENT:
        raise Refused(12, f"memory guard: {free}% free < {MIN_FREE_PERCENT}%; nothing started")


def film_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def film_key(film_sha: str, start_s: float, end_s: float) -> str:
    """Film hash + frame range + conformity tag of a shot's cache entries."""
    return f"{film_sha[:12]}.{round(start_s * 1000)}-{round(end_s * 1000)}.r24"


def cache_names(cache_dir: Path, sid: int, fkey: str) -> tuple[Path, Path]:
    """The flow and depth cache files of a shot: keyed by the flow code, the depth code, the film bytes, the frame range and the conformity tag."""
    key = f"{F.cache_key()}.{D.cache_key()}.{fkey}"
    return Path(cache_dir) / "flow" / f"shot{sid}.{key}.npz", Path(cache_dir) / "depth" / f"shot{sid}.{key}.npz"


def cut_clip(film: Path, start_s: float, end_s: float, dst: Path) -> int:
    """The shot written LOSSLESSLY (FFV1) at TARGET_FPS: output frame j is the film frame nearest to start_s + j / TARGET_FPS, j < floor((end_s - start_s) x TARGET_FPS); returns the frame count."""
    cap = cv2.VideoCapture(str(film))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_out = int((end_s - start_s) * TARGET_FPS)
    src = [round((start_s + j / TARGET_FPS) * fps) for j in range(n_out)]
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"FFV1"), TARGET_FPS, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, src[0] if src else 0)
    pos, last, n = src[0] if src else 0, None, 0
    for i in src:
        while pos <= i:
            ok, fr = cap.read()
            if not ok:
                break
            last, pos = fr, pos + 1
        if last is None:
            break
        wr.write(last)
        n += 1
    wr.release()
    cap.release()
    return n


def predict(P: np.ndarray, atoms: list[str]) -> tuple[list[list[str]], list[bool]]:
    """Amendment 10 item 2: atoms with p >= 0.5, else the single top-1 atom (never empty). Returns (sets in `atoms` order, fallback flags)."""
    sets, fb = [], []
    for p in P:
        idx = [j for j in range(len(atoms)) if p[j] >= THRESHOLD]
        fb.append(not idx)
        sets.append([atoms[j] for j in (idx or [int(np.argmax(p))])])
    return sets, fb


def is_contradictory(atoms) -> dict:
    """The declared definition (docs/phase4_film_move.md): both atoms of an opposite pair, or `static` together with any other atom."""
    s = set(atoms)
    opposite = any(a in s and b in s for a, b in OPPOSITE_PAIRS)
    static_other = "static" in s and len(s) > 1
    return {"opposite_pair": opposite, "static_with_other": static_other, "contradictory": opposite or static_other}


def describe(sets: list[list[str]], fallback: list[bool]) -> dict:
    """Set-size distribution, static alone, fallbacks, and the contradictory-set rate with its two components."""
    n = len(sets)
    sizes = {}
    for s in sets:
        sizes[len(s)] = sizes.get(len(s), 0) + 1
    c = [is_contradictory(s) for s in sets]
    order = {a: i for i, a in enumerate(schema_moves())}
    return {"n_shots": n, "set_sizes": dict(sorted(sizes.items())), "median_set_size": float(np.median([len(s) for s in sets])), "static_alone": sum(s == ["static"] for s in sets), "static_alone_share": sum(s == ["static"] for s in sets) / n,
            "fallback_top1": int(sum(fallback)), "contradictory": sum(x["contradictory"] for x in c), "contradictory_share": sum(x["contradictory"] for x in c) / n,
            "opposite_pair": sum(x["opposite_pair"] for x in c), "static_with_other": sum(x["static_with_other"] for x in c),
            "atoms_in_schema_order": sorted({a for s in sets for a in s}, key=order.get)}
