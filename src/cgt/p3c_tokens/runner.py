"""Cached, idempotent, resumable film runner (docs/phase3c_plan.md section 7). One keyframe column at a time; results are appended per shot
to data/cache/p3c/runs/<film>_<key>/scores_<kf>.jsonl, so a killed run resumes where it stopped and a finished run is a no-op.
Run key = sha256 of {film id, adapter sha256, code git SHA, method, image budget} (first 12 hex); for a non-2B model (pipeline version
p3c-v2-4b, docs/plans/default_4b.md) the blob also holds the pinned model identity, so 2B and 4B runs never share a directory.
`run_film` keeps the 2B as its library default (model="2b": the old default, byte for byte); the film-pipeline default lives in
cgt.p3c_tokens.models.DEFAULT_MODEL and is passed explicitly by the entry points."""
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import BUDGET_PX, MULTIPLE, shrunk_path
from cgt.p3c_tokens import aggregate as ag
from cgt.p3c_tokens import models
from cgt.p3c_tokens.methods import score_attribute
from cgt.p3c_tokens.vocab import ATTRS, MULTI, labels
from cgt.pipeline import registry

ROOT = Path(__file__).resolve().parents[3]
KEYFRAMES = ("kf_mid", "kf_start", "kf_end")  # kf_mid is the primary column
CODE_PATHS = ["src/cgt/p3c_tokens", "scripts/p3c_run_film.py", "docs/phase3c_plan.md", "pyproject.toml", "uv.lock",
              "src/cgt/film4b", "src/cgt/p3b_vlm/adapter_load.py", "src/cgt/p3m_matched/registry.py"]  # the 4B load code too (p3c-v2-4b)


def code_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def run_key(film: str, adapter_sha: str, sha: str, method: str, budget_px: int = BUDGET_PX, model_identity: dict | None = None) -> str:
    d = {"film": film, "adapter": adapter_sha, "code": sha, "method": method, "budget_px": budget_px}
    if model_identity is not None:  # 4B (p3c-v2-4b); the 2B blob, hence its key, is unchanged
        d["model_identity"] = model_identity
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def require_public_domain(film: str, allow_owned: bool = False) -> dict:
    """An unregistered film is always refused. A registered film that is not public domain (owned, local only) is refused unless `allow_owned` is passed explicitly
   : that is LOCAL INFERENCE ONLY, its outputs stay under the gitignored data/ tree and are never published.
    The film is resolved in the merged registry. Every overlay film is
    local only: it needs `allow_owned` whatever its public_domain flag says. A films.yaml film returns its films.yaml entry unchanged."""
    films = registry.load_films(ROOT)
    if film not in films:
        raise guards.NotAuthorized(f"{film}: not registered in configs/films.yaml")
    e = films[film]
    if registry.is_overlay(e):
        if not allow_owned:
            raise guards.NotAuthorized(f"{film}: an ad-hoc film of the local overlay {registry.OVERLAY}; Phase 3C runs on it only with the explicit allow-owned flag (local inference, never published)")
        return e
    if not e.get("public_domain") and not allow_owned:
        raise guards.NotAuthorized(f"{film}: not a public-domain film in configs/films.yaml; Phase 3C runs on an owned film only with the explicit allow-owned flag (local inference, never published)")
    return {k: v for k, v in e.items() if k != "registry"}


def keyframe_root(film: str) -> Path:
    """The only place a film's keyframes may come from: <repo>/data/interim/<film>/keyframes."""
    return ROOT / "data/interim" / film / "keyframes"


def keyframe_path(film: str, rel: str) -> Path:
    """Shot-table keyframe path -> checked absolute path. Fail closed (DataRootViolation, exit 5) unless it lies inside the authorised film's
    keyframe root after lexical normalisation (rejects '..' traversal, other films, ShotBench, gold, dev, data/raw, absolute paths) and, after
    resolution, reaches it without a symlink component below the root. Never opens the file."""
    root = keyframe_root(film)
    if os.path.isabs(rel):  # a table written with absolute paths (another checkout of the same data tree): accept it only if it lies inside the RESOLVED keyframe root
        real = root.resolve()
        p = Path(os.path.normpath(rel))
        try:
            part = p.relative_to(real)
        except ValueError as e:
            raise guards.DataRootViolation(f"keyframe {rel!r} is outside {real}") from e
        return guards.require_under_image_root(part, real)
    p = Path(os.path.normpath(ROOT / rel))
    try:
        part = p.relative_to(os.path.normpath(root))
    except ValueError as e:
        raise guards.DataRootViolation(f"keyframe {rel!r} is outside {root.relative_to(ROOT)}") from e
    return guards.require_under_image_root(part, root)


def validate_shot_table(film: str, shots: pd.DataFrame) -> pd.DataFrame:
    """Check every keyframe path of the shot table (all three columns) BEFORE any image is opened or the model is built. Returns shots with resolved paths."""
    need = {"film_id", "shot_id", *KEYFRAMES}
    if need - set(shots.columns):
        raise ValueError(f"shot table lacks columns {sorted(need - set(shots.columns))}")
    if set(shots.film_id) != {film}:
        raise guards.DataRootViolation(f"shot table contains film ids {sorted(set(shots.film_id))}, authorised film is {film}")
    if shots.shot_id.duplicated().any():
        raise ValueError("duplicate shot_id in the shot table")
    out = shots.copy()
    for kf in KEYFRAMES:
        out[kf] = [str(keyframe_path(film, r)) for r in shots[kf]]
    return out


def _valid_record(r: object) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("shot_id"), int)
            and all(isinstance(r.get(a), list) and len(r[a]) == len(labels(a)) for a in ATTRS))


def repair_cache(path: Path) -> set[int]:
    """Make a scores_*.jsonl file safe to append to: keep the longest prefix of complete, parseable, valid records with distinct shot ids and
    truncate everything from the first torn/invalid/duplicate line on (an interrupted write leaves at most one such line). Returns the shot ids kept."""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    keep, ids = 0, set()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            r = json.loads(line)
        except ValueError:
            break
        if not _valid_record(r) or r["shot_id"] in ids:
            break
        ids.add(r["shot_id"])
        keep += len(line)
    if keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return ids


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def verify_coverage(run_dir: Path, shot_ids: list[int]) -> dict[str, int]:
    """Parse every scores file; raise unless each keyframe file has exactly one valid line for every shot id. Returns lines per keyframe."""
    want, counts = set(shot_ids), {}
    for kf in KEYFRAMES:
        recs = []
        for line in (run_dir / f"scores_{kf}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not _valid_record(r):
                raise ValueError(f"{kf}: invalid record")
            recs.append(r["shot_id"])
        if len(recs) != len(set(recs)) or set(recs) != want:
            raise ValueError(f"{kf}: coverage {len(set(recs))}/{len(want)} shots, {len(recs)} lines")
        counts[kf] = len(recs)
    return counts


def keyframe_paths(film: str, shots: pd.DataFrame) -> list[Path]:
    """Every keyframe file of the shot table (validated paths), in (shot_id, start/mid/end) order: the inputs the scoring run depends on."""
    v = validate_shot_table(film, shots).sort_values("shot_id")
    return [Path(p) for row in v.itertuples(index=False) for p in (row.kf_start, row.kf_mid, row.kf_end)]


def keyframe_inputs_digest(film: str, shots: pd.DataFrame, root: Path | None = None, write: bool = True) -> str:
    """Digest of the CONTENT of all keyframes (per-file SHA-256 cached against size+mtime in data/cache/p3c/keyframe_digests.json). Bound into the run manifest, so replaced
    or regenerated keyframes can never return predictions computed on other images."""
    from cgt.pipeline import identity
    return identity.digest_of_files(keyframe_paths(film, shots), (root or ROOT) / "data/cache/p3c/keyframe_digests.json", write)


def archive_stale_run(run_dir: Path, why: str, log=print) -> Path:
    """Move a run directory aside (never delete it) when its recorded input identity no longer matches; the caller then starts a fresh run in the original path."""
    dst = run_dir.with_name(f"{run_dir.name}.stale-{time.strftime('%Y%m%dT%H%M%S')}")
    run_dir.rename(dst)
    log(f"archived the previous run to {dst.name}: {why}")
    return dst


STOPPED = "stopped_at_deadline"


def run_film(film: str, method: str, adapter: Path, engine=None, limit: int | None = None, log=print, allow_owned: bool = False,
             model: str = "2b", stop_at: datetime | None = None, est_item_s: float | None = None,
             now: Callable[[], datetime] = lambda: datetime.now().astimezone()) -> Path:
    """Score every shot (or the first `limit` shots: a smoke test, recorded as `limited`, never as a finished film run).
    `model`: "2b" (library default = the old default, unchanged) or "4b" (p3c-v2-4b: pinned identity in the run key and manifest).
    `stop_at`: before each new (shot, keyframe) no new item starts if now + the last measured item time (`est_item_s` before the first)
    would pass it; the run then keeps its cache, is recorded as 'stopped_at_deadline' (never complete, no table) and resumes later."""
    model = models.resolve(model)
    entry = require_public_domain(film, allow_owned)
    owned_local_only = not entry.get("public_domain") or registry.is_overlay(entry)
    st_ = guards.require_clean_tree(CODE_PATHS)
    all_shots = validate_shot_table(film, pd.read_parquet(ROOT / f"data/processed/{film}_shots.parquet"))  # before any image access or model build
    shots = all_shots.iloc[:limit] if limit else all_shots
    eng = engine or models.make_engine(model, adapter)  # imported late: mlx is not needed for the CPU tests
    ident_m = models.model_identity(model)
    if ident_m is not None and eng.adapter_sha != ident_m["adapter_sha256"]:
        raise guards.AdapterHashError(f"engine adapter {eng.adapter_sha} is not the pinned {model} adapter {ident_m['adapter_sha256']}")
    sha = st_["commit"]
    key = run_key(film, eng.adapter_sha, sha, method, model_identity=ident_m)
    run_dir = ROOT / "data/cache/p3c/runs" / f"{film}_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_digest = keyframe_inputs_digest(film, all_shots, ROOT)
    ident = {"film": film, "key": key, "method": method, "adapter_sha256": eng.adapter_sha, "code_sha": sha, "budget_px": BUDGET_PX}
    if ident_m is not None:
        ident["model_identity"] = ident_m
    mf = run_dir / "manifest.json"
    if mf.exists():  # input identity of an existing run: a changed or unprovable keyframe set means the old scores must not be reused
        old = json.loads(mf.read_text())
        if old.get("inputs_digest") not in (None, inputs_digest):
            archive_stale_run(run_dir, "keyframe images changed since the run started", log)
            run_dir.mkdir(parents=True, exist_ok=True)
        elif old.get("inputs_digest") is None and old.get("state"):
            # a run that predates the content identity cannot prove which images it scored: keep it as evidence, never adopt it, rescore from the current keyframes
            archive_stale_run(run_dir, "legacy run without a keyframe content identity", log)
            run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**ident, "inputs_digest": inputs_digest, "multiple": MULTIPLE, "keyframes": list(KEYFRAMES), "vocab": {a: labels(a) for a in ATTRS},
                "n_shots": len(all_shots), "invocations": [], "owned_local_only": owned_local_only}
    if mf.exists():
        old = json.loads(mf.read_text())
        if {k: old.get(k) for k in ident} != ident:
            raise ValueError("run directory manifest does not match the request")
        manifest["invocations"] = old.get("invocations", [])
        manifest["started"] = old.get("started")
    manifest.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    inv = {"start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "limit": limit, "state": "running"}
    manifest["invocations"].append(inv)
    manifest["state"] = "running"
    manifest.pop("finished", None)
    _atomic_json(mf, manifest)  # started/running state is on disk before the first pass
    img_root = ROOT / "data/cache/p3c/images_film" / film / inputs_digest[:16]  # content-addressed: a replaced keyframe can never reuse an old shrunk copy
    t0, p0 = time.perf_counter(), eng.passes
    stopped, last = False, est_item_s
    for kf in KEYFRAMES:
        if stopped:
            break
        out = run_dir / f"scores_{kf}.jsonl"
        done = repair_cache(out)
        todo = shots[~shots.shot_id.isin(done)]
        log(f"{kf}: {len(done)} done, {len(todo)} to do")
        with open(out, "a") as f:
            for i, row in enumerate(todo.itertuples(index=False)):
                if stop_at is not None and now() + timedelta(seconds=last or 0.0) > stop_at:
                    stopped = True
                    log(f"{kf}: no new item: now + {last or 0:.1f}s would pass the stop time {stop_at.isoformat()}")
                    break
                ti = time.perf_counter()
                src = getattr(row, kf)  # already validated and resolved
                guards.require_images_readable([src])
                state = eng.image_state(shrunk_path(src, img_root))
                rec = {"shot_id": int(row.shot_id)}
                for a in ATTRS:
                    s, _ = score_attribute(eng, state, a, method)
                    rec[a] = [float(x) for x in s]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                last = time.perf_counter() - ti
                if (i + 1) % 25 == 0:
                    log(f"{kf} {i + 1}/{len(todo)} shots, {time.perf_counter() - t0:.0f}s elapsed, passes {eng.passes}")
    inv.update({"end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": time.perf_counter() - t0, "forward_passes": eng.passes - p0})
    if stopped:
        inv["state"] = manifest["state"] = STOPPED  # cache kept, resumable; never complete, so no table can be built from it
    elif limit and limit < len(all_shots):
        inv["state"] = manifest["state"] = "limited_complete"  # smoke test: NOT a finished film run
    else:
        manifest["coverage"] = verify_coverage(run_dir, [int(x) for x in all_shots.shot_id])  # raises (state stays 'running') on any gap
        inv["state"] = manifest["state"] = "complete"
        manifest["finished"] = inv["end"]
    _atomic_json(mf, manifest)
    return run_dir


def build_table(film: str, run_dir: Path, alphas: dict[str, float], out_path: Path) -> pd.DataFrame:
    """Join the cached scores to the shot table on shot_id (docs/phase3c_interface.md)."""
    man = json.loads((run_dir / "manifest.json").read_text())
    shots = pd.read_parquet(ROOT / f"data/processed/{film}_shots.parquet")[["film_id", "shot_id"]]
    if "state" in man and man["state"] != "complete":
        raise ValueError(f"run {man['key']} is '{man['state']}', not a complete film run")
    verify_coverage(run_dir, [int(x) for x in shots.shot_id])
    recs = {kf: {r["shot_id"]: r for r in map(json.loads, (run_dir / f"scores_{kf}.jsonl").read_text().splitlines())} for kf in KEYFRAMES}
    cols: dict[str, list] = {c: [] for c in ["film_id", "shot_id"]}
    for sid, fid in zip(shots.shot_id, shots.film_id):
        cols["film_id"].append(fid)
        cols["shot_id"].append(int(sid))
        for a in ATTRS:
            labs = labels(a)
            per = {kf: (np.array(recs[kf][int(sid)][a]) if int(sid) in recs[kf] else None) for kf in KEYFRAMES}
            mid = per["kf_mid"]
            prim = {kf: (labs[int(np.argmax(v))] if v is not None else None) for kf, v in per.items()}
            p1, mg = ag.top_and_margin(mid) if mid is not None else (None, None)
            row = {f"{a}_pred": prim["kf_mid"], f"{a}_p1": p1, f"{a}_margin": mg,
                   f"{a}_scores": [float(x) for x in mid] if mid is not None else None,
                   f"{a}_pred_start": prim["kf_start"], f"{a}_pred_end": prim["kf_end"],
                   f"{a}_consistent": (None if None in prim.values() else len(set(prim.values())) == 1)}
            if a in MULTI:
                sets = {kf: ([labs[i] for i in ag.label_set(v, alphas[a])] if v is not None else None) for kf, v in per.items()}
                row[f"{a}_set"] = sets["kf_mid"]
                if None in sets.values():
                    row[f"{a}_set_jaccard"] = None
                else:
                    ss = [set(x) for x in sets.values()]
                    row[f"{a}_set_jaccard"] = float(np.mean([len(x & y) / len(x | y) for x, y in ((ss[0], ss[1]), (ss[0], ss[2]), (ss[1], ss[2]))]))
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    df = pd.DataFrame(cols)
    df["p3c_run_key"] = man["key"]
    df["p3c_method"] = man["method"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
