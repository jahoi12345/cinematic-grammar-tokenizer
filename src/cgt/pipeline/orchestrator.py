# ruff: noqa: ISC004
"""One-command film pipeline: stages p1 (shots+colour), p2 (pretrained models), p3c (frozen VLM scoring, GPU), tokens (v1 rows), move (GPU, when available).
Each stage reports a cache key and hit/miss, checks its own prerequisites, runs idempotently, and writes a manifest; one summary per film is written at the end.
Exit codes: see EXIT. Nothing here trains or tunes; a film is an inference target only."""
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cgt.pipeline import registry as films_registry
from cgt.pipeline.gpulock import GpuLock, LockHeld

ROOT = Path(__file__).resolve().parents[3]
ORDER = ["p1", "p2", "p3c", "move", "tokens"]  # move BEFORE tokens: the token rows read the movement table
EXIT = {"ok": 0, "usage": 2, "gpu_lock_held": 3, "film_not_allowed": 4, "unavailable_or_prerequisite": 5, "stage_failed": 6, "adapter_hash": 7, "dirty_tree": 8, "memory_guard": 9,
        "bad_input": 10, "registry_clash": 11}


class PipelineError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


@dataclass
class Status:
    hit: bool
    key: str
    detail: str = ""
    outputs: list[Path] = field(default_factory=list)
    counts: dict = field(default_factory=dict)             # shown in the summary (from the existing cache when it is a hit)
    original_seconds: float | None = None                  # compute time recorded by the run that produced the cache, if known


@dataclass
class Stage:
    name: str
    gpu: bool
    status: Callable[["Ctx"], Status]
    run: Callable[["Ctx"], dict]
    prerequisites: Callable[["Ctx"], list[tuple[str, str]]] = lambda ctx: []   # [(exit code name, message)]
    available: Callable[["Ctx"], str | None] = lambda ctx: None                 # reason if the stage cannot run at all
    reference: str = ""                                                         # expected runtime note (docs/film_onboarding.md)
    needs_gpu: Callable[["Ctx"], bool] | None = None                            # overrides `gpu` per call (e.g. a table rebuild from a valid cache needs no GPU)
    on_hit: Callable[["Ctx"], None] | None = None                               # persist an input-identity attestation when a legacy cache is adopted (never in a dry run)
    uses_gpu: Callable[["Ctx"], bool] | None = None                             # LOCK ONLY: the stage's run uses the GPU (e.g. p2 on MPS) though `gpu` reports it as CPU


@dataclass
class Ctx:
    film_id: str
    film: dict
    root: Path = ROOT
    allow_owned: bool = False
    log: Callable[[str], None] = print
    dry_run: bool = False                     # identity helpers must not write cache files in a dry run
    provisional: str | None = None            # reason: every output of this run is provisional (written to manifests and the summary)
    p3c_model: str | None = None              # Phase 3C model: None = the pipeline default (p3c-v2-4b, Step C 4B); "2b" = the old default

    @property
    def manifest_dir(self) -> Path:
        return self.root / "data/cache/pipeline" / self.film_id


def film_entry(film_id: str, root: Path = ROOT, pending: dict | None = None) -> dict:
    """The entry from the merged registry (configs/films.yaml + the gitignored overlay configs/films.local.yaml, the recorded rule; an id clash is an error).
    `pending` is an overlay entry not yet written (a dry run plans it without writing the overlay)."""
    try:
        films = films_registry.load_films(root, pending)
    except films_registry.RegistryClash as e:
        raise PipelineError("registry_clash", str(e)) from e
    if film_id in films:
        return films[film_id]
    raise PipelineError("film_not_allowed", f"film '{film_id}' is not registered in configs/films.yaml (or the local overlay {films_registry.OVERLAY})")


def check_film(film_id: str, stages: list[Stage], allow_owned: bool, root: Path = ROOT, publish: bool = False, pending: dict | None = None) -> dict:
    """Fail closed. The film must be registered. Non-public-domain (owned) films are refused unless `allow_owned` is given (a recorded decision: local inference only, outputs
    gitignored, never published); `publish` requires public_domain regardless and is always refused for an overlay (ad-hoc) film."""
    film = film_entry(film_id, root, pending)
    pd_ok = bool(film.get("public_domain"))
    if publish and films_registry.is_overlay(film):
        raise PipelineError("film_not_allowed", f"film '{film_id}' is an ad-hoc input from the local overlay {films_registry.OVERLAY}: nothing derived from it may be published")
    if publish and not pd_ok:
        raise PipelineError("film_not_allowed", f"film '{film_id}' is not public domain: nothing derived from it may be published")
    if not pd_ok and not allow_owned:
        raise PipelineError("film_not_allowed", f"film '{film_id}' is not public domain (owned, local only): it is processed only when explicitly asked (--allow-owned)")
    # with --allow-owned every stage (including GPU inference) may run, LOCALLY: outputs stay under the gitignored data/ tree and no results/ file is written for the film
    return film


def plan(ctx: Ctx, stages: list[Stage], explicit: bool) -> list[dict]:
    """Per requested stage: availability, prerequisites (as if earlier planned stages had run), status. Pure inspection; computes nothing."""
    out, will_run = [], set()
    for st in stages:
        why = st.available(ctx)
        if why:
            out.append({"stage": st, "state": "unavailable", "reason": why, "status": None, "problems": []})
            continue
        s = st.status(ctx)
        # prerequisites matter only for a stage that has to run; a cache hit needs nothing else
        problems = [] if s.hit else [(c, m) for c, m in st.prerequisites(ctx) if not _satisfied_by_plan(m, will_run)]
        out.append({"stage": st, "state": "hit" if s.hit else "miss", "reason": s.detail, "status": s, "problems": problems})
        if not s.hit:
            will_run.add(st.name)
    return out


def _satisfied_by_plan(message: str, will_run: set[str]) -> bool:
    """A prerequisite message of the form 'needs:<stage>' is satisfied when that stage is part of this run and will run first."""
    return message.startswith("needs:") and message.split(":", 1)[1].split()[0] in will_run


def _code_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_manifest(ctx: Ctx, name: str, rec: dict) -> None:
    ctx.manifest_dir.mkdir(parents=True, exist_ok=True)
    p = ctx.manifest_dir / f"{name}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(p)


def _read_manifest(ctx: Ctx, name: str) -> dict:
    p = ctx.manifest_dir / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def format_plan(ctx: Ctx, rows: list[dict]) -> str:
    if not ctx.film.get("public_domain"):
        note = "; OWNED: LOCAL INFERENCE ONLY, nothing is published"
    elif films_registry.is_overlay(ctx.film):
        note = f"; LOCAL ONLY: overlay film ({films_registry.OVERLAY}), nothing is published"
    else:
        note = ""
    L = [f"film {ctx.film_id} (public_domain={bool(ctx.film.get('public_domain'))}{note}); stages in order:"]
    for r in rows:
        st: Stage = r["stage"]
        if r["state"] == "unavailable":
            L.append(f"  {st.name:7s} {'GPU' if st.gpu else 'CPU'}  UNAVAILABLE: {r['reason']}")
            continue
        s: Status = r["status"]
        verdict = "cache HIT (nothing to compute)" if s.hit else "MISS: would RUN"
        L.append(f"  {st.name:7s} {'GPU' if st.gpu else 'CPU'}  {verdict}  key={s.key[:12]}  {r['reason']}")
        for code, msg in r["problems"]:
            L.append(f"          prerequisite problem [{code}]: {msg}")
        if not s.hit and st.reference:
            L.append(f"          reference runtime: {st.reference}")
    return "\n".join(L)


def locks_gpu(st: Stage, ctx: Ctx) -> bool:
    """Whether running this stage now uses the GPU, so it must hold the GPU lock while it runs. Never changes what the stage computes."""
    if st.uses_gpu is not None:
        return bool(st.uses_gpu(ctx))
    if st.needs_gpu is not None:
        return bool(st.needs_gpu(ctx))
    return st.gpu


def lock_plan(ctx: Ctx, active: list[dict]) -> list[str]:
    """Stages of this run that will (or may, because an earlier stage of the run recomputes their inputs) use the GPU and so hold the lock while they run."""
    out, upstream_runs = [], False
    for r in active:
        st: Stage = r["stage"]
        if (r["state"] == "miss" or upstream_runs) and locks_gpu(st, ctx):  # cheap test first: a cache hit never evaluates locks_gpu (p2 would import torch)
            out.append(st.name)
        if r["state"] == "miss":
            upstream_runs = True
    return out


def _lock_plan_text(names: list[str]) -> str:
    if not names:
        return "GPU lock: no stage of this run will take it"
    return (f"GPU lock: held only while {', '.join(names)} run{'s' if len(names) == 1 else ''} (taken just before each GPU stage, released right after it); "
            "every other stage runs without it")


def run(film_id: str, requested: list[str] | None, registry: dict[str, Stage], dry_run: bool = False, allow_owned: bool = False, publish: bool = False,
        root: Path = ROOT, lock_path: Path | None = None, log: Callable[[str], None] = print, provisional: str | None = None,
        pending: dict | None = None, p3c_model: str | None = None) -> int:
    """Run (or, with dry_run, only plan) the stages. Returns the exit code. `pending`: an unwritten overlay entry, accepted only in a dry run."""
    explicit = requested is not None
    names = [n for n in ORDER if n in registry and (n in requested if explicit else True)]
    unknown = [n for n in (requested or []) if n not in registry]
    if unknown:
        log(f"unknown stage(s): {unknown}; known: {list(registry)}")
        return EXIT["usage"]
    stages = [registry[n] for n in names]
    try:
        if pending is not None and not dry_run:
            raise PipelineError("usage", "a pending (unwritten) registry entry may only be planned (dry run); register it first")
        film = check_film(film_id, stages, allow_owned, root, publish, pending)
    except PipelineError as e:
        log(f"REFUSED [{e.code}]: {e}")
        return EXIT[e.code]
    ctx = Ctx(film_id, film, root, allow_owned, log, dry_run, provisional, p3c_model)
    rows = plan(ctx, stages, explicit)
    # stages that are unavailable are skipped unless the user asked for them explicitly
    for r in rows:
        if r["state"] == "unavailable" and explicit:
            log(format_plan(ctx, rows))
            log(f"REFUSED [unavailable_or_prerequisite]: stage '{r['stage'].name}' was requested but is unavailable: {r['reason']}")
            return EXIT["unavailable_or_prerequisite"]
    active = [r for r in rows if r["state"] != "unavailable"]
    log(format_plan(ctx, rows))
    bad = [(r["stage"].name, c, m) for r in active for c, m in r["problems"]]
    if bad:
        for n, c, m in bad:
            log(f"PREREQUISITE FAILED in {n} [{c}]: {m}")
        return EXIT[bad[0][1]] if bad[0][1] in EXIT else EXIT["unavailable_or_prerequisite"]
    gpu_stages = lock_plan(ctx, active)
    log(_lock_plan_text(gpu_stages))
    if dry_run:
        log("dry run: nothing was computed, no lock was taken, no file was written")
        return EXIT["ok"]

    # fail fast: a GPU stage of this run would find the lock held, so nothing is started (the probe does not keep the lock; it is taken per GPU stage below)
    lock = GpuLock(f"run_film:{film_id}", lock_path)
    if gpu_stages:
        try:
            lock.acquire()
        except LockHeld as e:
            log(f"REFUSED [gpu_lock_held]: {e}. Nothing was started.")
            return EXIT["gpu_lock_held"]
        lock.release()
    results, code = [], EXIT["ok"]
    t_start = time.time()
    try:
        for r in active:
            st: Stage = r["stage"]
            s = st.status(ctx)  # re-evaluate: an earlier stage of this run may have changed the inputs
            rec = {"stage": st.name, "film": film_id, "provisional": ctx.provisional, "gpu": st.gpu, "key": s.key, "code_sha": _code_sha(root), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            prev = _read_manifest(ctx, st.name)
            if s.hit:
                if st.on_hit:
                    st.on_hit(ctx)
                rec.update({"status": "hit", "cache_hit": True, "seconds": 0.0, "original_seconds": s.original_seconds if s.original_seconds is not None else prev.get("original_seconds"),
                            "counts": s.counts or prev.get("counts", {}), "detail": s.detail})
                results.append(rec)
                _write_manifest(ctx, st.name, {**prev, **rec} if prev.get("status") == "ran" else rec)
                log(f"[{st.name}] cache hit")
                continue
            probs = st.prerequisites(ctx)
            if probs:
                rec.update({"status": "prerequisite_failed", "detail": "; ".join(m for _, m in probs)})
                results.append(rec)
                _write_manifest(ctx, st.name, rec)
                code = EXIT.get(probs[0][0], EXIT["unavailable_or_prerequisite"])
                log(f"[{st.name}] prerequisite failed: {probs[0][1]}")
                break
            gpu_now = locks_gpu(st, ctx)
            if gpu_now:
                try:
                    lock.acquire()
                except LockHeld as e:
                    rec.update({"status": "gpu_lock_held", "detail": str(e)})
                    results.append(rec)
                    _write_manifest(ctx, st.name, rec)
                    code = EXIT["gpu_lock_held"]
                    log(f"STOPPED [gpu_lock_held] before [{st.name}]: {e}. The stages completed so far are cached; re-run the same command when the lock is free "
                        "to resume from here.")
                    break
                log(f"GPU lock acquired for [{st.name}] ({lock.path})")
            rec.update({"status": "running", "cache_hit": False})
            _write_manifest(ctx, st.name, rec)  # started/running state is on disk before any work
            t0 = time.time()
            try:
                try:
                    counts = st.run(ctx) or {}
                finally:
                    if gpu_now:
                        lock.release()  # right after the GPU stage, success or failure; the output check below is CPU work
            except PipelineError as e:
                rec.update({"status": "failed", "error": str(e), "seconds": time.time() - t0})
                code = EXIT.get(e.code, EXIT["stage_failed"])
            except Exception as e:  # noqa: BLE001 - recorded in the manifest and in the exit code, never swallowed
                rec.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0})
                code = EXIT["stage_failed"]
            else:
                after = st.status(ctx)
                if not after.hit:
                    rec.update({"status": "failed", "error": f"stage finished but its outputs are not valid ({after.detail})", "seconds": time.time() - t0})
                    code = EXIT["stage_failed"]
                else:
                    rec.update({"status": "ran", "counts": counts, "seconds": time.time() - t0, "original_seconds": time.time() - t0, "key": after.key,
                                "outputs": [str(p) for p in after.outputs]})
            rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            results.append(rec)
            _write_manifest(ctx, st.name, rec)
            log(f"[{st.name}] {rec['status']} in {rec.get('seconds', 0):.0f}s")
            if rec["status"] != "ran":
                break
    finally:
        lock.release()
    write_summary(ctx, rows, results, time.time() - t_start, code)
    return code


def _film_label(film: dict, film_id: str) -> str:
    """'<title or id> (<year>)', or just '<title or id>' when the entry has no year (overlay entries)."""
    name = film.get("title") or film_id
    year = film.get("year")
    return f"{name} ({year})" if year not in (None, "") else str(name)


def write_summary(ctx: Ctx, rows: list[dict], results: list[dict], seconds: float, code: int) -> Path:
    """results/films/<film>.md: one table per film (aggregate counts and runtimes only; no frames, no labels)."""
    by = {r["stage"]: r for r in results}
    L = [f"# Pipeline summary: {ctx.film_id}", "",
         f"Run finished {time.strftime('%Y-%m-%d %H:%M:%S%z')}; wall time {seconds:.0f}s; exit code {code}; code `{_code_sha(ctx.root)[:10]}`. Film: {_film_label(ctx.film, ctx.film_id)}, "
         f"public_domain={bool(ctx.film.get('public_domain'))}. Outputs of model stages are predictions; accuracy on the film is unknown unless labels exist.", "",
         "| stage | resource | result | cache key | seconds this run | original compute seconds | counts |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        st: Stage = r["stage"]
        if r["state"] == "unavailable":
            L.append(f"| {st.name} | {'GPU' if st.gpu else 'CPU'} | unavailable ({r['reason']}) | | | | |")
            continue
        x = by.get(st.name)
        if x is None:
            L.append(f"| {st.name} | {'GPU' if st.gpu else 'CPU'} | not reached | {r['status'].key[:12]} | | | |")
            continue
        orig = x.get("original_seconds")
        cnt = ", ".join(f"{k} {v}" for k, v in (x.get("counts") or {}).items())
        L.append(f"| {st.name} | {'GPU' if st.gpu else 'CPU'} | {x['status']}{' (cache hit)' if x.get('cache_hit') else ''} | {x['key'][:12]} | {x.get('seconds', 0):.0f} | "
                 f"{'' if orig is None else f'{orig:.0f}'} | {cnt} |")
    hits = sum(1 for x in results if x.get("cache_hit"))
    L += ["", f"Cache hits: {hits} of {len(results)} stages executed in this invocation. Per-stage manifests: `data/cache/pipeline/{ctx.film_id}/<stage>.json` (not committed)."]
    if ctx.provisional:
        L.insert(2, f"**PROVISIONAL: every output of this run is provisional. Reason: {ctx.provisional}**\n")
    local = not ctx.film.get("public_domain") or films_registry.is_overlay(ctx.film)
    if local:  # owned or ad-hoc (overlay) film: the summary stays in the gitignored cache, never under results/
        out = ctx.manifest_dir
        kind = "OWNED FILM" if not ctx.film.get("public_domain") else "AD-HOC FILM (local overlay, the recorded rule)"
        L.insert(2, f"**{kind}: LOCAL ONLY. Do not commit or publish this file or anything derived from the film.**\n")
    else:
        out = ctx.root / "results/films"
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("summary.md" if local else f"{ctx.film_id}.md")
    p.write_text("\n".join(L) + "\n")
    return p
