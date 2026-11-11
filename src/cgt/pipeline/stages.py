"""The real stage definitions (p1, p2, p3c, tokens, move) over the existing phase code. Heavy imports are inside the functions so `--dry-run` stays light."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from cgt.p3c_tokens import models as p3c_models
from cgt.pipeline import identity
from cgt.pipeline import registry as films_registry
from cgt.pipeline.orchestrator import PipelineError, Stage, Status

SCORING_FILES = p3c_models.SCORING_FILES_2B   # the 2B's; a model's own list: p3c_models.scoring_files(model)
ADAPTER_REL = p3c_models.ADAPTER_2B_REL


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _pdf(root: Path, rel: str):
    import pandas as pd
    return pd.read_parquet(root / rel)


# ---------------------------------------------------------------- shared identity helpers
P1_CODE = ["src/cgt/p1_shots_color/pipeline.py", "src/cgt/p1_shots_color/color.py", "src/cgt/p1_shots_color/detection.py", "src/cgt/p1_shots_color/keyframes.py"]


def _hash_cache(ctx) -> Path:
    return ctx.root / "data/cache/pipeline" / ctx.film_id / "hash_cache.json"


def _video_path(ctx) -> Path:
    return ctx.root / ctx.film.get("path", "")


def _video_sha(ctx) -> str | None:
    v = _video_path(ctx)
    return identity.cached_sha256(v, _hash_cache(ctx), write=not ctx.dry_run)[0] if v.exists() else None


def _p1_code_sha(ctx) -> str:
    h = hashlib.sha256()
    for rel in P1_CODE:
        f = ctx.root / rel
        h.update(f.read_bytes() if f.exists() else b"absent")
    return h.hexdigest()


# ---------------------------------------------------------------- p1
def _p1_threshold(ctx) -> float:
    return float(ctx.film.get("cut_threshold", 27.0))


def _p1_record_path(ctx) -> Path:
    return ctx.root / "data/interim" / ctx.film_id / "p1_cache.json"


def _p1_threshold_source(ctx) -> str:
    """F-C2: where the threshold really comes from (wording only; the cache key binds the value)."""
    if ctx.film.get("cut_threshold") is None:
        return "the default (no cut_threshold registered)"
    if films_registry.is_overlay(ctx.film):
        return f"{films_registry.OVERLAY} (overlay, {ctx.film.get('cut_threshold_source', 'source unrecorded')})"
    return films_registry.MAIN


def p1_status(ctx) -> Status:
    """HIT only with an identity record WRITTEN BY A PHASE 1 RUN (`built_by == "p1_run"`) that binds the registered video SHA-256 (size + mtime cache, so a replaced file is re-hashed), the
    threshold, the cuts, the shot-table bytes and the digest of every keyframe image, all equal to the current ones. A cache without such a record (a legacy cache) is never adopted: timestamps,
    ids, names and equal re-derived cuts prove nothing about content, so `p1_run` always rebuilds cuts, table and keyframes from the video."""
    thr = _p1_threshold(ctx)
    cuts = ctx.root / "data/interim" / ctx.film_id / f"cuts_t{thr:g}.json"
    table = ctx.root / f"data/processed/{ctx.film_id}_shots.parquet"
    if not cuts.exists() or not table.exists():
        return Status(False, f"{ctx.film_id}:t{thr:g}:none", f"threshold {thr:g} from {_p1_threshold_source(ctx)}; cuts cache {'present' if cuts.exists() else 'missing'}, shot table {'present' if table.exists() else 'missing'}")
    video = _video_path(ctx)
    if not video.exists():
        return Status(False, f"{ctx.film_id}:t{thr:g}:novideo", f"registered source video {video} is missing: the cache identity cannot be established")
    vsha = _video_sha(ctx)
    n_cuts, df = len(json.loads(cuts.read_text())), _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")
    key = f"{ctx.film_id}:t{thr:g}:{_sha_file(cuts)[:12]}:v{vsha[:12]}"
    rec = _p1_record_path(ctx)
    r = json.loads(rec.read_text()) if rec.exists() else {}
    if r.get("built_by") != "p1_run":
        return Status(False, key, "legacy cache: no record written by a Phase 1 run that binds the video, the shot table and the keyframe bytes. Equal cuts, ids, names or timestamps are not evidence, so the outputs are REBUILT")
    if r.get("video_sha256") != vsha or r.get("threshold") != thr or r.get("cuts_sha256") != _sha_file(cuts):
        return Status(False, key, "the source video, threshold or cuts differ from the identity recorded when the shot table was built: rerun")
    if r.get("table_sha256") != _sha_file(table) or r.get("keyframes_digest") != _p1_keyframes_digest(ctx, df):
        return Status(False, key, "the shot table or the keyframe images differ from the bytes recorded when Phase 1 built them: rebuild")
    how = "recorded identity matches (video, cuts, table, keyframe bytes)"
    frames_ok = all((ctx.root / p).exists() for col in ("kf_start", "kf_mid", "kf_end") for p in df[col].dropna())
    if len(df) != n_cuts or not frames_ok:
        return Status(False, key, f"shot table has {len(df)} rows for {n_cuts} cuts (or keyframes are missing): rerun")
    return Status(True, key, f"threshold {thr:g}; {n_cuts} shots, keyframes present; {how}", [table, cuts], {"shots": len(df), "threshold": thr})


def _p1_keyframes_digest(ctx, df) -> str:
    paths = [ctx.root / p for row in df.sort_values("shot_id").itertuples(index=False) for p in (row.kf_start, row.kf_mid, row.kf_end) if isinstance(p, str)]
    return identity.digest_of_files(paths, _hash_cache(ctx), write=not ctx.dry_run)


def _p1_write_record(ctx) -> None:
    """Written ONLY at the end of a Phase 1 run that just produced the cuts, the shot table and the keyframes from the registered video: it binds the video SHA-256, the cuts, the
    table bytes and the digest of every keyframe image."""
    thr = _p1_threshold(ctx)
    cuts = ctx.root / "data/interim" / ctx.film_id / f"cuts_t{thr:g}.json"
    table = ctx.root / f"data/processed/{ctx.film_id}_shots.parquet"
    df = _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")
    _p1_record_path(ctx).write_text(json.dumps({
        "built_by": "p1_run", "video_sha256": _video_sha(ctx), "video_size": _video_path(ctx).stat().st_size, "threshold": thr, "cuts_sha256": _sha_file(cuts),
        "table_sha256": _sha_file(table), "keyframes_digest": _p1_keyframes_digest(ctx, df), "code_sha256": _p1_code_sha(ctx), "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))


def p1_run(ctx) -> dict:
    """Always a full rebuild: the cuts cache is removed and Phase 1 re-extracts cuts, shot table and keyframes from the registered video, then the identity record is written.
    An existing cache is never adopted or 'verified' by comparing cuts: equal cuts cannot prove that its keyframes and tables came from this video and code."""
    from cgt.p1_shots_color.pipeline import run_film
    thr = _p1_threshold(ctx)
    cuts = ctx.root / "data/interim" / ctx.film_id / f"cuts_t{thr:g}.json"
    cuts.unlink(missing_ok=True)
    df = run_film(ctx.film_id, thr)
    _p1_write_record(ctx)
    return {"shots": len(df), "threshold": thr}


def p1_prereq(ctx) -> list:
    video = ctx.root / ctx.film.get("path", "")
    return [] if video.exists() else [("unavailable_or_prerequisite", f"video file missing: {video} (download and register the film first)")]


# ---------------------------------------------------------------- p2
def _p2_inputs(ctx, shots) -> dict:
    """Content identity of what Phase 2 reads: the mid keyframes (blocking, depth) and the source video (audio)."""
    paths = [ctx.root / p for p in shots.sort_values("shot_id").kf_mid]
    return {"kf_mid_digest": identity.digest_of_files(paths, _hash_cache(ctx), write=not ctx.dry_run), "video_sha256": _video_sha(ctx)}


def _p2_record_path(ctx) -> Path:
    return ctx.manifest_dir / "p2.inputs.json"


def p2_status(ctx) -> Status:
    from cgt.p2_pretrained import pipeline as p2
    table = ctx.root / f"data/processed/{ctx.film_id}_shots.parquet"
    out, meta = ctx.root / f"data/processed/{ctx.film_id}_p2.parquet", ctx.root / f"data/processed/{ctx.film_id}_p2.meta.json"
    if not table.exists():
        return Status(False, "p2:none", "no shot table yet (p1 first)")
    shots = _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")
    stages = [s for s in p2.STAGES if s != "subtitles" or ctx.film.get("subtitles")]
    keys = {s: p2.stage_key(ctx.film_id, s, shots, ctx.film) for s in stages}
    inputs = _p2_inputs(ctx, shots)
    key = hashlib.sha256(json.dumps([keys, inputs], sort_keys=True).encode()).hexdigest()
    if not out.exists():
        return Status(False, key, "no Phase 2 table")
    prev, m = _pdf(ctx.root, f"data/processed/{ctx.film_id}_p2.parquet"), (json.loads(meta.read_text()) if meta.exists() else {})
    ids = set(shots.shot_id)
    missing = [s for s in stages if not (all(c in prev.columns for c in p2.MARKERS[s]) and ids <= set(prev.shot_id) and m.get(s) == keys[s])]
    if missing:
        why = "no key metadata file (the table predates it)" if not meta.exists() else "stale or incomplete"
        return Status(False, key, f"stages to compute: {', '.join(missing)} ({why})")
    rec = _p2_record_path(ctx)
    if not rec.exists():
        return Status(False, key, "legacy Phase 2 table: its input identity (mid keyframes, source video) was never recorded; timestamps are not evidence, so it is recomputed")
    r2 = json.loads(rec.read_text())
    if {k: r2.get(k) for k in inputs} != inputs:
        return Status(False, key, "the mid keyframes or the source video changed since Phase 2 ran: recompute")
    if r2.get("table_sha256") != _sha_file(out):
        return Status(False, key, "the Phase 2 table differs from the bytes recorded when it was computed: recompute")
    how = "recorded input identity matches"
    return Status(True, key, f"{len(stages)} model stages cached; {how}", [out], {"rows": len(prev), "columns": prev.shape[1]})


def _p2_write_record(ctx) -> None:
    shots = _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")
    ctx.manifest_dir.mkdir(parents=True, exist_ok=True)
    _p2_record_path(ctx).write_text(json.dumps({**_p2_inputs(ctx, shots), "table_sha256": _sha_file(ctx.root / f"data/processed/{ctx.film_id}_p2.parquet")}))


def p2_run(ctx) -> dict:
    from cgt.p2_pretrained.pipeline import STAGES, run_film
    shots = _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")
    rec = _p2_record_path(ctx)
    # recompute everything (force) unless a record written by a previous run of THIS rule binds the inputs and the table bytes: an older or missing record cannot vouch for the table
    cur = _p2_inputs(ctx, shots)
    r = json.loads(rec.read_text()) if rec.exists() else {}
    changed = not r or r.get("table_sha256") is None or {k: r.get(k) for k in cur} != cur
    df = run_film(ctx.film_id, tuple(s for s in STAGES if s != "subtitles" or ctx.film.get("subtitles")), force=changed)
    _p2_write_record(ctx)
    return {"rows": len(df), "columns": df.shape[1]}


def p2_uses_gpu(ctx) -> bool:
    """For the GPU lock only: depth (Depth Anything V2) and audio (AudioSet classifier) run on MPS whenever torch reports it available
    (cgt.p2_pretrained.depth/audio pick device="mps" then; there is no switch to force CPU), so p2 is then a GPU stage. Changes nothing p2 computes."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def p2_prereq(ctx) -> list:
    return [] if (ctx.root / f"data/processed/{ctx.film_id}_shots.parquet").exists() else [("unavailable_or_prerequisite", "needs:p1 (no shot table)")]


# ---------------------------------------------------------------- p3c (GPU)
def _selected_method(ctx) -> str:
    return json.loads((ctx.root / "results/phase3c/method_selection.json").read_text())["rule"]["selected"]


def _model(ctx) -> str:
    """The Phase 3C model of this run: the pipeline default (p3c-v2-4b, Step C 4B) unless the run asked for --model 2b."""
    return p3c_models.resolve(getattr(ctx, "p3c_model", None))


def _frozen_sha(ctx) -> str:
    return p3c_models.expected_sha(_model(ctx), ctx.root)


def _git_equal(root: Path, old: str, files: list[str]) -> bool:
    r = subprocess.run(["git", "diff", "--quiet", old, "HEAD", "--", *files], cwd=root, capture_output=True, check=False)
    return r.returncode == 0


def _p3c_shots(ctx):
    return _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")


def _keyframe_digest(ctx) -> str:
    from cgt.p3c_tokens import runner
    old_root = runner.ROOT
    runner.ROOT = ctx.root  # keyframe_root / keyframe_path resolve against the context root
    try:
        return runner.keyframe_inputs_digest(ctx.film_id, _p3c_shots(ctx), ctx.root, write=not ctx.dry_run)
    finally:
        runner.ROOT = old_root


def p3c_valid_run(ctx) -> tuple[Path, dict, str] | tuple[None, None, str]:
    """A complete run of this film with the frozen adapter and the selected method whose scoring code is unchanged since its commit, whose recorded KEYFRAME CONTENT identity
    equals the current keyframes (a legacy run without one is never adopted), and whose score files re-parse with exactly one valid line
    per shot and keyframe. Returns (run dir, manifest, how) or (None, None, why not)."""
    from cgt.p3c_tokens import runner
    runs = ctx.root / "data/cache/p3c/runs"
    best, why = None, "no complete compatible run"
    shots = _p3c_shots(ctx) if (ctx.root / f"data/processed/{ctx.film_id}_shots.parquet").exists() else None
    for mf in sorted(runs.glob(f"{ctx.film_id}_*/manifest.json")) if runs.exists() else []:
        if ".stale-" in mf.parent.name:
            continue
        m = json.loads(mf.read_text())
        compatible = (m.get("state") == "complete" and m.get("adapter_sha256") == _frozen_sha(ctx) and m.get("method") == _selected_method(ctx)
                      and m.get("model_identity") == p3c_models.model_identity(_model(ctx))  # 2B: none recorded; 4B: the pinned identity
                      and m.get("budget_px") == 147456 and _git_equal(ctx.root, m["code_sha"], p3c_models.scoring_files(_model(ctx))))
        if not compatible or shots is None:
            continue
        if not m.get("inputs_digest"):
            why = f"run {m['key']}: legacy run without a keyframe content identity; it cannot prove which images it scored and is rescored, not adopted"
            continue
        if m["inputs_digest"] != _keyframe_digest(ctx):
            why = f"run {m['key']}: the keyframe images changed since it was scored (content digest differs)"
            continue
        how = "keyframe content identity matches"
        try:
            runner.verify_coverage(mf.parent, [int(x) for x in shots.shot_id])
        except (ValueError, OSError, KeyError) as e:
            why = f"run {m['key']}: score files do not cover every shot ({e})"
            continue
        if best is None or m.get("finished", "") > best[1].get("finished", ""):
            best = (mf.parent, m, how)
    return best if best else (None, None, why)


def p3c_status(ctx) -> Status:
    table = ctx.root / f"data/processed/{ctx.film_id}_shots.parquet"
    rd, m, how = p3c_valid_run(ctx)
    if rd is None:
        return Status(False, p3c_models.stage_key(ctx.film_id, _selected_method(ctx), _model(ctx)),
                      f"{how}; would score all shots x 3 keyframes with the {_model(ctx)} model ({p3c_models.PIPELINE_VERSION[_model(ctx)]})")
    out = ctx.root / f"data/processed/{ctx.film_id}_p3c.parquet"
    if not out.exists() or not table.exists():
        return Status(False, m["key"], f"valid run {m['key']} exists ({how}) but the table {out.name} is missing: rebuilt from the cache without the GPU")
    import pandas as pd
    t, s = pd.read_parquet(out, columns=["shot_id", "p3c_run_key"]), pd.read_parquet(table, columns=["shot_id"])
    if t.shot_id.duplicated().any() or len(t) != len(s) or set(t.shot_id) != set(s.shot_id) or set(t.p3c_run_key) != {m["key"]}:
        return Status(False, m["key"], "the Phase 3C table does not match the shot table or the valid run")
    secs = sum(i.get("seconds", 0) for i in m.get("invocations", []) if i.get("state") == "complete")
    passes = sum(i.get("forward_passes", 0) for i in m.get("invocations", []))
    return Status(True, m["key"], f"complete run {m['key']} (code {m['code_sha'][:10]}, scoring code unchanged since; {how})", [out, rd / "manifest.json"],
                  {"shots": len(t), "keyframes": 3 * len(t), "forward_passes": passes}, secs or None)


def p3c_needs_gpu(ctx) -> bool:
    """A missing table next to a valid cached run is rebuilt on the CPU; only a missing or invalid scoring run needs the GPU."""
    return p3c_valid_run(ctx)[0] is None


def p3c_prereq(ctx) -> list:
    from cgt.p3c_tokens import runner
    out = []
    model = _model(ctx)
    adapter = p3c_models.adapter_path(model, ctx.root)
    src = ("copy the frozen ckpt_100.safetensors from the Phase 3B run, read-only" if model == "2b"
           else "the Step C 4B ckpt_100.safetensors (the Step C working copy, or CGT_STEPA4B_ADAPTER)")
    if not adapter.exists():
        out.append(("unavailable_or_prerequisite", f"{model} adapter missing: {adapter} ({src})"))
    elif _sha_file(adapter) != _frozen_sha(ctx):
        out.append(("adapter_hash", f"{model} adapter SHA-256 {_sha_file(adapter)[:12]}... != pinned {_frozen_sha(ctx)[:12]}..."))
    if not (ctx.root / f"data/processed/{ctx.film_id}_shots.parquet").exists():
        out.append(("unavailable_or_prerequisite", "needs:p1 (no shot table)"))
    if not ctx.film.get("public_domain") and not ctx.allow_owned:
        out.append(("film_not_allowed", "Phase 3C runs on an owned film only with --allow-owned (local inference, never published)"))
    dirty = [ln for ln in subprocess.run(["git", "status", "--porcelain", "--", *runner.CODE_PATHS], cwd=ctx.root, capture_output=True, text=True, check=False).stdout.splitlines() if ln.strip()]
    if dirty:
        out.append(("dirty_tree", f"uncommitted changes under the Phase 3C code paths: {dirty[:3]} (the run key records the git SHA)"))
    return out


def p3c_run(ctx) -> dict:
    from cgt.p3c_tokens import runner
    method = _selected_method(ctx)
    sel = json.loads((ctx.root / "results/phase3c/method_selection.json").read_text())
    alphas = {k: v["alpha"] for k, v in sel["results"][method]["per_attribute"].items() if "alpha" in v}
    rd, _, _ = p3c_valid_run(ctx)
    model = _model(ctx)
    if rd is None and model == "2b":  # the old default call, unchanged
        rd = runner.run_film(ctx.film_id, method, ctx.root / ADAPTER_REL, log=ctx.log, allow_owned=ctx.allow_owned)
    elif rd is None:
        from cgt.p3c_tokens import rebuild
        ident = p3c_models.model_identity(model)
        key = runner.run_key(ctx.film_id, ident["adapter_sha256"], runner.code_sha(), method, model_identity=ident)
        try:  # never overwrite another run's live table without a _pre_4b backup (scripts/rebuild_default_4b.py does backup, verify, swap)
            rebuild.guard_live_replace(ctx.film_id, ctx.root, key)
        except rebuild.RebuildRefused as e:
            raise PipelineError("unavailable_or_prerequisite", str(e)) from e
        rd = runner.run_film(ctx.film_id, method, p3c_models.adapter_path(model, ctx.root), log=ctx.log, allow_owned=ctx.allow_owned, model=model)
    if model != "2b":
        from cgt.p3c_tokens import rebuild
        try:
            rebuild.guard_live_replace(ctx.film_id, ctx.root, json.loads((rd / "manifest.json").read_text())["key"])
        except rebuild.RebuildRefused as e:
            raise PipelineError("unavailable_or_prerequisite", str(e)) from e
    df = runner.build_table(ctx.film_id, rd, alphas, ctx.root / f"data/processed/{ctx.film_id}_p3c.parquet")
    man = json.loads((rd / "manifest.json").read_text())
    return {"shots": len(df), "keyframes": 3 * len(df), "forward_passes": sum(i.get("forward_passes", 0) for i in man.get("invocations", []))}


# ---------------------------------------------------------------- tokens
def tokens_status(ctx) -> Status:
    from cgt.tokens import build, build_v1
    need = [ctx.root / f"data/processed/{ctx.film_id}_{s}.parquet" for s in ("shots", "p2", "p3c")]
    if not all(p.exists() for p in need) or not (ctx.root / "data/cache/tokens/palette_global_k32.npz").exists():
        return Status(False, "tokens:none", "inputs or the global palette are missing")
    try:
        build_v1.check_complete(ctx.film_id, ctx.root)
        h = build_v1.input_hashes(ctx.film_id, ctx.root)
    except (build.IncompletePhase3C, OSError, KeyError) as e:
        return Status(False, "tokens:none", f"inputs not ready: {e}")
    key = hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
    if _tokens_need_move_check(ctx):  # rows built from an identity-less or stale movement table are not valid, whatever their hash says
        ok, why = move_table_verified(ctx)
        if not ok:
            return Status(False, key, f"the movement slot rests on an unverified movement table: {why}")
    meta = ctx.root / f"data/processed/{ctx.film_id}_tokens_v1.meta.json"
    out = ctx.root / f"data/processed/{ctx.film_id}_tokens_v1.parquet"
    txt = ctx.root / f"data/processed/{ctx.film_id}_tokens_v1.txt"
    if out.exists() and meta.exists() and json.loads(meta.read_text()).get("inputs") == h:
        import pandas as pd
        df = pd.read_parquet(out)
        if not txt.exists() or txt.read_text() != "\n".join(build.text_lines(df)) + "\n":
            return Status(False, key, "the token text file is missing or differs from the table: rebuilt")
        return Status(True, key, "token rows v1 up to date for the current inputs", [out, txt], {"rows": len(df), "all_keyframes_agree": int(df.kf_all_agree.sum())})
    return Status(False, key, "token table missing or inputs changed")


def _tokens_need_move_check(ctx) -> bool:
    """The tokens stage checks the movement table whenever one exists (whether or not the runner is in this tree: F-V5), and whenever the runner exists (tokens wait for the move stage)."""
    return _move_table(ctx).exists() or move_available(ctx) is None


def tokens_run(ctx) -> dict:
    from cgt.tokens import build_v1
    if _move_table(ctx).exists():  # build_v1 reads any table on disk: an unverified one is refused here, never consumed
        ok, why = move_table_verified(ctx)
        if not ok:
            raise PipelineError("unavailable_or_prerequisite", f"needs:move (the movement table is not verified: {why})")
    out, _, _ = build_v1.write_tokens_v1(ctx.film_id, ctx.root)
    import pandas as pd
    df = pd.read_parquet(out, columns=["kf_all_agree"])
    return {"rows": len(df), "all_keyframes_agree": int(df.kf_all_agree.sum())}


def tokens_prereq(ctx) -> list:
    out = []
    for s, stage in (("p3c", "p3c"), ("p2", "p2"), ("shots", "p1")):
        if not (ctx.root / f"data/processed/{ctx.film_id}_{s}.parquet").exists():
            out.append(("unavailable_or_prerequisite", f"needs:{stage} (no {s} table)"))
    if not (ctx.root / "data/cache/tokens/palette_global_k32.npz").exists():
        out.append(("unavailable_or_prerequisite", "global palette missing: run scripts/fit_global_palette.py once (ShotQA train stills, CPU)"))
    if _tokens_need_move_check(ctx):  # the movement slot is filled from a VERIFIED movement table, never from an identity-less one
        ok, why = move_table_verified(ctx)
        if not ok:
            out.append(("unavailable_or_prerequisite", f"needs:move ({why})"))
    return out


# ---------------------------------------------------------------- move (CPU)
MOVE_CODE_FILES = ["scripts/run_move_film.py", "scripts/p4_final_eval.py", "src/cgt/p4_camera_move/film_move.py", "src/cgt/p4_camera_move/flow.py", "src/cgt/p4_camera_move/depth_parallax.py",
                   "src/cgt/p4_camera_move/final_run.py", "src/cgt/p4_camera_move/final_eval.py", "src/cgt/p4_camera_move/labels.py"]
MOVE_CONFIG_FILES = ["results/phase4/frozen_config_flow_depth.json", "results/phase4/split_manifest.json",  # the frozen geometry configuration and split (read-only inputs of the runner)
                     "data/cache/p4/features_shotqa.parquet", "data/cache/p4/features_depth_shotqa.parquet"]  # the ShotQA train feature tables the frozen model is fitted on
# Files the move stage reads but that are NOT part of the movement identity.
# grpo.json is the ShotQA TRAIN label file TrainContext reads to refit the frozen geometry model; only its existence is checked here, it is never opened.
MOVE_PREREQ_FILES = ["data/external/shotqa/grpo.json"]
MIN_FREE_PERCENT = 25


def _move_table(ctx) -> Path:
    return ctx.root / f"data/processed/{ctx.film_id}_move.parquet"


def _move_record_path(ctx) -> Path:
    return ctx.manifest_dir / "move.identity.json"


def _move_code_identity(ctx) -> str:
    """SHA-256 over the bytes of the movement code (runner, film_move, flow, depth parallax, evaluation context) and of the frozen configuration/split files that exist."""
    h = hashlib.sha256()
    for rel in [*MOVE_CODE_FILES, *MOVE_CONFIG_FILES]:
        f = ctx.root / rel
        h.update(rel.encode() + (f.read_bytes() if f.exists() else b"absent"))
    return h.hexdigest()


def _move_inputs(ctx) -> dict:
    shots = ctx.root / f"data/processed/{ctx.film_id}_shots.parquet"
    return {"video_sha256": _video_sha(ctx), "shots_table_sha256": _sha_file(shots) if shots.exists() else None, "code_identity": _move_code_identity(ctx)}


def move_table_verified(ctx) -> tuple[bool, str]:
    """(ok, why not): a movement table counts only with an identity record written by the movement stage that binds the video, the Phase 1 shot-table bytes, the movement code and frozen
    configuration, and the table bytes, and only if it covers exactly the film's shots with valid, non-empty atom sets. File time, name or ids never establish identity."""
    table = _move_table(ctx)
    if not table.exists():
        return False, "no movement table"
    rec = _move_record_path(ctx)
    r = json.loads(rec.read_text()) if rec.exists() else {}
    if r.get("built_by") != "move_stage":
        return False, "legacy movement table: no identity record written by the movement stage (never adopted on file time, name or ids)"
    cur = _move_inputs(ctx)
    for k, v in cur.items():
        if r.get(k) != v:
            return False, f"the {k.replace('_', ' ')} differs from the identity recorded when the table was computed"
    if r.get("table_sha256") != _sha_file(table):
        return False, "the movement table differs from the bytes recorded when it was computed"
    return _move_table_covers(ctx)


def _move_table_covers(ctx) -> tuple[bool, str]:
    import pandas as pd

    from cgt.p4_camera_move.labels import schema_moves
    t = pd.read_parquet(_move_table(ctx))
    shots = _pdf(ctx.root, f"data/processed/{ctx.film_id}_shots.parquet")
    if t.shot_id.duplicated().any() or set(t.shot_id) != set(shots.shot_id) or len(t) != len(shots):
        return False, "the movement table does not cover exactly the film's shots"
    ok = set(schema_moves())
    for txt in t["move_atoms"]:
        atoms = [a.strip() for a in str(txt).split("+")] if isinstance(txt, str) and txt.strip() else []
        if not atoms or set(atoms) - ok:
            return False, "the movement table holds an empty or non-schema atom set"
    return True, ""


def move_available(ctx) -> str | None:
    if not (ctx.root / "scripts/run_move_film.py").exists():
        return "no film-level movement inference script (scripts/run_move_film.py) in this tree"
    return None


def move_prereq(ctx) -> list:
    out = []
    if not (ctx.root / f"data/processed/{ctx.film_id}_shots.parquet").exists():
        out.append(("unavailable_or_prerequisite", "needs:p1 (no shot table)"))
    if not _video_path(ctx).exists():
        out.append(("unavailable_or_prerequisite", f"video file missing: {_video_path(ctx)}"))
    missing = [rel for rel in MOVE_CONFIG_FILES[2:] if not (ctx.root / rel).exists()]
    if missing:
        out.append(("unavailable_or_prerequisite", f"the frozen model's ShotQA train feature tables are missing: {missing} (copy them from the Phase 4 cache; they bind the movement identity)"))
    absent = [rel for rel in MOVE_PREREQ_FILES if not (ctx.root / rel).is_file()]
    if absent:
        out.append(("unavailable_or_prerequisite", f"the ShotQA train label file the frozen movement model is refitted on is missing: {absent} (copy it from the main checkout's data/external/shotqa/)"))
    return out


def move_status(ctx) -> Status:
    video = _video_path(ctx)
    shots = ctx.root / f"data/processed/{ctx.film_id}_shots.parquet"
    if not video.exists() or not shots.exists():
        return Status(False, f"move:{ctx.film_id}:none", "the registered video or the Phase 1 shot table is missing")
    inputs = _move_inputs(ctx)
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    ok, why = move_table_verified(ctx)
    if not ok:
        return Status(False, key, f"{why}; the stage recomputes it (per-shot series are reused only through the runner's cache identity)")
    import pandas as pd

    from cgt.p4_camera_move import film_move as FM
    t = pd.read_parquet(_move_table(ctx))
    st = FM.describe([x.split(" + ") for x in t.move_atoms], t.fallback_top1.tolist())
    return Status(True, key, "verified movement table (identity record binds video, shot table, movement code and configuration, table bytes)", [_move_table(ctx)],
                  {"shots": st["n_shots"], "median_set_size": st["median_set_size"], "contradictory_share": round(st["contradictory_share"], 4)})


def _free_percent() -> int:
    from cgt.p4_camera_move import film_move as FM
    return FM.free_percent()


def _run_move_script(ctx, cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ctx.root, check=False).returncode


def _needs_allow_owned(film: dict) -> bool:
    """F-C1: an owned film, and every overlay film (local only whatever its public_domain flag says), runs the movement script with --allow-owned."""
    return not film.get("public_domain") or films_registry.is_overlay(film)


def move_run(ctx) -> dict:
    """CPU only: never the GPU lock, `nice -n 19`, at most 3 workers, fails closed below 25% free memory. Always recomputes the TABLE (the runner reuses per-shot series only when their
    file names, which encode film hash, frame range and code keys, match); the identity record is written only after a successful run."""
    import sys

    import pandas as pd
    free = _free_percent()
    if free < MIN_FREE_PERCENT:
        raise PipelineError("memory_guard", f"only {free}% memory free (< {MIN_FREE_PERCENT}%): the movement stage waits")
    cache = ctx.root / "data/cache/p4/film_move" / ctx.film_id
    cmd = ["nice", "-n", "19", sys.executable, str(ctx.root / "scripts/run_move_film.py"), "--film", ctx.film_id, "--workers", "3", "--out", str(_move_table(ctx)), "--overwrite",
           "--cache-dir", str(cache)] + (["--allow-owned"] if _needs_allow_owned(ctx.film) else [])
    before = _move_inputs(ctx)  # the identity inputs the runner starts from (F-V4): the record binds these, never inputs read after the run
    rc = _run_move_script(ctx, cmd)
    if rc != 0 or not _move_table(ctx).exists():
        raise PipelineError("stage_failed", f"run_move_film.py exited with status {rc}")
    after = _move_inputs(ctx)
    changed = [k for k in before if before[k] != after.get(k)]
    if changed:
        raise PipelineError("stage_failed", f"identity inputs changed while the runner ran ({', '.join(changed)}): no identity record written, the movement table stays unverified; rerun the stage")
    ok, why = _move_table_covers(ctx)
    if not ok:
        raise PipelineError("stage_failed", why)
    ctx.manifest_dir.mkdir(parents=True, exist_ok=True)
    _move_record_path(ctx).write_text(json.dumps({"built_by": "move_stage", **before, "table_sha256": _sha_file(_move_table(ctx)),
                                                  "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    t = pd.read_parquet(_move_table(ctx))
    return {"shots": len(t), "fallback_top1": int(t.fallback_top1.sum()), "no_features": int(t.no_features.sum())}


def registry() -> dict[str, Stage]:
    return {
        "p1": Stage("p1", False, p1_status, p1_run, p1_prereq, reference="NOTLD (95 min): about 10 min upper bound (video to shot table, file times)"),
        "p2": Stage("p2", False, p2_status, p2_run, p2_prereq, reference="NOTLD: about 12 min (shot table to Phase 2 table, file times)", uses_gpu=p2_uses_gpu),
        "p3c": Stage("p3c", True, p3c_status, p3c_run, p3c_prereq, reference="NOTLD (1,077 shots, 3 keyframes, method D): 4B default about 80 min (4.45 s/shot); 2B 44 to 55 min", needs_gpu=p3c_needs_gpu),
        "tokens": Stage("tokens", False, tokens_status, tokens_run, tokens_prereq, reference="NOTLD: seconds (palette fit is a one-off, about 7 min)"),
        "move": Stage("move", False, move_status, move_run, move_prereq, available=move_available, reference="CPU, 3 workers: NOTLD about 1 h from a cold cache, minutes with a warm one"),
    }
