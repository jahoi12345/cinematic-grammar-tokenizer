"""Phase 3C step 3: run the Phase 3C model (inference only) over every shot of a public-domain film with the SELECTED method and write
data/processed/<film>_p3c.parquet. Resumable; idempotent.
DEFAULT model (pipeline version p3c-v2-4b, docs/plans/default_4b.md): the Step C 4B single pass (method D, no C5),
under /tmp/cgt_gpu.lock (a runner child checks the runner's nonce instead), MLX buffers 10/10. It never overwrites a live table of another
run that has no _pre_4b backup (use --out, then scripts/rebuild_default_4b.py). `--model 2b` = the old default, call for call. Usage:
  nohup caffeinate -i uv run python scripts/p3c_run_film.py --film night_of_the_living_dead --method <selected> > data/cache/p3c/logs/film_run.log 2>&1 &
  uv run python scripts/p3c_run_film.py --film ... --method ... --table-only      (rebuild the parquet from the cache)
Opt-in Track F mode (a recorded decision, docs/plans/stepa_films.md; default OFF, the default path is unchanged and never imports it):
  ... --method D --stepa [--sample 200 --seed 20261017] [--data-root <repo with data/>] [--budget-hours 2] [--stop-at ISO] [--siglip-models DIR]
  ... --model 2b                                   (the frozen 2B: old default, exactly; Track F --stepa runs only on it)
Opt-in 4B (+ C5) measurement mode (docs/plans/film_4b_c5.md; selected by --sample/--first/--c5 without --stepa; writes
only data/cache/film4b/); normally started by scripts/film4b_run.py (lock, budget, resume):
  ... --method D --model 4b [--c5] --sample 200 --seed 20261017 --data-root <repo with data/> [--keyframe-root ...] [--stop-at ISO]"""
import argparse
import json
from pathlib import Path

from cgt.p3b_vlm import guards
from cgt.p3c_tokens import models, runner

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "data/cache/p3c/ckpt_100.safetensors"


def _measurement_mode(a) -> bool:
    return bool(a.c5 or a.first or (a.sample is not None and not a.stepa))


def run_4b(a) -> None:
    """The default full-film path (p3c-v2-4b)."""
    from datetime import datetime

    from cgt.p3c_tokens import rebuild
    if a.stepa or a.c5 or a.first:
        raise SystemExit("internal: not the full-film 4B path")
    sha = models.expected_sha("4b")
    ident = models.model_identity("4b")
    out = Path(a.out) if a.out else ROOT / f"data/processed/{a.film}_p3c.parquet"
    alphas = _alphas(a.method)
    if a.table_only:
        runner.require_public_domain(a.film, a.allow_owned)
        if guards.sha256_file(models.adapter_path("4b")) != sha:
            raise guards.AdapterHashError(f"4B adapter sha256 != pinned {sha}")
        key = runner.run_key(a.film, sha, runner.code_sha(), a.method, model_identity=ident)
        if not a.out:
            rebuild.guard_live_replace(a.film, ROOT, key)
        df = runner.build_table(a.film, ROOT / "data/cache/p3c/runs" / f"{a.film}_{key}", alphas, out)
        print(f"wrote {out}: {len(df)} rows, {df.shape[1]} columns; run key {key}")
        return
    if not a.out and not a.limit:  # before the lock, the data or the model: never overwrite another run's live table without a backup
        runner.require_public_domain(a.film, a.allow_owned)
        rebuild.guard_live_replace(a.film, ROOT, runner.run_key(a.film, sha, runner.code_sha(), a.method, model_identity=ident))
    import os

    from cgt.film4b import mode as M
    from cgt.pipeline.gpulock import DEFAULT
    nonce = os.environ.get(M.LOCK_NONCE_ENV)
    lock = None
    if nonce:
        M.require_runner_lock(Path(a.lock) if a.lock else DEFAULT, nonce)
    else:
        from cgt.p3s_stepa.film_cli import acquire_lock
        lock = acquire_lock(f"p3c_run_film 4b {a.film}", a.wait_lock_min, lambda m: print(m, flush=True), Path(a.lock) if a.lock else None)
    try:
        stop = datetime.fromisoformat(a.stop_at).astimezone() if a.stop_at else None
        est = a.est_shot_s / 3 if a.est_shot_s else None
        rd = runner.run_film(a.film, a.method, models.adapter_path("4b"), limit=a.limit, log=lambda m: print(m, flush=True),
                             allow_owned=a.allow_owned, model="4b", stop_at=stop, est_item_s=est)
    finally:
        if lock is not None:
            lock.release()
    state = json.loads((rd / "manifest.json").read_text()).get("state")
    if state == runner.STOPPED:
        print(f"STOPPED at the stop time: run {rd.name} kept (resumable), no table written", flush=True)
        raise SystemExit(rebuild.STOPPED_EXIT)
    if a.limit:
        print("limited run: no table written")
        return
    df = runner.build_table(a.film, rd, alphas, out)
    print(f"wrote {out}: {len(df)} rows, {df.shape[1]} columns; run dir {rd}")


def _alphas(method: str) -> dict:
    sel = json.loads((ROOT / "results/phase3c/method_selection.json").read_text())
    return {k: v["alpha"] for k, v in sel["results"][method]["per_attribute"].items() if "alpha" in v}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--allow-owned", action="store_true", help="owned (not public-domain) film: LOCAL inference only, outputs stay gitignored, never published")
    ap.add_argument("--out", default=None, help="parquet path (default data/processed/<film>_p3c.parquet); use another path to compare before replacing")
    ap.add_argument("--stepa", action="store_true", help="opt-in Step A static winner views (Track F); writes only data/cache/stepa_films/")
    ap.add_argument("--sample", type=int, default=None, help="--stepa: seeded random sample of N shots")
    ap.add_argument("--seed", type=int, default=20261017, help="--stepa: sample seed")
    ap.add_argument("--data-root", default=None, help="--stepa: read-only repo root holding data/ (default: this repo)")
    ap.add_argument("--keyframe-root", default=None, help="--stepa: read-only repo root holding data/interim/<film>/keyframes (default: --data-root)")
    ap.add_argument("--budget-hours", type=float, default=2.0, help="--stepa: GPU-time cap; no new shot starts after it")
    ap.add_argument("--stop-at", default=None, help="--stepa / 4B: no new shot (4B default: no new keyframe) starts after this local time (ISO)")
    ap.add_argument("--wait-lock-min", type=float, default=30.0, help="--stepa: minutes to wait for /tmp/cgt_gpu.lock")
    ap.add_argument("--lock", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--siglip-models", default=None, help="--stepa: directory of the Phase 3A heads (default models/p3/<tag>)")
    ap.add_argument("--model", default=None, choices=list(models.MODELS),
                    help=f"Phase 3C model (default {models.DEFAULT_MODEL}: Step C 4B single pass, {models.PIPELINE_VERSION['4b']}); 2b = the old default (opt-in)")
    ap.add_argument("--c5", action="store_true", help="--model 4b: add the committed Step A-4B C5 SigLIP blend")
    ap.add_argument("--first", type=int, default=None, help="--model 4b: only the first K shots of the seeded sample (pipeline check)")
    ap.add_argument("--est-shot-s", type=float, default=None, help="--model 4b: shot time used by the stop guard before the first shot")
    a = ap.parse_args(argv)
    sel = json.loads((ROOT / "results/phase3c/method_selection.json").read_text())
    if sel["rule"]["selected"] != a.method:
        raise SystemExit(f"--method {a.method} is not the method chosen by the pre-registered rule ({sel['rule']['selected']})")
    if _measurement_mode(a):  # an earlier change's semantics unchanged: it needs an explicit --model 4b (check_args refuses otherwise)
        from cgt.film4b import cli as film4b_cli  # opt-in only
        try:
            film4b_cli.run(a)
        except guards.FailClosed as e:
            print(f"FAIL CLOSED ({type(e).__name__}): {e}", flush=True)
            raise SystemExit(e.exit_code) from e
        return
    if a.stepa and a.model not in (None, "2b"):
        raise SystemExit("--stepa (Track F) is defined on the frozen 2B: use --model 2b or no --model")
    if not a.stepa and models.resolve(a.model) == "4b":
        try:
            run_4b(a)
        except guards.FailClosed as e:
            print(f"FAIL CLOSED ({type(e).__name__}): {e}", flush=True)
            raise SystemExit(e.exit_code) from e
        return
    if a.stepa:
        from cgt.p3s_stepa import film_cli  # opt-in only
        try:
            film_cli.check_stepa_args(a)
            film_cli.run(a)
        except guards.FailClosed as e:
            print(f"FAIL CLOSED ({type(e).__name__}): {e}", flush=True)
            raise SystemExit(e.exit_code) from e
        return
    alphas = {k: v["alpha"] for k, v in sel["results"][a.method]["per_attribute"].items() if "alpha" in v}
    try:
        if a.table_only:
            runner.require_public_domain(a.film, a.allow_owned)
            sha = guards.sha256_file(ADAPTER)
            frozen = json.loads((ROOT / "results/phase3b/frozen_manifest.json").read_text())["adapter_sha256"]
            if sha != frozen:  # --table-only never trusts an unchecked adapter file to pick the cache directory
                raise guards.AdapterHashError(f"adapter {ADAPTER} sha256 {sha} != frozen {frozen}")
            run_dir = ROOT / "data/cache/p3c/runs" / f"{a.film}_{runner.run_key(a.film, sha, runner.code_sha(), a.method)}"
        else:
            run_dir = runner.run_film(a.film, a.method, ADAPTER, limit=a.limit, log=lambda m: print(m, flush=True), allow_owned=a.allow_owned)
        if a.limit:
            print("limited run: no table written")
            return
        df = runner.build_table(a.film, run_dir, alphas, Path(a.out) if a.out else ROOT / f"data/processed/{a.film}_p3c.parquet")
        print(f"wrote {a.out or f'data/processed/{a.film}_p3c.parquet'}: {len(df)} rows, {df.shape[1]} columns; run dir {run_dir}")
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


if __name__ == "__main__":
    main()
