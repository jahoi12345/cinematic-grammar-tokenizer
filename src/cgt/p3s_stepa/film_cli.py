"""`scripts/p3c_run_film.py --stepa` (Track F, opt-in): CPU SigLIP scores, then the GPU views under /tmp/cgt_gpu.lock with a memory guard, a GPU-time
budget and a stop-at time. Outputs only under <repo>/data/cache/stepa_films/<film>_<sample tag>/ (gitignored)."""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from cgt.p3b_vlm import guards
from cgt.p3s_stepa import film_run as R

CHOSEN = R.ROOT / f"results/phase3/runs/{R.SIGLIP_TAG}/chosen_heads_{R.SIGLIP_TAG}.json"


def out_dir(film: str, n: int | None, seed: int) -> Path:
    return R.ROOT / "data/cache/stepa_films" / (f"{film}_n{n}_seed{seed}" if n else f"{film}_all")


def acquire_lock(job: str, wait_min: float, log, lock_path: Path | None = None):
    from cgt.pipeline.gpulock import GpuLock, LockHeld
    t_end = time.time() + 60 * wait_min
    while True:
        try:
            return GpuLock(job, lock_path).acquire()
        except LockHeld as exc:
            if time.time() >= t_end:
                raise R.GpuBusy(str(exc)) from exc
            log(f"GPU lock held, waiting: {exc}")
            time.sleep(30)


def run(a, log=print, engine_factory=None, mem_reader=R.free_percent) -> Path:
    R.require_public_domain_strict(a.film)
    data_root = Path(a.data_root or R.ROOT)
    kf_root = Path(getattr(a, "keyframe_root", None) or data_root)
    sample = R.load_sample(data_root, a.film, a.sample, a.seed, kf_root)
    od = out_dir(a.film, a.sample, a.seed)
    od.mkdir(parents=True, exist_ok=True)
    (od / "sample.json").write_text(json.dumps({"film": a.film, "n": a.sample, "seed": a.seed, "shot_ids": [int(x) for x in sample.shot_id]}))
    # C5 input on CPU (before the GPU lock)
    sig_path = od / "siglip3a_scores.json"
    if not sig_path.exists():
        t0 = time.perf_counter()
        paths = [p for kf in R.KEYFRAMES for p in sample[kf]]
        feats = R.siglip_features(paths, od / "siglip_features", kf_root / "data/interim" / a.film / "keyframes")
        names = [Path(p).name for p in paths]
        X = np.stack([feats[n] for n in names])
        chosen = json.loads(CHOSEN.read_text())
        sc = R.siglip_label_scores(X, Path(a.siglip_models or R.ROOT / "models/p3" / R.SIGLIP_TAG), chosen)
        sig_path.write_text(json.dumps({"cpu_seconds": time.perf_counter() - t0, "chosen": chosen,
                                        "scores": {n: {k: [float(x) for x in v[i]] for k, v in sc.items()} for i, n in enumerate(names)}}))
        log(f"SigLIP scores for {len(names)} keyframes in {time.perf_counter() - t0:.0f}s (CPU)")
    free = mem_reader()
    if free < R.MIN_FREE_PERCENT:
        raise R.MemoryLow(f"memory guard: {free}% free < {R.MIN_FREE_PERCENT}%; nothing started")
    stop = datetime.fromisoformat(a.stop_at).astimezone() if a.stop_at else None
    lock = acquire_lock(f"stepa_films {a.film} n={a.sample}", a.wait_lock_min, log, Path(a.lock) if a.lock else None)
    try:
        t0 = time.perf_counter()
        adapter = data_root / "data/cache/p3c/ckpt_100.safetensors"
        if engine_factory is None:
            from cgt.p3c_tokens.engine import Engine  # mlx: imported only here
            eng = Engine(adapter)
        else:
            eng = engine_factory(adapter)
        load_s = time.perf_counter() - t0
        log(f"memory free {free}%; adapter {eng.adapter_sha[:16]} verified; model loaded in {load_s:.0f}s; {len(sample)} shots")
        st = R.run_views(eng, sample, od / "views.jsonl", od / "images", a.budget_hours * 3600 - load_s, stop, log)
    finally:
        lock.release()
    man = {"film": a.film, "sample": a.sample, "seed": a.seed, "adapter_sha256": eng.adapter_sha, "model_load_s": load_s,
           "memory_free_percent": free, "data_root": str(data_root), "keyframe_root": str(kf_root), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **st}
    runs = json.loads((od / "invocations.json").read_text()) if (od / "invocations.json").exists() else []
    (od / "invocations.json").write_text(json.dumps(runs + [man], indent=1))
    log(json.dumps(man))
    return od


def check_stepa_args(a) -> None:
    if a.allow_owned:
        raise guards.NotAuthorized("--stepa never runs on an owned film")
    if a.table_only or a.out:
        raise SystemExit("--stepa writes only its cache; --table-only/--out do not apply")
