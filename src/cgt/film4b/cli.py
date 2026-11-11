"""`scripts/p3c_run_film.py --model 4b [--c5]` (docs/plans/film_4b_c5.md): CPU SigLIP scores (with --c5), then the 4B method-D pass under
/tmp/cgt_gpu.lock with a memory guard and a per-shot stop-time guard. Outputs only under <repo>/data/cache/film4b/ (gitignored). Shared Track F helpers
(cgt.p3s_stepa.film_run): sample, keyframe guard, base image, SigLIP on CPU, memory reader."""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from cgt.film4b import mode as M
from cgt.p3b_vlm import guards
from cgt.p3s_stepa import film_run as R


def siglip_scores(od: Path, sample, kf_root: Path, film: str, c5: dict, log) -> Path:
    """Phase 3A scores of every sampled ORIGINAL keyframe (CPU, pinned heads): {keyframe file name: {attr: n scores}}."""
    sig_path = od / "siglip3a_scores.json"
    if sig_path.exists():
        return sig_path
    t0 = time.perf_counter()
    paths = [p for kf in M.KEYFRAMES for p in sample[kf]]
    feats = R.siglip_features(paths, od / "siglip_features", kf_root / "data/interim" / film / "keyframes")
    names = [Path(p).name for p in paths]
    X = np.stack([feats[n] for n in names])
    sc = R.siglip_label_scores(X, Path(c5["heads_dir"]), c5["chosen"])
    sig_path.write_text(json.dumps({"cpu_seconds": time.perf_counter() - t0, "chosen": c5["chosen"], "heads_dir": c5["heads_dir"],
                                    "siglip_identity_sha256": c5["siglip_identity_sha256"],
                                    "scores": {n: {k: [float(x) for x in v[i]] for k, v in sc.items()} for i, n in enumerate(names)}}))
    log(f"SigLIP scores for {len(names)} keyframes in {time.perf_counter() - t0:.0f}s (CPU)")
    return sig_path


def _lock(a, log):
    """(lock or None): a runner child checks the runner's nonce; a hand-started run takes the lock itself."""
    from cgt.pipeline.gpulock import DEFAULT
    nonce = os.environ.get(M.LOCK_NONCE_ENV)
    if nonce:
        M.require_runner_lock(Path(a.lock) if a.lock else DEFAULT, nonce)
        return None
    from cgt.p3s_stepa.film_cli import acquire_lock
    return acquire_lock(f"film4b {a.film} n={a.sample}", a.wait_lock_min, log, Path(a.lock) if a.lock else None)


def run(a, log=print, engine_factory=None, mem_reader=R.free_percent, provenance=None) -> Path:
    M.check_args(a)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    R.require_public_domain_strict(a.film)
    data_root = Path(a.data_root or M.ROOT)
    kf_root = Path(a.keyframe_root or data_root)
    sample = R.load_sample(data_root, a.film, a.sample, a.seed, kf_root)
    first = getattr(a, "first", None)
    if first:
        sample = sample.iloc[:first].reset_index(drop=True)
    od = M.out_dir(a.film, a.sample, a.seed, first)
    od.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in sample.shot_id]
    meta = {"film": a.film, "n": a.sample, "seed": a.seed, "first": first, "shot_ids": ids}
    if (od / "sample.json").exists() and json.loads((od / "sample.json").read_text()) != meta:
        raise guards.FailClosed(f"{od}/sample.json describes another sample")
    (od / "sample.json").write_text(json.dumps(meta))
    c5 = None
    if a.c5:  # C5 inputs on CPU, before the GPU lock
        c5 = M.load_blend(provenance=provenance)
        (od / "blend.json").write_text(json.dumps({k: v for k, v in c5.items()}, indent=1))
        siglip_scores(od, sample, kf_root, a.film, c5, log)
    done = M.done_keys(od / "logits.jsonl")
    if all((s, kf) in done for s in ids for kf in M.KEYFRAMES):
        log(f"all {len(ids)} shots already cached: no GPU work")
        return od
    free = mem_reader()
    if free < R.MIN_FREE_PERCENT:
        raise R.MemoryLow(f"memory guard: {free}% free < {R.MIN_FREE_PERCENT}%; nothing started")
    t_start = datetime.now().astimezone()
    stop = t_start + timedelta(hours=min(a.budget_hours, M.GPU_CAP_H))  # never more than the plan's cap per invocation
    if a.stop_at:
        stop = min(stop, datetime.fromisoformat(a.stop_at).astimezone())
    lock = _lock(a, log)
    try:
        t0 = time.perf_counter()
        adapter = M.adapter_path()
        if engine_factory is None:
            M.apply_mlx_env()  # mlx is imported only below, after the buffer env is set
            from cgt.film4b.engine4b import Engine4B
            eng = Engine4B(adapter)
        else:
            eng = engine_factory(adapter)
        load_s = time.perf_counter() - t0
        log(f"memory free {free}%; adapter {eng.adapter_sha[:16]} verified; 4B loaded in {load_s:.0f}s; {len(sample)} shots; stop {stop.isoformat()}")
        img_root = od / "images"
        equiv = equivalence(eng, sample, img_root) if engine_factory is None else None
        st = M.run_shots(eng, sample, od / "logits.jsonl", lambda src: R.view_image(src, img_root, "plain"), stop,
                         getattr(a, "est_shot_s", None), log)
    finally:
        if lock is not None:
            lock.release()
    man = {"film": a.film, "sample": a.sample, "seed": a.seed, "first": first, "c5": bool(a.c5), "adapter_sha256": eng.adapter_sha,
           "model": f"{M.REPO}@{M.REVISION}", "mlx_env": {k: os.environ.get(k) for k in M.MLX_ENV}, "model_load_s": load_s,
           "memory_free_percent": free, "data_root": str(data_root), "keyframe_root": str(kf_root), "stop_at": stop.isoformat(),
           "runner_held_lock": lock is None, "prefix_kv_vs_stock": equiv, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **st}
    runs = json.loads((od / "invocations.json").read_text()) if (od / "invocations.json").exists() else []
    (od / "invocations.json").write_text(json.dumps(runs + [man], indent=1))
    log(json.dumps(man))
    return od


def equivalence(eng, sample, img_root: Path) -> dict:
    """The prefix-KV scorer vs the stock full forward on the first sampled keyframe, lens (4 labels = letters A-D)."""
    from cgt.p3c_tokens.vocab import labels, option_prompt
    img = R.view_image(sample.iloc[0]["kf_start"], img_root, "plain")
    prompt = option_prompt("lens", labels("lens"))
    fast = np.asarray(eng.letter_logits(eng.image_state(img), prompt, 4), dtype=float)
    ref = np.asarray(eng.stock_letter_logits(img, prompt), dtype=float)
    return {"keyframe": Path(img).name, "attr": "lens", "max_abs_logit_diff": float(np.abs(fast - ref).max()),
            "same_argmax": int(np.argmax(fast)) == int(np.argmax(ref))}
