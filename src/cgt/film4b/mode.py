"""Film mode 4B (+ C5) (docs/plans/film_4b_c5.md): pure parts, no model and no mlx import. Fixed values, argument checks, the C5 blend with
the committed Step A-4B weights (nothing refitted), the per-shot loop with its stop-time guard, and the v1 soft vote of both variants.
Reached only through `scripts/p3c_run_film.py --model 4b [--c5]`; the default film path never imports this package."""
import json
import os
import time
from collections.abc import Callable, Mapping, MutableMapping
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from cgt.p3b_vlm import guards
from cgt.p3c_tokens.vocab import ATTRS, labels, option_prompt
from cgt.p3m_matched import registry

ROOT = Path(__file__).resolve().parents[3]
TAG = "letterlogit_4b_c5_v1"                      # registry entry pinning the 4B identity, the C5 blend and the SigLIP heads
REPO, REVISION = registry.STEPC_4B_REPO, registry.STEPC_4B_REVISION
ADAPTER_SHA = registry.STEPC_4B_ADAPTER_SHA256    # aba56883...
LORA = registry.STEPC_4B_LORA                     # rank 16, alpha 32, dropout 0.05
MLX_ENV = dict(registry.STEPC_MLX_ENV)            # {"MLX_MAX_OPS_PER_BUFFER": "10", "MLX_MAX_MB_PER_BUFFER": "10"}
EXPECTED_W = {"size": 0.3, "lens": 0.2, "angle": 0.25, "framing": 0.2, "light_type": 0.0, "light_cond": 0.25, "composition": 0.15}
KEYFRAMES = ("kf_start", "kf_mid", "kf_end")      # v1 soft-vote order
LOCK_NONCE_ENV = "CGT_FILM4B_LOCK_NONCE"          # set by scripts/film4b_run.py for its child: the runner holds the lock
GPU_CAP_H, HARD_STOP = 1.0, "2026-10-22T10:00:00+02:00"
MAX_CRASH_RESUMES = 10
CACHE = ROOT / "data/cache/film4b"


def adapter_path() -> Path:
    return Path(registry.STEPC_4B_ADAPTER_PATH)


def check_args(a) -> None:
    """--model 4b / --c5 combinations (default path untouched): fail before any data or model is touched."""
    if getattr(a, "c5", False) and a.model != "4b":
        raise SystemExit("--c5 is defined only for --model 4b (the committed Step A-4B blend)")
    if a.model != "4b":
        raise SystemExit(f"unknown opt-in model {a.model!r}")
    if a.allow_owned:
        raise guards.NotAuthorized("--model 4b never runs on an owned film")
    if a.stepa:
        raise SystemExit("--model 4b and --stepa are separate opt-in modes")
    if a.table_only or a.out or a.limit:
        raise SystemExit("--model 4b writes only its cache; --table-only/--out/--limit do not apply")


def apply_mlx_env(environ: MutableMapping[str, str] | None = None, modules: Mapping | None = None) -> None:
    """MLX command buffers 10/10 must be in the environment BEFORE mlx is imported; refuse if mlx is already loaded with other values."""
    import sys
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    if "mlx.core" in modules and any(environ.get(k) != v for k, v in MLX_ENV.items()):
        raise guards.FailClosed(f"mlx was imported before the MLX buffer env {MLX_ENV} was set")
    environ.update(MLX_ENV)


def out_dir(film: str, n: int | None, seed: int, first: int | None = None) -> Path:
    tag = f"{film}_n{n}_seed{seed}" if n else f"{film}_all"
    return CACHE / (f"pipecheck_{tag}_first{first}" if first else tag)


# ---------- C5 ----------
def load_blend(data_root: Path = ROOT, provenance=None) -> dict:
    """C5 weights re-read from the committed Step A-4B fit and checked against every pin of the registry tag (blend hash, SigLIP head
    files, chosen head kinds, backbone revision, processor hash). Returns {ensemble_w, heads_dir, chosen, seeds, blend_sha256, ...}."""
    e = registry.entry(TAG)
    c5 = registry.require_c5_components(e, Path(data_root), provenance=provenance)
    if c5["ensemble_w"] != EXPECTED_W:
        raise guards.AdapterHashError(f"C5 weights {c5['ensemble_w']} differ from a recorded decision recipe {EXPECTED_W}")
    return {**c5, "chosen": dict(e["siglip"]["chosen"]), "seeds": list(e["siglip"]["seeds"])}


def blend(vlm_logits, sig, w: float) -> np.ndarray:
    """(1 - w) log_softmax(VLM letter logits over the n labels) + w log_softmax(SigLIP option scores over the same n labels)."""
    v, s = np.asarray(vlm_logits, dtype=float), np.asarray(sig, dtype=float)
    if v.shape != s.shape:
        raise ValueError(f"VLM ({v.shape}) and SigLIP ({s.shape}) scores do not cover the same options")
    return (1 - w) * log_softmax(v) + w * log_softmax(s)


def keyframe_vectors(rec: dict, sig: Mapping | None = None, w: Mapping | None = None) -> dict[str, np.ndarray]:
    """Per-attribute score vectors (schema order) of one keyframe: method D (log_softmax of the n letter logits) or, with `sig` and `w`,
    the C5 blend."""
    out = {}
    for a in ATTRS:
        lg = rec["logits"][a]
        if len(lg) != len(labels(a)):
            raise ValueError(f"{a}: {len(lg)} logits for {len(labels(a))} labels")
        out[a] = log_softmax(np.asarray(lg, dtype=float)) if sig is None else blend(lg, sig[a], w[a])
    return out


def softmax(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def primaries(per_kf: list[dict]) -> dict[str, str]:
    """v1 soft vote: mean of the per-keyframe softmax, argmax (ties to the lowest index)."""
    return {a: labels(a)[int(np.argmax(np.mean([softmax(k[a]) for k in per_kf], axis=0)))] for a in ATTRS}


# ---------- the GPU loop (engine injected) ----------
def score_keyframe(eng, image: str) -> tuple[dict, dict]:
    """Method D on one keyframe: one image state, then one all-label prompt per attribute (exactly the default single pass)."""
    t0 = time.perf_counter()
    st = eng.image_state(image)
    t1 = time.perf_counter()
    lg = {}
    for a in ATTRS:
        labs = labels(a)
        lg[a] = [round(float(x), 5) for x in eng.letter_logits(st, option_prompt(a, labs), len(labs))]
    return lg, {"image_s": t1 - t0, "total_s": time.perf_counter() - t0}


def done_keys(path: Path) -> set[tuple[int, str]]:
    """(shot, keyframe) already cached; a torn last line from a killed run is cut off."""
    if not path.exists():
        return set()
    good, keep = set(), 0
    raw = path.read_bytes()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            r = json.loads(line)
        except ValueError:
            break
        good.add((int(r["shot_id"]), r["kf"]))
        keep += len(line)
    if keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return good


def run_shots(eng, sample: pd.DataFrame, out_path: Path, image_of: Callable[[str], str], stop_at: datetime | None,
              est_shot_s: float | None = None, log: Callable[[str], None] = print, clock: Callable[[], float] = time.perf_counter,
              now: Callable[[], datetime] = lambda: datetime.now().astimezone()) -> dict:
    """One flushed record per (shot, keyframe), resumable. Before EACH new shot: if now + the last measured shot time (or `est_shot_s`
    before the first) would pass `stop_at`, no new shot starts (drop, not shorten)."""
    done = done_keys(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0, n_new, shots_new, stopped, last = clock(), 0, 0, None, est_shot_s
    with open(out_path, "a") as f:
        for i, row in enumerate(sample.itertuples(index=False)):
            sid = int(row.shot_id)
            if all((sid, kf) in done for kf in KEYFRAMES):
                continue
            if stop_at is not None and now() + timedelta(seconds=last or 0.0) > stop_at:
                stopped = "stop_at"
                log(f"no new shot: now + {last or 0:.1f}s would pass the stop time {stop_at.isoformat()}")
                break
            ts = clock()
            for kf in KEYFRAMES:
                if (sid, kf) in done:
                    continue
                src = getattr(row, kf)
                guards.require_images_readable([src])
                lg, tm = score_keyframe(eng, image_of(src))
                f.write(json.dumps({"shot_id": sid, "kf": kf, "logits": lg, **{k: round(v, 4) for k, v in tm.items()}}) + "\n")
                f.flush()
                n_new += 1
            last = clock() - ts
            shots_new += 1
            if (i + 1) % 10 == 0:
                log(f"{i + 1}/{len(sample)} shots, {clock() - t0:.0f}s, last shot {last:.1f}s, passes {getattr(eng, 'passes', '?')}")
    return {"new_keyframes": n_new, "new_shots": shots_new, "seconds": clock() - t0, "stopped": stopped, "last_shot_s": last}


def load_records(path: Path) -> dict[tuple[int, str], dict]:
    return {(int(r["shot_id"]), r["kf"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


def require_runner_lock(lock_path: Path, nonce: str) -> None:
    """A runner child may skip taking the lock only if the lock file exists and carries the runner's nonce."""
    try:
        txt = Path(lock_path).read_text()
    except OSError as e:
        raise guards.FailClosed(f"runner nonce given but no GPU lock at {lock_path}") from e
    if not nonce or f"nonce={nonce}" not in txt:
        raise guards.FailClosed(f"the GPU lock at {lock_path} is not held by this child's runner")
