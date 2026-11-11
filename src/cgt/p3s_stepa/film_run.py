"""Track F opt-in film mode (docs/plans/stepa_films.md): run the Step A winner's views on film keyframes and keep the per-view letter logits in a
resumable cache; combine them with the saved Step A parameters (cgt.p3s_stepa.film) into per-keyframe score vectors and v1 soft-vote primaries.
Reached only through `scripts/p3c_run_film.py --stepa`; the default film path never imports this module. Writes only under
<this repo>/data/cache/stepa_films/ (gitignored); never writes the accepted p3c or token tables. Public-domain films only."""
import functools
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.images import BUDGET_PX, shrunk_path
from cgt.p3c_tokens.vocab import ATTRS, labels, option_prompt
from cgt.p3s_stepa import film as F
from cgt.p3s_stepa.views import prepare_image

ROOT = Path(__file__).resolve().parents[3]
KEYFRAMES = ("kf_start", "kf_mid", "kf_end")  # v1 soft-vote order
MIN_FREE_PERCENT = 25
SIGLIP_TAG = "siglip_base_filmsplit_lr1e-2_ep300"
SIGLIP_SEEDS = (0, 1, 2)


class MemoryLow(guards.FailClosed):
    exit_code = 12


class GpuBusy(guards.FailClosed):
    exit_code = 11


def free_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.split(":")[1].strip().rstrip("%"))
    return 0


def require_public_domain_strict(film: str, root: Path | None = None) -> dict:
    """Track F runs on configs/films.yaml public-domain films only: no owned film, no local-overlay clip, no override flag."""
    import yaml
    root = root or ROOT
    films = {f["id"]: f for f in yaml.safe_load((root / "configs/films.yaml").read_text())["films"]}
    if film not in films or films[film].get("public_domain") is not True:
        raise guards.NotAuthorized(f"{film}: Step A film mode runs on public-domain films of configs/films.yaml only")
    return films[film]


def keyframe_file(data_root: Path, film: str, rel: str) -> Path:
    """Shot-table keyframe path -> checked path inside <data_root>/data/interim/<film>/keyframes (DataRootViolation otherwise)."""
    kroot = Path("data/interim") / film / "keyframes"
    p = Path(os.path.normpath(rel))
    if p.is_absolute() or ".." in Path(rel).parts:
        raise guards.DataRootViolation(f"keyframe {rel!r} must be a relative path under {kroot}")
    try:
        part = p.relative_to(kroot)
    except ValueError as e:
        raise guards.DataRootViolation(f"keyframe {rel!r} is outside {kroot}") from e
    return guards.require_under_image_root(part, Path(data_root) / kroot)


def load_sample(data_root: Path, film: str, n: int | None, seed: int, kf_root: Path | None = None) -> pd.DataFrame:
    """Shot table from `data_root`; keyframes checked under `kf_root` (default `data_root`)/data/interim/<film>/keyframes."""
    shots = pd.read_parquet(Path(data_root) / f"data/processed/{film}_shots.parquet")
    if set(shots.film_id) != {film}:
        raise guards.DataRootViolation(f"shot table holds films {sorted(set(shots.film_id))}")
    keep = F.sample_shots(shots.shot_id, n, seed) if n else sorted(int(x) for x in shots.shot_id)
    s = shots[shots.shot_id.isin(keep)].sort_values("shot_id").reset_index(drop=True)
    for kf in KEYFRAMES:
        s[kf] = [str(keyframe_file(kf_root or data_root, film, r)) for r in s[kf]]
    return s[["film_id", "shot_id", *KEYFRAMES]]


def view_image(src: str, img_root: Path, transform: str) -> str:
    if transform == "plain":
        return shrunk_path(src, img_root / "base")  # exactly the default film path's image
    return str(prepare_image(src, img_root / f"{transform}_{BUDGET_PX}" / (Path(src).stem + ".jpg"), transform, BUDGET_PX))


def score_keyframe(eng, src: str, img_root: Path, views: list[str] | None = None) -> tuple[dict, dict]:
    """{view: {attr: n letter logits in that view's letter order}} for every applicable (view, attr), and timings: `default_s` = base image
    state + the s=0 method-D passes (what the default mode does), `total_s` = everything."""
    views = views or list(F.FILM_VIEWS)
    out: dict[str, dict] = {v: {} for v in views}
    t0 = time.perf_counter()
    states: dict[str, object] = {}
    default_s = 0.0
    for v in views:
        shift, tf = F.FILM_VIEWS[v]
        attrs = [a for a in ATTRS if v in F.views_for(a, views)]
        if not attrs:
            continue
        t1 = time.perf_counter()
        if tf not in states:
            states[tf] = eng.image_state(view_image(src, img_root, tf))
        for a in attrs:
            labs = labels(a)
            out[v][a] = [round(float(x), 5) for x in eng.letter_logits(states[tf], option_prompt(a, F.shifted(labs, shift)), len(labs))]
        if v == "base":
            default_s = time.perf_counter() - t1
    return out, {"default_s": default_s, "total_s": time.perf_counter() - t0}


def _done(path: Path) -> set[tuple[int, str]]:
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
    if keep != len(raw):  # torn last line from a killed run
        with open(path, "r+b") as f:
            f.truncate(keep)
    return good


def run_views(eng, sample: pd.DataFrame, out_path: Path, img_root: Path, budget_s: float, stop_at: datetime | None = None,
              log: Callable[[str], None] = print, clock: Callable[[], float] = time.perf_counter) -> dict:
    """Append one record per (shot, keyframe) to `out_path`; resumable. No new SHOT starts after `budget_s` seconds of this invocation or
    after `stop_at` (the shot in progress finishes its 3 keyframes)."""
    done = _done(out_path)
    t0, n_new, stopped = clock(), 0, None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for i, row in enumerate(sample.itertuples(index=False)):
            if all((int(row.shot_id), kf) in done for kf in KEYFRAMES):
                continue
            if clock() - t0 >= budget_s:
                stopped = "budget"
                break
            if stop_at and datetime.now().astimezone() >= stop_at:
                stopped = "stop_at"
                break
            for kf in KEYFRAMES:
                if (int(row.shot_id), kf) in done:
                    continue
                src = getattr(row, kf)
                guards.require_images_readable([src])
                views, tm = score_keyframe(eng, src, img_root)
                f.write(json.dumps({"shot_id": int(row.shot_id), "kf": kf, "views": views, **{k: round(v, 4) for k, v in tm.items()}}) + "\n")
                f.flush()
                n_new += 1
            if (i + 1) % 10 == 0:
                log(f"{i + 1}/{len(sample)} shots, {clock() - t0:.0f}s, passes {getattr(eng, 'passes', '?')}")
    return {"new_keyframes": n_new, "seconds": clock() - t0, "stopped": stopped}


def load_records(path: Path) -> dict[tuple[int, str], dict]:
    return {(int(r["shot_id"]), r["kf"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


# ---------- Phase 3A SigLIP scores of keyframes (CPU) ----------
def siglip_features(paths: list[str], cache_dir: Path, allowed_root: Path) -> dict[str, np.ndarray]:
    """{basename: l2-normalised SigLIP-base embedding} via the Phase 3A embedder (cgt.p3_static_attrs.features), forced to CPU, offline."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import transformers  # noqa: F401  (imported before the patch below: its import inspects torch)

    from cgt.p3_static_attrs import features, heads
    torch.backends.mps.is_available = functools.wraps(torch.backends.mps.is_available)(lambda: False)  # CPU only: no GPU use outside the lock
    features.embed_images([Path(p) for p in paths], cache_dir, workers=0, allowed_root=allowed_root, log=lambda m: None)
    ids, feats = features.load_features(cache_dir)
    X = heads.l2norm(feats)
    return {str(i): X[n] for n, i in enumerate(ids)}


def siglip_label_scores(X: np.ndarray, model_dir: Path, chosen: dict[str, str]) -> dict[str, np.ndarray]:
    """{attr: (N, n_labels) option scores with every label a single-label option}, exactly the Phase 3A scoring of scripts/p3s_stepa_siglip.py."""
    import torch

    from cgt.eval import metrics as M
    from cgt.p3_static_attrs import heads
    out = {}
    x = torch.from_numpy(np.asarray(X, dtype=np.float32))
    for a in ATTRS:
        kind_name, n = chosen[a], len(labels(a))
        preds = []
        for s in SIGLIP_SEEDS:
            if a == "size" and kind_name in {"softmax", "coral", "softmax_mlp", "coral_mlp"}:
                kind, hidden = kind_name.split("_")[0], 512 if kind_name.endswith("_mlp") else 0
            else:
                kind, hidden = "multihot", (512 if kind_name == "mlp" else 0)
            h = heads.Head(x.shape[1], n, kind, hidden)
            h.load_state_dict(torch.load(Path(model_dir) / f"{a}_{kind_name}_s{s}.pt"))
            h.eval()
            preds.append(h.predict(x))
        p = np.mean(preds, axis=0)
        opt = [[i] for i in range(n)]
        fn = M.multihot_option_scores if kind == "multihot" else M.categorical_option_scores
        out[a] = np.array([fn(row, opt) for row in p])
    return out


# ---------- combination and soft vote ----------
def keyframe_scores(rec: dict, sig: dict[str, np.ndarray] | None, params: dict, **switches) -> dict[str, np.ndarray]:
    return {a: F.combine(a, {v: rec["views"][v][a] for v in rec["views"] if a in rec["views"][v]}, None if sig is None else sig[a], params,
                         **switches) for a in ATTRS}


def softmax(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def primaries(per_kf: list[dict[str, np.ndarray]]) -> dict[str, str]:
    """v1 soft vote: mean of the per-keyframe softmax, argmax (ties to the lowest index)."""
    return {a: labels(a)[int(np.argmax(np.mean([softmax(k[a]) for k in per_kf], axis=0)))] for a in ATTRS}
