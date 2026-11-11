"""Label-free direction probes for the film movement slot (docs/phase4_move_probes.md): transforms of shot clips (time reversal, horizontal and vertical mirror), the partner maps, the statistics and the bands
of Check A (does the frozen geometry model read direction?) and of Check B (does its static / moving decision follow the size of the global motion?). They measure CONSISTENCY OF MODEL BEHAVIOUR, never accuracy. Nothing
here trains, tunes or selects; the frozen model, the features code and the Amendment 10 rule are untouched. A tripwire makes every label path unopenable."""
import builtins
import contextlib
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr

from cgt.p4_camera_move import film_move as FM
from cgt.p4_camera_move.labels import schema_moves

SEED = 0
N_BOOT = 1000
CONFIDENT = 0.25
MIN_CONFIDENT = 30
# Partner maps (docs/phase4_move_probes.md); an atom mapped to itself is invariant under that transform
PARTNERS = {
    "T": {"push-in": "pull-out", "pull-out": "push-in", "zoom-in": "zoom-out", "zoom-out": "zoom-in", "pan-left": "pan-right", "pan-right": "pan-left", "tilt-up": "tilt-down", "tilt-down": "tilt-up",
          "boom-up": "boom-down", "boom-down": "boom-up", "move-left": "move-right", "move-right": "move-left", "static": "static"},
    "H": {"pan-left": "pan-right", "pan-right": "pan-left", "move-left": "move-right", "move-right": "move-left"},
    "V": {"tilt-up": "tilt-down", "tilt-down": "tilt-up", "boom-up": "boom-down", "boom-down": "boom-up"},
}
TRANSFORMS = ("T", "H", "V")
FORBIDDEN = ("data/gold", "data/dev", "test.jsonl", "shotbench/test.tsv", "camerabench", "videos.csv", "shotqa/grpo.json")
TRAIN_LABEL_FILE = "data/external/shotqa/grpo.json"  # a recorded decision: the ONLY label file the probes may open, only inside `train_labels_allowed()`
_train_ok = [False]


# ---------------------------------------------------------------------------------------------- label tripwire
def _denied(path) -> bool:
    p = str(path)
    if any(f in p.lower() for f in FORBIDDEN):
        return not (_train_ok[0] and p.endswith(TRAIN_LABEL_FILE))  # the one exception: the ShotQA TRAIN labels, while `train_labels_allowed()` is active
    return False


@contextlib.contextmanager
def train_labels_allowed():
    """a recorded decision: permit exactly `data/external/shotqa/grpo.json` (ShotQA TRAIN labels, to rebuild the frozen predictor through `TrainContext`); every other label path stays denied, and so does this file outside the context."""
    _train_ok[0] = True
    try:
        yield
    finally:
        _train_ok[0] = False


@contextlib.contextmanager
def tripwire():
    """While active, `open` / `io.open` (hence pathlib's read_text and pandas) raise on any label path of the forbidden list: gold, dev, the CameraBench and ShotBench test files and the ShotQA label file (the latter
    only inside `train_labels_allowed()`)."""
    real = builtins.open

    def guarded(file, *a, **k):
        if _denied(file):
            raise PermissionError(f"label path refused by the tripwire: {file}")
        return real(file, *a, **k)
    builtins.open = io.open = guarded
    try:
        yield
    finally:
        builtins.open = io.open = real


def impl_identity() -> str:
    """SHA-256 over the bytes of the transform implementation (this module) and the clip-conformity implementation (`film_move`) and the target rate, read at call time: the producing identity of every cached
    transformed series (P-C1)."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update(Path(FM.__file__).read_bytes())
    h.update(repr(FM.TARGET_FPS).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- partner maps
def excluded_atoms(schema: list[str] | None = None) -> list[str]:
    """Schema atoms that appear in no partner map (no clear partner): printed in the report."""
    mapped = {a for m in PARTNERS.values() for a in m}
    return [a for a in (schema or schema_moves()) if a not in mapped]


def check_maps(schema: list[str] | None = None) -> None:
    """Every partner is a schema atom and every map is an involution."""
    schema = schema or schema_moves()
    for t, m in PARTNERS.items():
        for a, b in m.items():
            if a not in schema or b not in schema or m[b] != a:
                raise ValueError(f"partner map {t}: {a} -> {b} is not a valid involution over the schema")


def swapping_pairs(t: str) -> list[tuple[str, str]]:
    """The unordered opposite pairs swapped by transform `t` (a before a' in the map order)."""
    seen, out = set(), []
    for a, b in PARTNERS[t].items():
        if a != b and (b, a) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def invariant_atoms(t: str, model_atoms: list[str]) -> list[str]:
    """Model atoms evaluated for invariance under `t`: the atoms that map to themselves or are in no map of `t`, except the atoms excluded for lack of a partner."""
    exc = set(excluded_atoms())
    return [a for a in model_atoms if a not in exc and PARTNERS[t].get(a, a) == a]


# ---------------------------------------------------------------------------------------------- clip transforms
def _flip_code(t: str) -> int:
    return {"H": 1, "V": 0}[t]  # cv2.flip: 1 = about the vertical axis (left-right), 0 = about the horizontal axis (top-bottom)


def write_transformed_clip(src: Path, dst: Path, t: str | None, block: int = 64) -> int:
    """Write the lossless (FFV1) clip `dst` from the lossless clip `src` under transform `t` ('T' reversal, 'H' / 'V' mirror, None = identity through the same writing path); returns the frame count.
    Reversal reads blocks of `block` frames from the end, so memory stays bounded for 1080p shots."""
    cap = cv2.VideoCapture(str(src))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    written = 0
    if t == "T":
        for start in range(((n - 1) // block) * block, -1, -block):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(min(block, n - start)):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            for fr in reversed(frames):
                wr.write(fr)
                written += 1
    else:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            wr.write(cv2.flip(fr, _flip_code(t)) if t in ("H", "V") else fr)
            written += 1
    wr.release()
    cap.release()
    return written


# ---------------------------------------------------------------------------------------------- statistics
def rho(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def boot_rho(x, y, n_boot: int | None = None, seed: int = SEED) -> list[float]:
    """Percentile 95% interval of the Spearman rho over `n_boot` resamples of the shots (seed 0)."""
    n_boot = N_BOOT if n_boot is None else n_boot
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(x), len(x))
        vals.append(rho(x[ix], y[ix]))
    vals = [v for v in vals if np.isfinite(v)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else [float("nan"), float("nan")]


def band_pair(r: float) -> str:
    if not np.isfinite(r):
        return "undefined"
    return "reads direction" if r <= -0.5 else ("weak" if r < -0.2 else "does not read direction")


def band_invariant(r: float) -> str:
    return "stable" if np.isfinite(r) and r >= 0.8 else "unstable"


def band_auc(a: float) -> str:
    if not np.isfinite(a):
        return "undefined"
    return "consistent" if a >= 0.80 else ("weak" if a >= 0.60 else "inconsistent")


def auc_greater(x, y) -> float:
    """Probability that a random element of x is larger than a random element of y (ties count 1/2): the Mann-Whitney U / (n_x n_y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    r = rankdata(np.concatenate([x, y]))
    return float((r[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def boot_auc(x, y, n_boot: int | None = None, seed: int = SEED) -> list[float]:
    n_boot = N_BOOT if n_boot is None else n_boot
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    v = [auc_greater(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))]) for _ in range(n_boot)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


# ---------------------------------------------------------------------------------------------- Check A
def _set_has(sets, atom):
    return np.array([atom in s for s in sets])


def pair_stats(P_f, P_t, sets_f, sets_t, ia, ib, a, b, ok) -> dict:
    """Statistics of one swapped pair (a, b) over the non-flagged shots `ok` (docs/phase4_move_probes.md)."""
    d_f, d_t = (P_f[:, ia] - P_f[:, ib]), (P_t[:, ia] - P_t[:, ib])
    prim = rho(d_f[ok], d_t[ok])
    conf = ok & (np.abs(d_f) >= CONFIDENT)
    sec = rho(d_f[conf], d_t[conf])
    rng = np.random.default_rng(SEED)
    null = rho(d_f[ok], d_t[ok][rng.permutation(int(ok.sum()))])  # transformed table permuted across shots
    fa, fb, ta, tb = _set_has(sets_f, a), _set_has(sets_f, b), _set_has(sets_t, a), _set_has(sets_t, b)
    sets_stat = {}
    for name, (x, y, tx, ty) in {f"{a}_not_{b}": (fa, fb, ta, tb), f"{b}_not_{a}": (fb, fa, tb, ta)}.items():
        sel = ok & x & ~y
        n = int(sel.sum())
        sets_stat[name] = {"n": n, "flipped": round(float((ty & ~tx)[sel].mean()), 3) if n else None, "unchanged": round(float((tx & ~ty)[sel].mean()), 3) if n else None,
                           "other": round(float((~((ty & ~tx) | (tx & ~ty)))[sel].mean()), 3) if n else None}
    n_conf = int(conf.sum())
    return {"pair": [a, b], "n": int(ok.sum()), "rho_primary": round(prim, 3), "ci_primary": boot_rho(d_f[ok], d_t[ok]), "n_confident": n_conf, "rho_confident": round(sec, 3), "ci_confident": boot_rho(d_f[conf], d_t[conf]),
            "band_primary": band_pair(prim) if n_conf >= MIN_CONFIDENT else "insufficient support", "band_confident": band_pair(sec) if n_conf >= MIN_CONFIDENT else "insufficient support",
            "null_rho_permuted": round(null, 3), "sets": sets_stat}


def evaluate_check_a(P_f: np.ndarray, P_by_t: dict[str, np.ndarray], atoms: list[str], flagged: np.ndarray) -> dict:
    """Check A over one film's subset: `P_f` forward probabilities, `P_by_t[t]` the probabilities of the transformed clips (same shots, same order), `flagged` shots excluded from the primary analysis."""
    ok = ~np.asarray(flagged, dtype=bool)
    idx = {a: i for i, a in enumerate(atoms)}
    sets_f = FM.predict(P_f, atoms)[0]
    out: dict = {"n_subset": len(P_f), "n_flagged_excluded": int((~ok).sum()), "excluded_atoms": excluded_atoms(), "transforms": {}}
    for t in TRANSFORMS:
        P_t = P_by_t[t]
        sets_t = FM.predict(P_t, atoms)[0]
        pairs = [pair_stats(P_f, P_t, sets_f, sets_t, idx[a], idx[b], a, b, ok) for a, b in swapping_pairs(t) if a in idx and b in idx]
        inv = []
        for a in invariant_atoms(t, atoms):
            r = rho(P_f[ok, idx[a]], P_t[ok, idx[a]])
            rng = np.random.default_rng(SEED)
            inv.append({"atom": a, "rho": round(r, 3), "ci": boot_rho(P_f[ok, idx[a]], P_t[ok, idx[a]]), "band": band_invariant(r),
                        "null_rho_permuted": round(rho(P_f[ok, idx[a]], P_t[ok, idx[a]][rng.permutation(int(ok.sum()))]), 3)})
        out["transforms"][t] = {"pairs": pairs, "invariant": inv}
    return out


# ---------------------------------------------------------------------------------------------- Check B
def motion_summaries(series: dict) -> tuple[float, float]:
    """Primary `motion_translation` = median over frame pairs of sqrt(tx^2 + ty^2); secondary `motion_affine` = median over frame pairs of sqrt(div^2 + rot^2) (flow series, half-image-widths per second)."""
    return (float(np.median(np.hypot(series["tx"], series["ty"]))), float(np.median(np.hypot(series["div"], series["rot"]))))


def groups(sets: list[list[str]]) -> np.ndarray:
    """S = exactly {static}; M = set excludes static; X = static together with other atoms."""
    return np.array(["S" if s == ["static"] else ("X" if "static" in s else "M") for s in sets])


def _summ(v):
    v = np.asarray(v, dtype=float)
    return {"n": len(v), "median": round(float(np.median(v)), 4) if len(v) else None, "iqr": [round(float(np.percentile(v, 25)), 4), round(float(np.percentile(v, 75)), 4)] if len(v) else None}


def evaluate_check_b(sets: list[list[str]], p_static: np.ndarray, motion: dict[str, np.ndarray], flagged: np.ndarray) -> dict:
    """Check B over one film: groups from the predicted sets, motion-size summaries `motion` ({'primary': values, 'secondary': values}); the flagged shots are reported separately."""
    ok = ~np.asarray(flagged, dtype=bool)
    g = groups(sets)
    out: dict = {"n_all": len(g), "n_flagged_excluded": int((~ok).sum()), "n_flagged_by_group": {k: int(((g == k) & ~ok).sum()) for k in "SMX"}}
    for name, m in motion.items():
        m = np.asarray(m, dtype=float)
        e = {"groups": {k: _summ(m[ok & (g == k)]) for k in "SMX"}}
        mm, ss, xx = m[ok & (g == "M")], m[ok & (g == "S")], m[ok & (g == "X")]
        a = auc_greater(mm, ss)
        e["auc_M_vs_S"] = {"auc": round(a, 3), "ci": boot_auc(mm, ss), "band": band_auc(a) if min(len(mm), len(ss)) >= MIN_CONFIDENT else "insufficient support (a group has fewer than 30 shots)"}
        e["auc_X_vs_S"] = round(auc_greater(xx, ss), 3)
        e["auc_M_vs_X"] = round(auc_greater(mm, xx), 3)
        e["rho_motion_vs_p_static"] = round(rho(m[ok], p_static[ok]), 3)
        e["rho_ci"] = boot_rho(m[ok], p_static[ok])
        e["null_rho_permuted"] = round(rho(m[ok], p_static[ok][np.random.default_rng(SEED).permutation(int(ok.sum()))]), 3)
        has_static = np.array(["static" in s for s in sets])
        edges = np.percentile(m[ok], np.linspace(0, 100, 11))
        dec = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, 9)
        e["deciles"] = [{"decile": d + 1, "n": int((ok & (dec == d)).sum()), "motion_range": [round(float(edges[d]), 4), round(float(edges[d + 1]), 4)], "share_static_in_set": round(float(has_static[ok & (dec == d)].mean()), 3) if (ok & (dec == d)).any() else None}
                        for d in range(10)]
        e["flagged_median"] = round(float(np.median(m[~ok])), 4) if (~ok).any() else None
        out[name] = e
    return out
