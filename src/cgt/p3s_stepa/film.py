"""Track F: the Step A static winner C1+C3+C4*+C5 applied to film keyframes with full-vocabulary
method D. Opt-in only; the default film path never imports this module. Pure numpy here (no model, no file writes): parameter loading from the
accepted Step A results, the n-option shift mapping, and the combination with the parameters exactly as fitted (nothing is refitted)."""
import json
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from cgt.p3s_stepa.split import ATTRS
from cgt.p3s_stepa.views import applies

ROOT = Path(__file__).resolve().parents[3]
STATIC_VAL = ROOT / "results/p3s_stepa/static_val.json"
WINNER = "C1+C3+C4*+C5"
# view -> (cyclic option shift, image transform); the attributes a view applies to come from cgt.p3s_stepa.views.VIEWS
FILM_VIEWS = {"base": (0, "plain"), "shift1": (1, "plain"), "shift2": (2, "plain"), "shift3": (3, "plain"), "hflip": (0, "hflip"),
              "crop80": (0, "crop80")}


def load_params(path: Path = STATIC_VAL) -> dict:
    """The winner's saved fit: {views, calib: {attr: [log T, b_B, b_C, b_D]}, w: {attr: weight}}. Refuses anything but the adopted winner
    with letter calibration."""
    d = json.loads(Path(path).read_text())
    if d["decision"]["winner"] != WINNER:
        raise ValueError(f"Step A winner is {d['decision']['winner']}, expected {WINNER}")
    fit = d["results"][WINNER]["fit"]
    if fit["C4*"]["picked"] != "letter":
        raise ValueError("expected letter calibration")
    views = list(fit["views"])
    if views != list(FILM_VIEWS):
        raise ValueError(f"unexpected view set {views}")
    calib = {a: [float(x) for x in fit["calibration_params"][a]] for a in ATTRS}
    w = {a: float(fit["ensemble_w"][a]) for a in ATTRS}
    if any(len(p) != 4 for p in calib.values()):
        raise ValueError("letter calibration has 4 parameters per attribute")
    return {"views": views, "calib": calib, "w": w}


def shifted(labels: list, s: int) -> list:
    """Letter j shows label (j + s) mod n."""
    n = len(labels)
    return [labels[(j + s) % n] for j in range(n)]


def to_schema(view_scores, s: int) -> np.ndarray:
    """Scores in a shifted view's letter order -> schema order: label i was shown at letter (i - s) mod n."""
    v = np.asarray(view_scores, dtype=float)
    n = len(v)
    return np.array([v[(i - s) % n] for i in range(n)])


def views_for(attr: str, views: list[str] | None = None) -> list[str]:
    return [v for v in (views or list(FILM_VIEWS)) if applies(v, attr)]


def averaged(attr: str, view_logits: dict[str, list], views: list[str] | None = None) -> np.ndarray:
    """Mean over the applicable views of option log-probs in schema order. A missing applicable view raises KeyError."""
    use = views_for(attr, views)
    if not use:
        raise ValueError(f"no view applies to {attr}")
    return np.mean([to_schema(log_softmax(np.asarray(view_logits[v], dtype=float)), FILM_VIEWS[v][0]) for v in use], axis=0)


def letter_design(n: int) -> np.ndarray:
    """(n, 3): the Step A letter design (original positions A-D, b_A = 0) on schema positions 0-3; positions >= 4 have no parameter (0)."""
    d = np.zeros((n, 3))
    for k in range(1, min(n, 4)):
        d[k, k - 1] = 1.0
    return d


def calibrate(s: np.ndarray, p: list[float], letter_bias: bool = True) -> np.ndarray:
    z = np.asarray(s, dtype=float) / np.exp(p[0])
    return z + letter_design(len(z)) @ np.asarray(p[1:]) if letter_bias else z


def blend(z: np.ndarray, sig, w: float) -> np.ndarray:
    return (1 - w) * log_softmax(z) + w * log_softmax(np.asarray(sig, dtype=float))


def combine(attr: str, view_logits: dict[str, list], sig, params: dict, views: list[str] | None = None, calib: bool = True,
            letter_bias: bool = True, ensemble: bool = True) -> np.ndarray:
    """Full-vocabulary score vector (schema order) of one keyframe for `attr`. Defaults = the winner C1+C3+C4*+C5; the switches give the
    exploratory decomposition only."""
    s = averaged(attr, view_logits, views or params["views"])
    if calib:
        s = calibrate(s, params["calib"][attr], letter_bias)
    if ensemble:
        if sig is None:
            raise ValueError("C5 needs the Phase 3A scores")
        s = blend(s, sig, params["w"][attr])
    return np.asarray(s, dtype=float)


def sample_shots(shot_ids, n: int, seed: int) -> list[int]:
    ids = sorted(int(x) for x in shot_ids)
    if n >= len(ids):
        return ids
    return sorted(int(x) for x in np.random.default_rng(seed).choice(ids, n, replace=False))


# ---------- applying saved parameters to the ShotQA Step A cache (reproduction check; no fit) ----------
def apply_saved_shotqa(qs: list[dict], views: dict, sig: dict, params: dict) -> dict:
    """Winner scores for ShotQA 4-option questions with the SAVED parameters (no fit): Step A's combine path with fixed numbers."""
    from cgt.p3s_stepa import combine as C
    sc = C.averaged(qs, views, params["views"])
    out = {}
    for q in qs:
        p = np.asarray(params["calib"][q["attribute"]])
        z = C.calibrated(sc[q["key"]], p, C._design(q, "letter"))
        out[q["key"]] = blend(z, sig[q["key"]], params["w"][q["attribute"]])
    return out
