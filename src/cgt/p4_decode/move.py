"""Movement set decoders D0-D3 and the option-level scoring of docs/acc_step_a_move_plan.md (pure, CPU). The defaults of the film runner (`film_move.predict`) are NOT changed:
these are options. Thresholds are fitted ONLY from the calibration arrays passed to `fit_thresholds`; no function here reads a label store."""
import itertools
import math

import numpy as np

from cgt.p4_camera_move.baseline import option_scores

LETTERS = "ABCD"
GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95
CANDIDATES = ("D0", "D1", "D2", "D3")
_EPS = 1e-6
FUSION_GRID = tuple(round(0.1 * i, 1) for i in range(11))
# copies of film_move.OPPOSITE_PAIRS / film_move.is_contradictory (film_move imports torch; equality is asserted in the tests)
OPPOSITE_PAIRS = (("push-in", "pull-out"), ("pan-left", "pan-right"), ("tilt-up", "tilt-down"), ("boom-up", "boom-down"), ("zoom-in", "zoom-out"), ("move-left", "move-right"))


def is_contradictory(atoms) -> dict:
    """The declared definition (docs/phase4_film_move.md): both atoms of an opposite pair, or `static` together with any other atom."""
    s = set(atoms)
    opposite = any(a in s and b in s for a, b in OPPOSITE_PAIRS)
    static_other = "static" in s and len(s) > 1
    return {"opposite_pair": opposite, "static_with_other": static_other, "contradictory": opposite or static_other}


# ---------------------------------------------------------------------------------------------------- thresholds / recalibration
def _f1(y: np.ndarray, yhat: np.ndarray) -> float:
    tp = int((y & yhat).sum())
    fp = int((~y & yhat).sum())
    fn = int((y & ~yhat).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def fit_thresholds(P_cal: np.ndarray, Y_cal: np.ndarray, grid=GRID) -> np.ndarray:
    """Per atom: the grid value maximising F1 of (p >= t) on the CALIBRATION data; ties: closest to 0.5, then the smaller; no calibration positive -> 0.5."""
    P_cal = np.asarray(P_cal, dtype=np.float64)
    Y_cal = np.asarray(Y_cal).astype(bool)
    out = np.full(P_cal.shape[1], 0.5)
    for j in range(P_cal.shape[1]):
        y = Y_cal[:, j]
        if not y.any():
            continue
        scored = [(_f1(y, P_cal[:, j] >= t), t) for t in grid]
        best = max(s for s, _ in scored)
        out[j] = min((t for s, t in scored if s >= best - 1e-12), key=lambda t: (abs(t - 0.5), t))
    return out


def _logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(x) - np.log1p(-x)


def recalibrate(P: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """q_j = sigmoid(logit p_j - logit t_j): the per-atom threshold t_j becomes the 0.5 point (t = 0.5 leaves p unchanged)."""
    P = np.asarray(P, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    if np.allclose(t, 0.5):
        return P.copy()
    return 1.0 / (1.0 + np.exp(-(_logit(P) - _logit(t))))


# ---------------------------------------------------------------------------------------------------- set decoders
def _d0(p: np.ndarray, atoms: list[str]) -> list[str]:
    idx = [j for j in range(len(atoms)) if p[j] >= 0.5]
    return [atoms[j] for j in (idx or [int(np.argmax(p))])]


def constrain(set_: list[str], p: np.ndarray, atoms: list[str]) -> list[str]:
    """D1: per opposite pair keep the higher-p member (tie: the first of the pair); then `static` vs the movement atoms by comparing p(static) with their max p."""
    ai = {a: i for i, a in enumerate(atoms)}
    s = set(set_)
    for a, b in OPPOSITE_PAIRS:
        if a in s and b in s:
            s.discard(b if p[ai[a]] >= p[ai[b]] else a)
    moves = [x for x in s if x != "static"]
    if "static" in s and moves:
        s = {"static"} if p[ai["static"]] >= max(p[ai[x]] for x in moves) else set(moves)
    return [a for a in atoms if a in s]


def decode(P: np.ndarray, atoms: list[str], candidate: str, thresholds: np.ndarray | None = None) -> list[list[str]]:
    """Decoded atom sets (in `atoms` order) of every row of P under D0 / D1 / D2 / D3 (D2, D3 need the calibration thresholds)."""
    P = np.asarray(P, dtype=np.float64)
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}")
    if candidate in ("D2", "D3"):
        if thresholds is None:
            raise ValueError(f"{candidate} needs thresholds fitted on calibration data")
        Q = recalibrate(P, thresholds)
    else:
        Q = P
    out = []
    for q in Q:
        s = _d0(q, atoms)
        if candidate in ("D1", "D3"):
            s = constrain(s, q, atoms)
        out.append(s)
    return out


# ---------------------------------------------------------------------------------------------------- option scoring (Phase 4 scorer)
def option_geometry_scores(question: dict, p: np.ndarray, atoms: list[str], *, mask: bool = False) -> np.ndarray:
    """Bernoulli set log-likelihood per letter (`baseline.option_scores`); with `mask`, contradictory options get -inf unless all four are contradictory."""
    sc = option_scores(question, np.asarray(p, dtype=np.float64), atoms)
    s = np.array([sc[L] for L in LETTERS], dtype=np.float64)
    if mask:
        bad = np.array([is_contradictory(question["option_atoms"][L])["contradictory"] for L in LETTERS])
        if not bad.all():
            s = np.where(bad, -np.inf, s)
    return s


def lsm(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def candidate_probs(P: np.ndarray, candidate: str, thresholds: np.ndarray | None) -> np.ndarray:
    return recalibrate(P, thresholds) if candidate in ("D2", "D3") else np.asarray(P, dtype=np.float64)


def option_hits(questions: list[dict], P: np.ndarray, atoms: list[str], candidate: str, thresholds: np.ndarray | None = None,
                video_logits=None, w_s: float = 0.0, w_g: float = 1.0) -> np.ndarray:
    """Hit per question; argmax with the lowest letter on ties. Without video logits: geometry only; with them: w_s * log_softmax(video) + w_g * geometry."""
    Q = candidate_probs(P, candidate, thresholds)
    mask = candidate in ("D1", "D3")
    hits = []
    for i, (q, p) in enumerate(zip(questions, Q, strict=True)):
        g = option_geometry_scores(q, p, atoms, mask=mask)
        s = g if video_logits is None else w_s * lsm(video_logits[i]) + w_g * g
        hits.append(int(np.argmax(s)) == LETTERS.index(q["gold_letter"]))
    return np.array(hits, dtype=bool)


def fit_fusion_weights(sem: list[np.ndarray], geo: list[np.ndarray], gold: list[int], grid=FUSION_GRID) -> tuple[float, float]:
    """D4: (w_s, w_g) maximising the conditional log-likelihood of the gold option on the CALIBRATION questions; ties to the smaller w_s + w_g (rule of p4_video_combine)."""
    def ll(ws, wg):
        tot = 0.0
        for s_, g_, k in zip(sem, geo, gold, strict=True):
            s = ws * np.asarray(s_) + wg * np.asarray(g_)
            s = s - s.max()
            tot += float(s[k] - math.log(np.exp(s).sum()))
        return tot
    vals = {(ws, wg): ll(ws, wg) for ws, wg in itertools.product(grid, grid)}
    best = max(vals.values())
    return min((k for k, v in vals.items() if v >= best - 1e-12), key=lambda k: (k[0] + k[1], k))


# ---------------------------------------------------------------------------------------------------- set metrics
def set_metrics(pred: list[list[str]], gold: list[tuple[str, ...]], atoms: list[str]) -> dict:
    """Exact match, mean Jaccard, per-atom P/R/F1 (macro over atoms with support > 0), contradictory share."""
    ex = [set(p) == set(g) for p, g in zip(pred, gold, strict=True)]
    jac = [len(set(p) & set(g)) / len(set(p) | set(g)) if (p or g) else 1.0 for p, g in zip(pred, gold, strict=True)]
    rows = []
    for a in atoms:
        y = np.array([a in g for g in gold])
        yh = np.array([a in p for p in pred])
        tp, fp, fn = int((y & yh).sum()), int((~y & yh).sum()), int((y & ~yh).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        rows.append({"atom": a, "support": int(y.sum()), "predicted": int(yh.sum()), "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4)})
    sup = [r for r in rows if r["support"] > 0]
    return {"n": len(pred), "exact_match": round(float(np.mean(ex)), 4), "jaccard": round(float(np.mean(jac)), 4),
            "macro_f1_supported": round(float(np.mean([r["f1"] for r in sup])), 4) if sup else None, "n_atoms_supported": len(sup),
            "contradictory_share": round(float(np.mean([is_contradictory(p)["contradictory"] for p in pred])), 4), "per_atom": rows}


def paired_bootstrap(x: np.ndarray, y: np.ndarray, sources: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Mean(x) - mean(y) with the 95% percentile interval over source resamples (as scripts/p4_video_validate.py)."""
    x, y, sources = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(sources)
    usrc = sorted(set(sources))
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([np.where(sources == s)[0] for s in rng.choice(usrc, len(usrc), replace=True)]) for _ in range(n)]
    d = [x[ix].mean() - y[ix].mean() for ix in boots]
    return {"diff": round(float(x.mean() - y.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}


def decide(acc: dict[str, float], boot_vs_d0: dict[str, dict], jaccard: dict[str, float]) -> dict:
    """Pre-registered rule: top of D1-D3 by accuracy (ties: lower index); wins iff CI lower bound > 0 and Jaccard drop <= 0.010; else D0."""
    top = max(("D1", "D2", "D3"), key=lambda c: (acc[c], -int(c[1])))
    ci_ok = boot_vs_d0[top]["ci95"][0] > 0
    jac_ok = jaccard[top] - jaccard["D0"] >= -0.010 - 1e-12
    return {"top_ranked": top, "ci_excludes_zero_above": ci_ok, "jaccard_ok": jac_ok, "winner": top if (ci_ok and jac_ok) else "D0"}
