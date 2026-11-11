"""Combining cached views into candidate scores, calibration (C4), ensemble with Phase 3A (C5), the 13 ranked configurations and the
film-clustered paired bootstrap (plan sections 3-6). Pure numpy/scipy; no model, no file access.

Data layout: `qs` = list of question dicts (key, attribute, half, film, options, answer); `views[name][key]` = 4 letter logits in that view's
letter order; `sig[key]` = 4 Phase 3A option scores in the original order. Gold (`answer`) is read ONLY by `gold_index` and only for
questions handed to a fit function or to scoring."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from cgt.eval.metrics import pick_option
from cgt.p3_static_attrs.labels import parse_label_set, vocab
from cgt.p3s_stepa.split import ATTRS
from cgt.p3s_stepa.views import LETTERS, VIEWS, applies, to_original

L2 = 1e-2
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def gold_index(q: dict) -> int:
    return LETTERS.index(q["answer"])


def view_logprobs(views: dict, view: str, key: str) -> np.ndarray:
    """Option log-probs of one view in the ORIGINAL option order."""
    return to_original(log_softmax(np.asarray(views[view][key], dtype=float)), VIEWS[view]["shift"])


def averaged(qs: list[dict], views: dict, view_set: list[str]) -> dict[str, np.ndarray]:
    """{key: mean over the views in `view_set` that apply to the question's attribute of option log-probs}. A view that applies but has
    no output for a question raises KeyError (no silent fallback)."""
    out = {}
    for q in qs:
        use = [v for v in view_set if applies(v, q["attribute"])]
        if not use:
            raise ValueError(f"no view applies to {q['attribute']}")
        out[q["key"]] = np.mean([view_logprobs(views, v, q["key"]) for v in use], axis=0)
    return out


# ---------- C4 calibration ----------
def _label_matrix(q: dict, labs: list[str]) -> np.ndarray:
    """(4, n_labels): row i = indicator of the labels of option i divided by their count (mean over the option's label set)."""
    m = np.zeros((4, len(labs)))
    idx = {lab: i for i, lab in enumerate(labs)}
    for i, x in enumerate(LETTERS):
        ls = parse_label_set(q["attribute"], q["options"][x])
        for lab in ls:
            m[i, idx[lab]] = 1.0 / len(ls)
    return m


def _design(q: dict, kind: str) -> np.ndarray:
    if kind == "letter":
        return np.eye(4)[:, 1:]  # b_A fixed to 0
    if kind == "label":
        return _label_matrix(q, vocab(q["attribute"]))
    raise ValueError(kind)


def calibrated(s: np.ndarray, params: np.ndarray, design: np.ndarray) -> np.ndarray:
    """s / T + design @ beta, params = [log T, beta...]."""
    return s / np.exp(params[0]) + design @ params[1:]


def fit_calibration(fit_qs: list[dict], scores: dict, kind: str) -> np.ndarray:
    """Per-attribute parameters minimising mean NLL + L2 * |beta|^2 over `fit_qs` (all one attribute). Only `fit_qs` are read."""
    S = np.array([scores[q["key"]] for q in fit_qs])
    D = np.array([_design(q, kind) for q in fit_qs])
    y = np.array([gold_index(q) for q in fit_qs])
    n_beta = D.shape[2]

    def obj(p):
        z = S / np.exp(p[0]) + np.einsum("nij,j->ni", D, p[1:])
        lp = log_softmax(z, axis=1)
        return -lp[np.arange(len(y)), y].mean() + L2 * float(p[1:] @ p[1:])

    r = minimize(obj, np.zeros(1 + n_beta), method="L-BFGS-B", bounds=[(-3, 3)] + [(None, None)] * n_beta)
    return r.x


def apply_calibration(qs: list[dict], scores: dict, fit_qs: list[dict], kind: str) -> tuple[dict, dict]:
    """Fit per attribute on `fit_qs`, apply to every question in `qs`. Returns (new scores, params per attribute)."""
    params, out = {}, {}
    for a in ATTRS:
        fq = [q for q in fit_qs if q["attribute"] == a]
        if fq:
            params[a] = fit_calibration(fq, scores, kind)
    for q in qs:
        out[q["key"]] = calibrated(scores[q["key"]], params[q["attribute"]], _design(q, kind))
    return out, {a: [round(float(x), 4) for x in p] for a, p in params.items()}


def nll(qs: list[dict], scores: dict) -> float:
    return float(np.mean([-log_softmax(scores[q["key"]])[gold_index(q)] for q in qs]))


def crossfit_pick_c4(cal_qs: list[dict], scores: dict, seed: int = 1) -> tuple[str, dict]:
    """'letter' or 'label': lower mean held-out NLL in a 2-fold film cross-fit INSIDE the calibration half."""
    films = sorted({q["film"] for q in cal_qs})
    perm = np.random.default_rng(seed).permutation(len(films))
    fold = {films[i]: (n % 2) for n, i in enumerate(perm)}
    res = {}
    for kind in ("letter", "label"):
        vals = []
        for k in (0, 1):
            tr = [q for q in cal_qs if fold[q["film"]] != k]
            te = [q for q in cal_qs if fold[q["film"]] == k]
            sc, _ = apply_calibration(te, scores, tr, kind)
            vals.append(nll(te, sc))
        res[kind] = float(np.mean(vals))
    return min(res, key=lambda k: (res[k], k != "letter")), res


# ---------- C5 ensemble ----------
def fit_ensemble_w(fit_qs: list[dict], vlm: dict, sig: dict) -> dict[str, float]:
    """Per attribute, w in W_GRID with the lowest NLL of (1-w) VLM + w 3A option log-probs on `fit_qs` (ties: smaller w)."""
    out = {}
    for a in ATTRS:
        fq = [q for q in fit_qs if q["attribute"] == a]
        if not fq:
            continue
        best = None
        for w in W_GRID:
            v = nll(fq, mix({q["key"]: vlm[q["key"]] for q in fq}, sig, {a: w}, fq))
            if best is None or v < best[0] - 1e-12:
                best = (v, float(w))
        out[a] = best[1]
    return out


def mix(vlm: dict, sig: dict, w: dict, qs: list[dict]) -> dict:
    return {q["key"]: (1 - w[q["attribute"]]) * log_softmax(vlm[q["key"]]) + w[q["attribute"]] * log_softmax(np.asarray(sig[q["key"]], float))
            for q in qs}


# ---------- configurations ----------
BASE_RES = ["base", "shift1", "shift2", "shift3", "hflip", "crop80"]
CONFIGS = {  # name -> (views, calibration: None | 'letter' | 'label' | 'C4*', ensemble)
    "C0": (["base"], None, False),
    "C1": (["base", "shift1", "shift2", "shift3"], None, False),
    "C2a": (["res262k"], None, False),
    "C2b": (["res590k"], None, False),
    "C3": (["base", "hflip", "crop80"], None, False),
    "C4a": (["base"], "letter", False),
    "C4b": (["base"], "label", False),
    "C5": (["base"], None, True),
    "C1+C3": (BASE_RES, None, False),
    "C1+C4*": (["base", "shift1", "shift2", "shift3"], "C4*", False),
    "C1+C4*+C5": (["base", "shift1", "shift2", "shift3"], "C4*", True),
    "C1+C3+C4*+C5": (BASE_RES, "C4*", True),
    "C1+C2*+C3+C4*+C5": (BASE_RES + ["C2*"], "C4*", True),
}
ORDER = list(CONFIGS)


def accuracy(qs: list[dict], scores: dict) -> dict:
    """Per-attribute option-restricted accuracy and the unweighted 7-attribute mean (ties to the lowest letter)."""
    per = {}
    for a in ATTRS:
        r = [q for q in qs if q["attribute"] == a]
        if r:
            per[a] = float(np.mean([pick_option(np.asarray(scores[q["key"]]), list(LETTERS)) == q["answer"] for q in r]))
    return {"per_attribute": per, "mean": float(np.mean(list(per.values())))}


def hits(qs: list[dict], scores: dict) -> np.ndarray:
    return np.array([pick_option(np.asarray(scores[q["key"]]), list(LETTERS)) == q["answer"] for q in qs])


def run_config(name: str, qs: list[dict], views: dict, sig: dict | None) -> dict:
    """Scores for ALL `qs` under configuration `name`; every fitted quantity uses only questions with half == 'cal'."""
    vset, cal, ens = CONFIGS[name]
    cal_qs = [q for q in qs if q["half"] == "cal"]
    info = {}
    if "C2*" in vset:
        c2 = {v: accuracy(cal_qs, averaged(cal_qs, views, [v]))["mean"] for v in ("res262k", "res590k")}
        pick = max(("res262k", "res590k"), key=lambda v: (c2[v], v == "res262k"))
        info["C2*"] = {"picked": pick, "cal_mean": c2}
        vset = [pick if v == "C2*" else v for v in vset]
    scores = averaged(qs, views, vset)
    if cal == "C4*":
        cal, cv = crossfit_pick_c4(cal_qs, scores)
        info["C4*"] = {"picked": cal, "crossfit_nll": cv}
    if cal:
        scores, info["calibration_params"] = apply_calibration(qs, scores, cal_qs, cal)
    if ens:
        w = fit_ensemble_w(cal_qs, scores, sig)
        info["ensemble_w"] = w
        scores = mix(scores, sig, w, qs)
    info["views"] = vset
    return {"scores": scores, "info": info}


# ---------- bootstrap ----------
def _mean_from_counts(correct: np.ndarray, n: np.ndarray) -> np.ndarray:
    return (correct / np.maximum(n, 1)).mean(axis=-1)


def paired_bootstrap(qs: list[dict], hits_a: np.ndarray, hits_b: np.ndarray, cluster: str = "film", n_boot: int = 10_000, seed: int = 0) -> dict:
    """Difference (b - a) of 7-attribute mean accuracy, in points, with a percentile 95% interval. cluster='film' resamples films with
    replacement (each resample recomputes per-attribute accuracies from the resampled films' questions); cluster='question' resamples
    questions within each attribute."""
    attrs = sorted({q["attribute"] for q in qs}, key=ATTRS.index)
    ai = np.array([attrs.index(q["attribute"]) for q in qs])
    ha, hb = hits_a.astype(float), hits_b.astype(float)

    def stat(wt):  # wt: (n_q,) weights
        n = np.array([wt[ai == k].sum() for k in range(len(attrs))])
        return 100 * (_mean_from_counts(np.array([(wt * hb)[ai == k].sum() for k in range(len(attrs))]), n)
                      - _mean_from_counts(np.array([(wt * ha)[ai == k].sum() for k in range(len(attrs))]), n))

    point = float(stat(np.ones(len(qs))))
    rng = np.random.default_rng(seed)
    if cluster == "film":
        films = sorted({q["film"] for q in qs})
        fi = np.array([films.index(q["film"]) for q in qs])
        # per-film per-attribute sums, then each resample is a film-count vector
        A = np.zeros((len(films), len(attrs), 3))
        for k in range(len(qs)):
            A[fi[k], ai[k]] += (1, ha[k], hb[k])
        cnt = rng.multinomial(len(films), np.full(len(films), 1 / len(films)), size=n_boot)  # (B, F)
        S = np.einsum("bf,fak->bak", cnt, A)
        d = 100 * ((S[:, :, 2] / np.maximum(S[:, :, 0], 1)).mean(1) - (S[:, :, 1] / np.maximum(S[:, :, 0], 1)).mean(1))
    elif cluster == "question":
        d = np.zeros(n_boot)
        for k in range(len(attrs)):
            idx = np.flatnonzero(ai == k)
            pick = rng.integers(0, len(idx), size=(n_boot, len(idx)))
            d += (hb[idx][pick].mean(1) - ha[idx][pick].mean(1))
        d = 100 * d / len(attrs)
    else:
        raise ValueError(cluster)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff_points": point, "ci95": [float(lo), float(hi)], "cluster": cluster, "n_boot": n_boot, "seed": seed}


def decide(sel_means: dict[str, float], boot_vs_c0: dict[str, dict], available: list[str], min_gain: float = 0.5) -> dict:
    """Plan section 6: the top available configuration (ties to the earlier one in ORDER) is adopted iff gain >= min_gain points and the
    film-bootstrap interval excludes 0 (lower bound > 0); else C0. Only the top one is tested."""
    ranked = sorted(available, key=lambda n: (-sel_means[n], ORDER.index(n)))
    top = ranked[0]
    if top == "C0":
        return {"top": top, "winner": "C0", "reason": "C0 is the top configuration"}
    gain = 100 * (sel_means[top] - sel_means["C0"])
    lo = boot_vs_c0[top]["ci95"][0]
    ok = gain >= min_gain and lo > 0
    return {"top": top, "gain_points": gain, "ci95": boot_vs_c0[top]["ci95"], "winner": top if ok else "C0",
            "reason": "gain >= 0.5 and interval excludes 0" if ok else f"rule not met (gain {gain:.2f}, lower bound {lo:.2f}): keep C0"}
