"""Pure helpers for the descriptive 4B error analysis. Aggregates only; no fitting, tuning or selection.

Records are dicts with at least: `gold` (frozenset of canonical labels of the gold option), `pred` (frozenset of the chosen
option), `gold_text` / `pred_text` (option strings), `film`, `correct` (bool). Multi-label options (e.g. "Hard light, Side light")
are analysed per component label: a label's recall is the share of questions whose gold option contains it where the chosen
option also contains it; its precision is the share of questions whose chosen option contains it where the gold option does too."""
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np


def gold_rank(scores: Sequence[float], gold: int) -> int:
    """1-based rank of the gold option under argmax with ties to the lowest index (the project's pick_option convention)."""
    s = np.asarray(scores, dtype=float)
    g = s[gold]
    return 1 + int(np.sum(s > g)) + int(np.sum(s[:gold] == g))


def component_prf(records: Iterable[dict], labels: Sequence[str]) -> dict:
    """{label: {support, predicted, recall, precision}} per component label (None when undefined)."""
    sup, prd, tp = Counter(), Counter(), Counter()
    for r in records:
        for c in r["gold"]:
            sup[c] += 1
        for c in r["pred"]:
            prd[c] += 1
        for c in r["gold"] & r["pred"]:
            tp[c] += 1
    return {c: {"support": sup[c], "predicted": prd[c], "recall": tp[c] / sup[c] if sup[c] else None,
                "precision": tp[c] / prd[c] if prd[c] else None} for c in labels}


def component_confusions(records: Iterable[dict], top: int = 10) -> list[dict]:
    """On wrong answers: each gold label the chosen option misses (gold - pred) paired with each label the chosen option adds
    (pred - gold); counts of (missed -> added). Pairs with an empty side are counted as (c -> '(none)') / ('(none)' -> d)."""
    cnt = Counter()
    for r in records:
        if r["correct"]:
            continue
        miss, add = sorted(r["gold"] - r["pred"]) or ["(none)"], sorted(r["pred"] - r["gold"]) or ["(none)"]
        for a in miss:
            for b in add:
                cnt[(a, b)] += 1
    return [{"gold": a, "pred": b, "count": n} for (a, b), n in cnt.most_common(top)]


def option_confusions(records: Iterable[dict], top: int = 10) -> list[dict]:
    cnt = Counter((r["gold_text"], r["pred_text"]) for r in records if not r["correct"])
    return [{"gold": a, "pred": b, "count": n} for (a, b), n in cnt.most_common(top)]


def film_concentration(films: Sequence[str], correct: Sequence[bool], min_q: int = 10) -> dict:
    """How concentrated errors are across films: share of errors in the k films with most errors vs their share of questions,
    spread of per-film accuracy (films with >= min_q questions), and the dispersion ratio of per-film error counts against a
    binomial with the pooled error rate (1 = no more film-to-film variation than chance)."""
    n, e = Counter(films), Counter(f for f, c in zip(films, correct, strict=True) if not c)
    tot_q, tot_e = len(films), sum(e.values())
    order = sorted(n, key=lambda f: (-e[f], -n[f], f))
    out = {"films": len(n), "questions": tot_q, "errors": tot_e, "films_with_errors": sum(1 for f in n if e[f])}
    for k in (5, 10):
        top = order[:k]
        out[f"top{k}_films_error_share"] = sum(e[f] for f in top) / tot_e if tot_e else None
        out[f"top{k}_films_question_share"] = sum(n[f] for f in top) / tot_q
    # films needed to cover half of the errors
    acc, half = 0, None
    for i, f in enumerate(order, 1):
        acc += e[f]
        if tot_e and acc >= tot_e / 2:
            half = i
            break
    out["films_holding_half_of_errors"] = half
    big = [f for f in n if n[f] >= min_q]
    a = np.array([1 - e[f] / n[f] for f in big]) if big else np.array([])
    out[f"films_ge_{min_q}_q"] = len(big)
    out["per_film_accuracy"] = ({"min": float(a.min()), "p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
                                 "p90": float(np.percentile(a, 90)), "max": float(a.max())} if len(a) else None)
    p = tot_e / tot_q if tot_q else 0.0
    if 0 < p < 1:
        chi = sum((e[f] - n[f] * p) ** 2 / (n[f] * p * (1 - p)) for f in n)
        out["dispersion_ratio"] = chi / max(len(n) - 1, 1)
    else:
        out["dispersion_ratio"] = None
    return out


def auc(x: Sequence[float], pos: Sequence[bool]) -> float | None:
    """Mann-Whitney AUC: P(x of a positive > x of a negative), ties count half. None if a group is empty."""
    x, pos = np.asarray(x, float), np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def accuracy_by_quantile(x: Sequence[float], correct: Sequence[bool], q: int = 5) -> list[dict]:
    """Accuracy in q equal-count bins of x (bin edges = empirical quantiles; descriptive)."""
    x, c = np.asarray(x, float), np.asarray(correct, bool)
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, q - 1)
    return [{"bin": b + 1, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int((idx == b).sum()),
             "accuracy": float(c[idx == b].mean()) if (idx == b).any() else None} for b in range(q)]


def image_stats(rgb: np.ndarray) -> dict:
    """Cheap global statistics of an RGB uint8 image (H, W, 3). Luma = Rec.601 on [0, 1].
    mean_luma, rms_contrast (std of luma), p95_p5 (luma 95th - 5th percentile), dark_frac (luma < 0.1), bright_frac (> 0.9),
    saturation (mean HSV saturation), warmth (mean R - mean B), sharpness (variance of a 4-neighbour Laplacian of luma),
    grad_cx (x of the gradient-magnitude centroid, 0 = left, 1 = right), lr_balance (right-half minus left-half share of
    gradient mass, -1..1), aspect (W / H)."""
    f = rgb.astype(np.float64) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = f.max(axis=2), f.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0.0)
    lap = (-4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:])
    gx = np.abs(np.diff(y, axis=1))[:-1, :]
    gy = np.abs(np.diff(y, axis=0))[:, :-1]
    gm = gx + gy
    w = gm.shape[1]
    xs = (np.arange(w) + 0.5) / w
    tot = gm.sum()
    col = gm.sum(axis=0)
    cx = float((col * xs).sum() / tot) if tot > 0 else 0.5
    lr = float((col[xs >= 0.5].sum() - col[xs < 0.5].sum()) / tot) if tot > 0 else 0.0
    p5, p95 = np.percentile(y, [5, 95])
    return {"mean_luma": float(y.mean()), "rms_contrast": float(y.std()), "p95_p5": float(p95 - p5),
            "dark_frac": float((y < 0.1).mean()), "bright_frac": float((y > 0.9).mean()), "saturation": float(sat.mean()),
            "warmth": float(r.mean() - b.mean()), "sharpness": float(lap.var()), "grad_cx": cx, "lr_balance": lr,
            "aspect": float(rgb.shape[1] / rgb.shape[0])}


def group_medians(values: Sequence[float], groups: Sequence[str]) -> dict:
    d = defaultdict(list)
    for v, k in zip(values, groups, strict=True):
        d[k].append(v)
    return {k: {"n": len(v), "median": float(np.median(v))} for k, v in sorted(d.items())}
