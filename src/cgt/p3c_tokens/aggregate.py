"""Pure numpy aggregation of 4-option answers into a full-vocabulary score vector (docs/phase3c_plan.md sections 2, 3, 5). No model, no I/O."""
import math
from collections.abc import Callable

import numpy as np

ALPHAS = (0.25, 0.5, 0.75, 1.0)


def log_softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - (m + math.log(np.exp(x - m).sum()))


def probs(s) -> np.ndarray:
    return np.exp(log_softmax(s))


def make_windows(n: int, r: float) -> list[list[int]]:
    """Plan section 2 (A): W = max(1, ceil(n*r/4)) windows of 4 consecutive labels (circular, schema order) starting at floor(k*n/W);
    option order rotated left by (k mod 4). n must be >= 4. For n == 4 there is exactly one window."""
    if n < 4:
        raise ValueError("need at least 4 labels")
    w = 1 if n == 4 else max(1, math.ceil(n * r / 4))
    out = []
    for k in range(w):
        start = (k * n) // w
        idx = [(start + j) % n for j in range(4)]
        rot = k % 4
        out.append(idx[rot:] + idx[:rot])
    return out


def agg_mean(n: int, windows: list[list[int]], logq: list[np.ndarray]) -> np.ndarray:
    """m: s_l = mean over windows containing l of log q_w(l). Every label must occur in at least one window."""
    tot, cnt = np.zeros(n), np.zeros(n)
    for w, lq in zip(windows, logq):
        for pos, lab in enumerate(w):
            tot[lab] += lq[pos]
            cnt[lab] += 1
    if (cnt == 0).any():
        raise ValueError("a label is in no window")
    return tot / cnt


def agg_bt(n: int, windows: list[list[int]], logq: list[np.ndarray], steps: int = 300, lr: float = 0.5, l2: float = 1e-3) -> np.ndarray:
    """b: theta minimising sum_w KL(q_w || softmax_{l in w}(theta_l)) + l2*|theta|^2 by gradient descent from zero (mean-centred result)."""
    theta = np.zeros(n)
    q = [np.exp(np.asarray(lq, dtype=np.float64)) for lq in logq]
    q = [x / x.sum() for x in q]
    for _ in range(steps):
        g = 2 * l2 * theta
        for w, qw in zip(windows, q):
            p = np.exp(log_softmax(theta[w]))
            np.add.at(g, w, p - qw)
        theta = theta - lr * g
    return theta - theta.mean()


def tournament(n: int, ask: Callable[[list[int]], np.ndarray]) -> tuple[np.ndarray, int]:
    """B: knockout. `ask(window)` -> log q over the 4 options in that order. Returns (score vector, number of asks).
    Round 1: make_windows(n, 1) (last window wrapped by the circular rule). Final window: distinct winners, padded to 4 with the
    best-q round-1 runners-up not yet included. Finalists score log q_final, the others -100 + their best round-1 log q."""
    wins = make_windows(n, 1)
    calls = 0
    r1: dict[int, float] = {}  # best round-1 log q per label
    winners: list[int] = []
    runners: list[tuple[float, int]] = []
    for w in wins:
        lq = np.asarray(ask(w), dtype=np.float64)
        calls += 1
        order = np.argsort(-lq, kind="stable")
        for pos, lab in enumerate(w):
            r1[lab] = max(r1.get(lab, -1e30), float(lq[pos]))
        win = w[int(order[0])]
        if win not in winners:
            winners.append(win)
        runners += [(float(lq[p]), w[int(p)]) for p in order[1:]]
    s = np.array([-100.0 + r1[i] for i in range(n)])
    if len(winners) == 1:
        s[winners[0]] = 0.0
        return s, calls
    final = list(winners)
    for _, lab in sorted(runners, key=lambda t: -t[0]):
        if len(final) >= 4:
            break
        if lab not in final:
            final.append(lab)
    final = final[:4]
    lq = log_softmax(np.asarray(ask(final), dtype=np.float64))
    calls += 1
    for pos, lab in enumerate(final):
        s[lab] = lq[pos]
    # the argmax must be the final's argmax even if an unlucky tie occurred: finalists are always above the -100 floor
    return s, calls


def top_and_margin(s) -> tuple[float, float]:
    p = np.sort(probs(s))[::-1]
    return float(p[0]), float(p[0] - p[1])


def label_set(s, alpha: float) -> list[int]:
    """Plan section 3: indices with p_l >= alpha * p_1, always including the argmax; ordered by decreasing score."""
    p = probs(s)
    order = np.argsort(-p, kind="stable")
    keep = [int(i) for i in order if p[i] >= alpha * p[order[0]]]
    return keep if keep else [int(order[0])]


def set_f1(pred: set, gold: set) -> float:
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    pr, rc = inter / len(pred), inter / len(gold)
    return 2 * pr * rc / (pr + rc)


def option_scores(s, options: dict[str, list[int]]) -> dict[str, float]:
    """Option score = mean of s over its member label indices (compound options average their members)."""
    return {k: float(np.mean([s[i] for i in v])) for k, v in options.items()}


def pick_alpha(scores: list[np.ndarray], golds: list[set], names: list[str] | None = None) -> tuple[float, float]:
    """Alpha in ALPHAS with the best mean set-F1 (ties: larger alpha). Returns (alpha, f1). `names` maps score indices to the label
    names used in `golds` (None: golds are index sets)."""
    best = (None, -1.0)

    def pred(s, a):
        idx = label_set(s, a)
        return {names[i] for i in idx} if names is not None else set(idx)
    for a in sorted(ALPHAS, reverse=True):
        f1 = float(np.mean([set_f1(pred(s, a), g) for s, g in zip(scores, golds)]))
        if f1 > best[1] + 1e-12:
            best = (a, f1)
    return best
