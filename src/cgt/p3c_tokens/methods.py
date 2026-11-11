"""The candidate full-vocabulary scoring methods of docs/phase3c_plan.md section 2. Each returns (score vector over schema order, forward passes)."""
import numpy as np

from cgt.p3c_tokens import aggregate as ag
from cgt.p3c_tokens.vocab import DISPLAY, free_prompt, labels, option_prompt

METHODS = ["A2m", "A3m", "A2b", "A3b", "B", "C", "D"]
PASS_RANK = {"D": 0, "B": 1, "A2m": 2, "A2b": 2, "A3m": 3, "A3b": 3, "C": 4}  # plan section 6 tie order (fewer passes first)


def _window_logq(engine, st, attr: str, labs: list[str], window: list[int]) -> np.ndarray:
    lg = engine.letter_logits(st, option_prompt(attr, [labs[i] for i in window]), 4)
    return ag.log_softmax(lg)


def score_attribute(engine, st, attr: str, method: str) -> tuple[np.ndarray, int]:
    labs = labels(attr)
    n = len(labs)
    before = engine.passes
    if method[0] == "A":
        r = int(method[1])
        wins = ag.make_windows(n, r)
        logq = [_window_logq(engine, st, attr, labs, w) for w in wins]
        s = ag.agg_mean(n, wins, logq) if method[2] == "m" else ag.agg_bt(n, wins, logq)
    elif method == "B":
        s, _ = ag.tournament(n, lambda w: _window_logq(engine, st, attr, labs, w))
    elif method == "C":
        user = free_prompt(attr)
        s = np.array([engine.completion_loglik(st, user, DISPLAY[attr][lab]) for lab in labs])
    elif method == "D":
        s = ag.log_softmax(engine.letter_logits(st, option_prompt(attr, labs), n))
    else:
        raise ValueError(method)
    return np.asarray(s, dtype=np.float64), engine.passes - before
