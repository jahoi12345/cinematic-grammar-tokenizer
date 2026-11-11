"""Option-restricted and full-vocabulary metrics (docs/phase3_interface.md section 5). Pure numpy."""
import numpy as np


def _log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def multihot_option_scores(logits: np.ndarray, option_idx: list[list[int]]) -> np.ndarray:
    """Set likelihood under independent Bernoullis: sum_{c in S} log s(z_c) + sum_{c not in S} log(1 - s(z_c))."""
    ls, lns = _log_sigmoid(logits), _log_sigmoid(-logits)
    base = lns.sum()
    return np.array([base + sum(ls[c] - lns[c] for c in o) for o in option_idx])


def categorical_option_scores(probs: np.ndarray, option_idx: list[list[int]]) -> np.ndarray:
    """log of the mean class probability over the option's members (single option = log p(c))."""
    return np.array([np.log(max(float(np.mean(probs[o])), 1e-12)) for o in option_idx])


def pick_option(scores: np.ndarray, letters: list[str]) -> str:
    """Highest score; ties resolve to the lowest letter."""
    best = np.max(scores)
    return min(letters[i] for i in np.flatnonzero(scores == best))


def option_restricted_hits(examples, rows_logits, vocab: list[str], kind: str = "multihot") -> np.ndarray:
    """Boolean array (n_examples,): True where the highest-scoring option is the answer letter (see option_restricted_accuracy)."""
    index = {v: i for i, v in enumerate(vocab)}
    hits = []
    for ex, row in zip(examples, rows_logits, strict=True):
        letters = sorted(ex.options)
        opt = [[index[v] for v in ex.options[ltr]] for ltr in letters]
        sc = multihot_option_scores(row, opt) if kind == "multihot" else categorical_option_scores(row, opt)
        hits.append(pick_option(sc, letters) == ex.answer_letter)
    return np.array(hits, dtype=bool)


def option_restricted_accuracy(examples, rows_logits, vocab: list[str], kind: str = "multihot") -> float:
    """examples: sequence of labels.Example; rows_logits[i]: logits (multihot) or probs (categorical) for example i."""
    hits = option_restricted_hits(examples, rows_logits, vocab, kind)
    return float(hits.sum() / max(len(examples), 1))


def hit_at_1(scores: np.ndarray, targets: np.ndarray) -> float:
    """scores, targets: (N, K); fraction whose argmax class is in the true label set."""
    return float(targets[np.arange(len(scores)), scores.argmax(1)].mean())


def multilabel_f1(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (1 / (1 + np.exp(-logits))) >= threshold
    t = targets.astype(bool)
    tp, fp, fn = (pred & t).sum(), (pred & ~t).sum(), (~pred & t).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"micro_f1": float(f1), "exact_set_acc": float((pred == t).all(1).mean())}


def ordinal_mae(pred_idx: np.ndarray, true_idx: np.ndarray) -> float:
    return float(np.abs(pred_idx - true_idx).mean())


def confusion(pred_idx: np.ndarray, true_idx: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (true_idx, pred_idx), 1)
    return m
