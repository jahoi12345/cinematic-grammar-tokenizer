"""Eye-level head on frozen SigLIP features.

Everything that SELECTS (hyper-parameters, the override threshold) sees CineScale2 split_v3 TRAIN only, through grouped
cross-validation whose groups are an earlier change's near-duplicate groups. CineScale2 val is scored once by `evaluate`; ShotQA validation
angle images are scored only by `override_rate` (guard metric, never fitted on).

Pure numpy / scikit-learn; CPU only."""
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# CineScale2 angle classes in a fixed order; "neutral" is schema v2 "eye-level" (cgt.cinescale2.labels.ANGLE_TO_SCHEMA)
CLASSES = ("overhead", "high", "neutral", "low", "dutch")
EYE = CLASSES.index("neutral")

# Phase 3A embedder identity (cgt.p3_static_attrs.features.MODEL; revision and weights hash as recorded in the G1 pool)
EMBEDDER = {"model": "google/siglip-base-patch16-224", "revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
            "weights_sha256": "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"}

# ---- pre-registered constants (docs/plans/eyelevel_head.md section 3-5); changing any of them breaks the pre-registration
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CLASS_WEIGHTS = (None, "balanced")
FAMILIES = ("multinomial", "binary")
N_FOLDS, CV_SEED = 5, 0
TARGET_PRECISION = 0.80            # threshold chosen on pooled out-of-fold train scores
ADOPT_PRECISION, ADOPT_RECALL = 0.75, 0.60   # on CineScale2 val (point estimates)
ADOPT_GUARD_CAP = 0.05             # max override rate on ShotQA validation angle images
N_BOOT, BOOT_SEED = 2000, 0


@dataclass(frozen=True)
class Config:
    family: str          # "multinomial" (5 classes; eye score = P(neutral)) or "binary" (neutral vs rest)
    C: float
    class_weight: str | None

    def key(self) -> str:
        return f"{self.family}|C={self.C}|cw={self.class_weight}"


def grid() -> list[Config]:
    return [Config(f, c, w) for f in FAMILIES for c in C_GRID for w in CLASS_WEIGHTS]


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def check_embedder(stats_json: str) -> dict:
    """Raise unless a feature cache's stats name the Phase 3A embedder (model, revision, weights hash)."""
    st = json.loads(stats_json)
    bad = {k: (st.get(k), v) for k, v in EMBEDDER.items() if st.get(k) != v}
    if bad:
        raise ValueError(f"feature cache is not the Phase 3A embedder: {bad}")
    return st


# ---------------------------------------------------------------- split / groups
def angle_split(split: dict, ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """(train, val) CineScale2 ANGLE relpaths of split_v3. Train = angle ids in `ids` that are neither val nor excluded.
    Excluded ids are dropped; a val id is never in train. Raises if a val id is missing from `ids`. (The caller also checks
    train against split_v3's train_counts, so a pool that misses train ids is caught there.)"""
    excluded = set().union(*map(set, split["excluded"].values()))
    val_all = set(split["val"])
    have = set(ids)
    angle = [i for i in have if i.split("/")[-3] == "angle"]
    val = sorted(i for i in val_all if i.split("/")[-3] == "angle")
    train = sorted(i for i in angle if i not in val_all and i not in excluded)
    missing = [i for i in val if i not in have]
    if missing:
        raise ValueError(f"{len(missing)} val ids missing from the feature pool, e.g. {missing[:3]}")
    if set(train) & set(val):
        raise AssertionError("train/val overlap")
    return train, val


def label_of(relpath: str) -> int:
    return CLASSES.index(relpath.split("/")[-2])


def check_groups(group_of: dict[str, str], train: Iterable[str], val: Iterable[str]) -> None:
    """Raise if any near-duplicate group has members on both sides, or an id has no group."""
    tr = {group_of[i] for i in train}
    va = {group_of[i] for i in val}
    if tr & va:
        raise AssertionError(f"{len(tr & va)} groups span train and val")


def build_groups(hash_ids: Sequence[str], full, trim, thumbs, root_marker: str = "/unpacked/") -> dict[str, str]:
    """Recompute an earlier change's CineScale2-internal near-duplicate groups (scripts/cs2_prepare.py: dHash candidate <= 8, same-frame
    thumbnail check, flat frames give no edges) from the cached hashes. Keys are relpaths under the unpacked root."""
    from cgt.cinescale2 import contamination as C
    from cgt.cinescale2 import split as S
    rel = [h.split(root_marker, 1)[1] for h in hash_ids]
    ip = C.candidate_pairs(full, trim, full, trim, 8, same_set=True)
    same, _, _ = C.same_frame(thumbs, thumbs, ip)
    flat = C.is_flat(thumbs)
    keep = same & ~flat[ip[:, 0]] & ~flat[ip[:, 1]] if len(ip) else np.zeros(0, bool)
    return S.groups(rel, [(int(i), int(j)) for i, j in ip[keep, :2]])


# ---------------------------------------------------------------- model
def make_model(cfg: Config):
    return make_pipeline(StandardScaler(), LogisticRegression(C=cfg.C, class_weight=cfg.class_weight, max_iter=3000))


def fit(cfg: Config, x: np.ndarray, y: np.ndarray):
    """Fit one config. Binary family: target = (y == neutral)."""
    m = make_model(cfg)
    m.fit(x, (y == EYE).astype(int) if cfg.family == "binary" else y)
    return m


def eye_score(cfg: Config, model, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    cls = list(model.classes_)
    return p[:, cls.index(1 if cfg.family == "binary" else EYE)]


def class_probs(model, x: np.ndarray) -> np.ndarray:
    """(N, 5) probabilities in CLASSES order (multinomial family only)."""
    p = model.predict_proba(x)
    out = np.zeros((len(x), len(CLASSES)), np.float64)
    out[:, list(model.classes_)] = p
    return out


def folds(y: np.ndarray, groups: Sequence[str], n_folds: int = N_FOLDS, seed: int = CV_SEED) -> list[np.ndarray]:
    """Fold index per sample (StratifiedGroupKFold, shuffled, seeded): a group never spans two folds."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        fold[te] = k
    return fold


def oof(cfg: Config, x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Out-of-fold (eye score, 5-class probs or None) for one config."""
    eye = np.zeros(len(y))
    probs = np.zeros((len(y), len(CLASSES))) if cfg.family == "multinomial" else None
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        m = fit(cfg, x[tr], y[tr])
        eye[te] = eye_score(cfg, m, x[te])
        if probs is not None:
            probs[te] = class_probs(m, x[te])
    return eye, probs


def eye_ap(is_eye: np.ndarray, score: np.ndarray) -> float:
    """Average precision of the eye-level score."""
    return float(average_precision_score(np.asarray(is_eye, bool), score))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


# ---------------------------------------------------------------- selection (TRAIN ONLY)
def _assert_train_only(ids: Sequence[str], val_ids: Iterable[str]) -> None:
    v = set(val_ids)
    if not v:
        raise ValueError("val_ids must be given (non-empty) so the train-only rule can be checked")
    bad = [i for i in ids if i in v]
    if bad:
        raise ValueError(f"selection must use CineScale2 train only; {len(bad)} val ids passed, e.g. {bad[:3]}")


def select_configs(results: dict[str, dict], ids: Sequence[str], val_ids: Iterable[str]) -> tuple[str, str]:
    """(5-class head key, eye-level detector key) from out-of-fold train results {key: {'family', 'C', 'cw',
    'balanced_acc' (multinomial only), 'eye_ap'}}. 5-class head: max OOF balanced accuracy among multinomial configs.
    Detector: max OOF eye-level average precision among all configs. Ties: smaller C, then multinomial, then cw None."""
    _assert_train_only(ids, val_ids)

    def tie(k):
        r = results[k]
        return (r["C"], 0 if r["family"] == "multinomial" else 1, 0 if r["cw"] is None else 1)
    multi = [k for k, r in results.items() if r["family"] == "multinomial"]
    head = max(multi, key=lambda k: (round(results[k]["balanced_acc"], 6), tuple(-t for t in tie(k))))
    det = max(results, key=lambda k: (round(results[k]["eye_ap"], 6), tuple(-t for t in tie(k))))
    return head, det


def select_threshold(ids: Sequence[str], scores: np.ndarray, is_eye: np.ndarray, val_ids: Iterable[str],
                     target: float = TARGET_PRECISION) -> dict:
    """Override threshold from out-of-fold TRAIN scores: the lowest score t such that precision of (score >= t) >= target
    (max recall at that precision). If no t reaches the target: the max-F1 threshold, with reached=False.
    Raises ValueError if any id is a CineScale2 val id."""
    _assert_train_only(ids, val_ids)
    scores, is_eye = np.asarray(scores, float), np.asarray(is_eye, bool)
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], is_eye[order]
    tp = np.cumsum(t)
    n_pos = np.arange(1, len(s) + 1)
    # only cut where the score changes (ties go together)
    last = np.r_[s[1:] != s[:-1], True]
    prec, rec = (tp / n_pos)[last], (tp / max(t.sum(), 1))[last]
    thr = s[last]
    ok = np.nonzero(prec >= target)[0]
    if len(ok):
        k = ok[np.argmax(rec[ok])]   # max recall; among equal recall the highest threshold (fewest flags)
        return {"threshold": float(thr[k]), "train_oof_precision": float(prec[k]), "train_oof_recall": float(rec[k]),
                "reached_target": True, "target_precision": target}
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    return {"threshold": float(thr[k]), "train_oof_precision": float(prec[k]), "train_oof_recall": float(rec[k]),
            "reached_target": False, "target_precision": target}


# ---------------------------------------------------------------- evaluation
def eye_prf(is_eye: np.ndarray, flagged: np.ndarray) -> dict:
    is_eye, flagged = np.asarray(is_eye, bool), np.asarray(flagged, bool)
    tp = int((is_eye & flagged).sum())
    p = tp / flagged.sum() if flagged.sum() else 0.0
    r = tp / is_eye.sum() if is_eye.sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f), "tp": tp, "flagged": int(flagged.sum()),
            "positives": int(is_eye.sum()), "n": len(is_eye)}


def five_class(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"accuracy": float(np.mean(pred == y)), "balanced_accuracy": balanced_accuracy(y, pred),
            "recall": {c: float(np.mean(pred[y == i] == i)) for i, c in enumerate(CLASSES) if (y == i).any()}}


def group_bootstrap(groups: Sequence[str], stat, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """stat(index array) over `n_boot` resamples of whole groups with replacement. Returns the (n_boot,) statistics."""
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.nonzero(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = stat(np.concatenate([members[k] for k in pick]))
    return out


def ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def override_rate(flagged: np.ndarray) -> float:
    return float(np.mean(np.asarray(flagged, bool))) if len(flagged) else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


# ---------------------------------------------------------------- adoption rule (fixed in the plan, section 5)
def adoption(val_precision: float, val_recall: float, guard_rate: float, threshold_reached: bool = True) -> tuple[bool, list[str]]:
    """Adopt only if (a) the train-CV threshold reached the target precision, (b) CineScale2 val eye-level precision
    >= 0.75 and (c) recall >= 0.60 (point estimates), and (d) the ShotQA validation override rate <= 0.05."""
    reasons = []
    if not threshold_reached:
        reasons.append(f"train-CV threshold did not reach precision {TARGET_PRECISION}")
    if not val_precision >= ADOPT_PRECISION:
        reasons.append(f"val eye-level precision {val_precision:.3f} < {ADOPT_PRECISION}")
    if not val_recall >= ADOPT_RECALL:
        reasons.append(f"val eye-level recall {val_recall:.3f} < {ADOPT_RECALL}")
    if not guard_rate <= ADOPT_GUARD_CAP:
        reasons.append(f"ShotQA override rate {guard_rate:.4f} > {ADOPT_GUARD_CAP}")
    return not reasons, reasons
