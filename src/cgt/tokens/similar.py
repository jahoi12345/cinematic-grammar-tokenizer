"""Similar shots within one film by token row (PLAN.md Phase 6 stretch). Definition: docs/similar_shots_spec.md (fixed before this code).
Pure numpy/pandas, no I/O except reading the vocabularies from configs. Every slot value is a PREDICTION or unverified heuristic of unmeasured accuracy: a high score means
the pipeline wrote similar tokens, not that the shots look alike or that the film-makers intended a callback."""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cgt.tokens import v1
from cgt.tokens.format import PEOPLE_MAX

ROOT = Path(__file__).resolve().parents[3]
SLOTS = list(v1.SLOTS)  # size, lens, angle, framing, light, comp, key, DoF, people, sound, len, palette, move
STARRED = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")  # attributes that can carry `*`
STAR_FACTOR = 0.5
WINDOW = 2  # shots within +-2 positions (and the shot itself) are never candidates
K_DEFAULT = 5
S_DECIMALS = 12  # S is rounded to 12 decimals before ranking and tie detection (spec implementation note); float sum errors are ~1e-16
CATEGORICAL = {"angle": "angle", "framing": "framing", "comp": "composition", "sound": "sound"}  # slot -> column


def _schema(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / "configs/schema.yaml").read_text())["attributes"]


def ordinal_levels(root: Path = ROOT) -> dict[str, list]:
    """slot -> ordered levels. size/lens/DoF from schema.yaml, key in its schema order (low, mid, high), len = the 7 fixed bins of tokens.yaml, people 0..PEOPLE_MAX."""
    a = _schema(root)
    names = v1.load_config(root / "configs/tokens.yaml")["len_bins"]["names"]
    return {"size": list(a["size"]["values"]), "lens": list(a["lens"]["values"]), "key": list(a["key"]["values"]), "DoF": list(a["dof"]["values"]),
            "people": list(range(PEOPLE_MAX + 1)), "len": list(range(len(names)))}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v in ("", "?"))


def _codes(values, levels=None) -> np.ndarray:
    """Integer code per value (index in `levels`, or a stable code of the categorical value); -1 = missing / unknown."""
    if levels is not None:
        pos = {x: i for i, x in enumerate(levels)}
        return np.array([-1 if _missing(v) else pos.get(v, -1) for v in values], dtype=np.int64)
    seen: dict = {}
    return np.array([-1 if _missing(v) else seen.setdefault(str(v), len(seen)) for v in values], dtype=np.int64)


def _conf(df: pd.DataFrame, attr: str) -> np.ndarray:
    col = f"{attr}_consistent"
    if col not in df.columns:
        return np.ones(len(df))
    return np.array([STAR_FACTOR if isinstance(c, (bool, np.bool_)) and not c else 1.0 for c in df[col]], dtype=float)


def _len_codes(df: pd.DataFrame, root: Path) -> np.ndarray:
    if "len_bin" in df.columns:
        return np.array([-1 if _missing(v) else int(v) for v in df["len_bin"]], dtype=np.int64)
    edges = v1.load_config(root / "configs/tokens.yaml")["len_bins"]["edges"]
    return np.array([-1 if _missing(v) else v1.len_bin(v, edges) for v in df["len_s"]], dtype=np.int64)


def _palette_ids(df: pd.DataFrame) -> np.ndarray:
    if "palette_global" in df.columns:
        return np.array([-1 if _missing(v) else int(v) for v in df["palette_global"]], dtype=np.int64)
    out = []
    for t in df["palette_tok"]:
        out.append(int(t[1:]) if isinstance(t, str) and t[:1] == "g" and t[1:].isdigit() else -1)
    return np.array(out, dtype=np.int64)


def _move_masks(df: pd.DataFrame, atoms_order: list[str]) -> np.ndarray:
    pos = {a: i for i, a in enumerate(atoms_order)}
    out = []
    for xs in df["move_atoms"]:
        if xs is None or (isinstance(xs, float) and math.isnan(xs)) or len(xs) == 0:
            out.append(-1)
            continue
        m = 0
        for a in xs:
            m |= 1 << pos.setdefault(str(a), len(pos))
        out.append(m)
    return np.array(out, dtype=np.int64)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.int64)
    while np.any(x):
        c += (x & np.uint64(1)).astype(np.int64)
        x = x >> np.uint64(1)
    return c


def centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between palette centroids divided by their maximum (exactly symmetric, zero diagonal)."""
    c = np.asarray(centroids, dtype=np.float64)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    m = d.max()
    return d / m if m > 0 else d


@dataclass
class Features:
    shot_id: np.ndarray
    start_s: np.ndarray
    ordinal: dict          # slot -> (codes, span)
    categorical: dict      # slot -> codes
    conf: dict             # attr -> factor per shot (STARRED attrs)
    light: tuple           # (type codes, cond codes)
    palette: np.ndarray
    palette_d: np.ndarray | None
    move: np.ndarray       # bit masks, -1 missing
    move_used: bool

    @property
    def n_slots(self) -> int:
        return len(SLOTS) if self.move_used else len(SLOTS) - 1

    @property
    def palette_mode(self) -> str:
        return "exact_id" if self.palette_d is None else "centroid"


def features(df: pd.DataFrame, centroids: np.ndarray | None = None, root: Path = ROOT) -> Features:
    """Per-shot slot codes, ordered by shot_id. `centroids` None -> the palette slot is exact id match (palette_mode exact_id)."""
    d = df.sort_values("shot_id", kind="stable").reset_index(drop=True)
    lev = ordinal_levels(root)
    cols = {"size": "size", "lens": "lens", "key": "key", "DoF": "dof"}
    ordn = {s: (_codes(d[c], lev[s]), len(lev[s]) - 1) for s, c in cols.items()}
    ordn["people"] = (np.array([-1 if _missing(v) else min(int(v), PEOPLE_MAX) for v in d["people"]], dtype=np.int64), PEOPLE_MAX)
    ordn["len"] = (_len_codes(d, root), len(lev["len"]) - 1)
    cat = {s: _codes(d[c]) for s, c in CATEGORICAL.items()}
    conf = {a: _conf(d, a) for a in STARRED}
    light = (_codes(d["light_type"]), _codes(d["light_cond"]))
    move_used = "move_present" in d.columns and bool(d["move_present"].any()) and "move_atoms" in d.columns
    move = _move_masks(d, list(_schema(root)["move"]["values"])) if move_used else np.full(len(d), -1, dtype=np.int64)
    pd_ = None if centroids is None else centroid_distances(centroids)
    return Features(d["shot_id"].to_numpy(), d["start_s"].to_numpy(dtype=float), ordn, cat, conf, light, _palette_ids(d), pd_, move, move_used)


def slot_scores_row(f: Features, i: int) -> dict[str, np.ndarray]:
    """s_k(shot at position i, every shot) for each slot of the spec, in SLOTS order (confidence factors included)."""
    n = len(f.shot_id)
    out: dict[str, np.ndarray] = {}
    starred_attr = {"size": "size", "lens": "lens", "angle": "angle", "framing": "framing", "comp": "composition"}

    def conf(attr):
        c = f.conf[attr]
        return c * c[i]

    for s in SLOTS:
        if s in f.ordinal:
            o, span = f.ordinal[s]
            v = 1.0 - np.abs(o - o[i]) / span
            v[(o < 0) | (o[i] < 0)] = 0.0
        elif s in f.categorical:
            c = f.categorical[s]
            v = ((c == c[i]) & (c >= 0) & (c[i] >= 0)).astype(float)
        elif s == "light":
            t, c = f.light
            vt = ((t == t[i]) & (t >= 0) & (t[i] >= 0)).astype(float) * conf("light_type")
            vc = ((c == c[i]) & (c >= 0) & (c[i] >= 0)).astype(float) * conf("light_cond")
            v = 0.5 * vt + 0.5 * vc
        elif s == "palette":
            p = f.palette
            if f.palette_d is None:
                v = ((p == p[i]) & (p >= 0)).astype(float)
            else:
                v = np.zeros(n)
                ok = (p >= 0) & (p[i] >= 0)
                v[ok] = 1.0 - f.palette_d[p[i], p[ok]]
            if p[i] < 0:
                v[:] = 0.0
        elif s == "move":
            if not f.move_used:
                continue
            m = f.move
            inter = _popcount(m & m[i])
            union = _popcount(m | m[i])
            v = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            v[(m < 0) | (m[i] < 0)] = 0.0
        else:  # pragma: no cover
            raise KeyError(s)
        if s in starred_attr:
            v = v * conf(starred_attr[s])
        out[s] = v
    return out


def similarity_row(f: Features, i: int) -> np.ndarray:
    """S(shot i, every shot) = sum of the slot scores in SLOTS order / number of slots used, ROUNDED to S_DECIMALS decimals. Exactly symmetric.
    The rounding is an implementation note of the spec (definition unchanged): mathematically equal scores reached through different slot contributions
    (e.g. 34/39 via 1/6 and 1/2 steps) can differ in the last float bit; rounding makes them equal, so ranking and tie detection follow the smaller-id rule."""
    total = np.zeros(len(f.shot_id))
    for v in slot_scores_row(f, i).values():
        total = total + v
    return np.round(total / f.n_slots, S_DECIMALS)


def similarity_matrix(f: Features) -> np.ndarray:
    return np.stack([similarity_row(f, i) for i in range(len(f.shot_id))])


def candidates(n: int, i: int, window: int = WINDOW) -> np.ndarray:
    """Positions eligible as partners of position i: abs(p - i) > window (the shot itself and its neighbours are excluded)."""
    return np.flatnonzero(np.abs(np.arange(n) - i) > window)


def rank(f: Features, i: int, scores: np.ndarray, k: int = K_DEFAULT, window: int = WINDOW) -> np.ndarray:
    """Top-k candidate positions: S descending, ties to the smaller shot_id."""
    c = candidates(len(f.shot_id), i, window)
    order = np.lexsort((f.shot_id[c], -scores[c]))
    return c[order[:k]]


def top_k_table(df: pd.DataFrame, k: int = K_DEFAULT, centroids: np.ndarray | None = None, window: int = WINDOW, root: Path = ROOT) -> pd.DataFrame:
    """Long table, k rows per shot: shot_id, rank (1..k), other_shot_id, score, start_s, other_start_s, gap_min, tie_next (score equals the next candidate's score)."""
    f = features(df, centroids, root)
    rows = []
    for i in range(len(f.shot_id)):
        s = similarity_row(f, i)
        top = rank(f, i, s, k + 1, window)
        for r, j in enumerate(top[:k]):
            nxt = s[top[r + 1]] if r + 1 < len(top) else None
            rows.append({"shot_id": int(f.shot_id[i]), "rank": r + 1, "other_shot_id": int(f.shot_id[j]), "score": float(s[j]),
                         "start_s": float(f.start_s[i]), "other_start_s": float(f.start_s[j]), "gap_min": abs(float(f.start_s[j] - f.start_s[i])) / 60.0,
                         "tie_next": nxt is not None and float(nxt) == float(s[j])})
    out = pd.DataFrame(rows)
    out.attrs.update({"palette_mode": f.palette_mode, "n_slots": f.n_slots, "move_used": f.move_used})
    return out


def similar_to(df: pd.DataFrame, shot_id: int, k: int = K_DEFAULT, centroids: np.ndarray | None = None, window: int = WINDOW, root: Path = ROOT) -> pd.DataFrame:
    """Top-k of one shot (other_shot_id, score), same rules as top_k_table."""
    f = features(df, centroids, root)
    i = int(np.flatnonzero(f.shot_id == shot_id)[0])
    s = similarity_row(f, i)
    top = rank(f, i, s, k, window)
    return pd.DataFrame({"rank": np.arange(1, len(top) + 1), "other_shot_id": f.shot_id[top], "score": s[top], "other_start_s": f.start_s[top]})


# ---- pre-declared descriptive analysis (spec section "Pre-declared descriptive analysis") ------------------------------------------------------------

GAP_BINS = [0, 1, 5, 10, 30, 60, math.inf]
GAP_BIN_NAMES = ["0-1", "1-5", "5-10", "10-30", "30-60", "60+"]


def gap_stats(gaps) -> dict:
    g = np.asarray(gaps, dtype=float)
    hist = np.histogram(g, bins=GAP_BINS)[0]
    return {"n": len(g), "min": float(g.min()), "q25": float(np.quantile(g, 0.25)), "median": float(np.median(g)), "q75": float(np.quantile(g, 0.75)),
            "mean": float(g.mean()), "max": float(g.max()), **{f"share_gt_{t}": float((g > t).mean()) for t in (1, 5, 10, 30)},
            "hist": dict(zip(GAP_BIN_NAMES, hist.tolist(), strict=True))}


def all_candidate_gaps(start_s: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Gaps (minutes) of every ordered candidate pair (a, b), abs(p(b) - p(a)) > window; start_s in shot order."""
    s = np.asarray(start_s, dtype=float)
    n = len(s)
    idx = np.arange(n)
    m = np.abs(idx[:, None] - idx[None, :]) > window
    return (np.abs(s[:, None] - s[None, :]) / 60.0)[m]


def long_range_pairs(top: pd.DataFrame, min_gap: float = 10.0, n: int = 10) -> pd.DataFrame:
    """Top-1 pairs with gap > min_gap as unordered pairs (counted once), S descending, ties by (smaller id, larger id)."""
    t = top[(top["rank"] == 1) & (top["gap_min"] > min_gap)].copy()
    t["a"] = t[["shot_id", "other_shot_id"]].min(axis=1)
    t["b"] = t[["shot_id", "other_shot_id"]].max(axis=1)
    t["mutual"] = t.duplicated(["a", "b"], keep=False)
    t = t.drop_duplicates(["a", "b"])
    return t.sort_values(["score", "a", "b"], ascending=[False, True, True], kind="stable").head(n).reset_index(drop=True)


def mutual_top1_share(top: pd.DataFrame) -> float:
    t1 = top[top["rank"] == 1]
    m = dict(zip(t1.shot_id, t1.other_shot_id, strict=True))
    return float(np.mean([m.get(b) == a for a, b in m.items()]))
