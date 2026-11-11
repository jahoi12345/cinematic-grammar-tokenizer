"""Visual-arc series of a film from its token table (Phase 6, analysis #1). Pure pandas/numpy. Series are windowed summaries of PREDICTIONS or unverified heuristics:
nothing here is accuracy and nothing is tuned on the film."""
import numpy as np
import pandas as pd

SIZE_ORDER = ["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS"]  # ordinal, tight to wide (configs/schema.yaml)
CLOSE = {"ECU", "CU", "MCU"}
# series name -> (kind, function of the window's shot rows -> per-shot value whose mean is the series), kind marks the source
SERIES = {
    "cuts_per_min": ("heuristic", None),
    "median_len_s": ("heuristic", None),
    "close_share": ("prediction", lambda d: d["size"].isin(CLOSE).astype(float)),
    "mean_size_ordinal": ("prediction", lambda d: d["size"].map(SIZE_ORDER.index).astype(float)),
    "low_key_share": ("heuristic", lambda d: (d["key"] == "low-key").astype(float)),
    "people_mean": ("heuristic", lambda d: d["people"].clip(upper=4).astype(float)),
    "multi_person_share": ("heuristic", lambda d: (d["people"] >= 2).astype(float)),
    "dialogue_share": ("heuristic", lambda d: (d["sound"] == "dialogue").astype(float)),
    "high_contrast_share": ("prediction", lambda d: (d["light_cond"] == "high-contrast").astype(float)),
    "daylight_share": ("prediction", lambda d: (d["light_type"] == "daylight").astype(float)),
    "low_angle_share": ("prediction", lambda d: (d["angle"] == "low").astype(float)),
    "keyframes_all_agree_share": ("prediction-stability", lambda d: d["kf_all_agree"].astype(float)),
}


def windows(df: pd.DataFrame, minutes: float = 5.0) -> pd.Series:
    """Window index of each shot by its start time (the window width in minutes)."""
    return (df["start_min"] // minutes).astype(int)


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0, q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """Percentile bootstrap of the mean over shots (a resampling interval of the window's own shots; it says nothing about label accuracy)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, q)
    return float(lo), float(hi)


def series_table(df: pd.DataFrame, minutes: float = 5.0, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Long table: window, window_start_min, n_shots, series, kind, value, ci_lo, ci_hi."""
    w = windows(df, minutes)
    film_end = float(df["end_s"].max() / 60.0) if "end_s" in df.columns else float("inf")
    rows = []
    for k, d in df.groupby(w):
        span = max(min(minutes, film_end - k * minutes), 1e-9)  # the last window is usually partial: rates use its real duration
        for name, (kind, fn) in SERIES.items():
            if name == "cuts_per_min":
                v, lo, hi = len(d) / span, float("nan"), float("nan")
            elif name == "median_len_s":
                v = float(d["len_s"].median())
                lo, hi = float(np.percentile(d["len_s"], 25)), float(np.percentile(d["len_s"], 75))  # interquartile range, not a CI
            else:
                x = fn(d).to_numpy()
                v, (lo, hi) = float(x.mean()), bootstrap_ci(x, n_boot, seed)
            rows.append({"window": int(k), "window_start_min": k * minutes, "n_shots": len(d), "series": name, "kind": kind, "value": v, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def spearman_with_time(tbl: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation of each series with window position (descriptive: a drift indicator, not a significance test)."""
    out = []
    for name, d in tbl.groupby("series"):
        out.append({"series": name, "kind": d["kind"].iloc[0], "spearman_with_time": float(d["value"].corr(d["window_start_min"], method="spearman"))})
    return pd.DataFrame(out)


MOVE_GROUPS = {"pan": ["pan-left", "pan-right"], "tilt": ["tilt-up", "tilt-down"], "push/pull": ["push-in", "pull-out"], "lateral": ["move-left", "move-right"],
               "boom": ["boom-up", "boom-down"], "zoom": ["zoom-in", "zoom-out"]}


def movement_series_table(df: pd.DataFrame, minutes: float = 5.0, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Movement series (PREDICTIONS of the Phase 4 geometry model, accuracy on the film unmeasured) per window: share of shots whose predicted set contains `static`, share that are
    exactly `static` alone, mean set size, and the share containing any atom of each group. Same long format and interval conventions as `series_table` (bootstrap for shares
    and the mean; no interval is claimed to measure accuracy). Requires the `move_atoms` list column (movement table present)."""
    if "move_atoms" not in df.columns or df["move_atoms"].isna().all():
        raise ValueError("no movement table in this token table")
    w = windows(df, minutes)
    fns = {"move_static_share": lambda d: d["move_atoms"].map(lambda a: "static" in a).astype(float),
           "move_static_alone_share": lambda d: d["move_atoms"].map(lambda a: list(a) == ["static"]).astype(float),
           "move_mean_set_size": lambda d: d["move_atoms"].map(len).astype(float)}
    for g, atoms in MOVE_GROUPS.items():
        fns[f"move_{g}_share"] = lambda d, atoms=atoms: d["move_atoms"].map(lambda a: bool(set(atoms) & set(a))).astype(float)
    rows = []
    for k, d in df.groupby(w):
        for name, fn in fns.items():
            x = fn(d).to_numpy()
            lo, hi = bootstrap_ci(x, n_boot, seed)
            rows.append({"window": int(k), "window_start_min": k * minutes, "n_shots": len(d), "series": name, "kind": "prediction (movement)", "value": float(x.mean()), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def contradictory_share(atoms_series: pd.Series) -> float:
    """Share of shots whose predicted set is CONTRADICTORY by the declared definition of docs/phase4_film_move.md (an opposite pair, or `static` with another atom); the definition is
    imported from the film runner library, not re-implemented."""
    from cgt.p4_camera_move.film_move import is_contradictory
    return float(np.mean([is_contradictory(a)["contradictory"] for a in atoms_series]))


def contradictory_series(df: pd.DataFrame, minutes: float, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per-window share of contradictory sets (bootstrap interval over the window's shots); same long format as `series_table`."""
    from cgt.p4_camera_move.film_move import is_contradictory
    w = windows(df, minutes)
    rows = []
    for k, d in df.groupby(w):
        x = d["move_atoms"].map(lambda a: float(is_contradictory(a)["contradictory"])).to_numpy()
        lo, hi = bootstrap_ci(x, n_boot, seed)
        rows.append({"window": int(k), "window_start_min": k * minutes, "n_shots": len(d), "series": "move_contradictory_share", "kind": "prediction (movement)", "value": float(x.mean()), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
